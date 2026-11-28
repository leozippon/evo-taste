# Scale scout: larger base models (working notes, 27–28 Nov 2026)

Working notes of the feasibility scout for moving off Qwen3-1.7B. Facts, commands, measurements and failures are appended as they happen; the final report is at the end once written. Scratch scripts and raw results live in `logs/scout-models/` (not in git): `sample.py` (vLLM start, memory, throughput, base probe accuracy, saved samples), `train.py` (the engine's `Policy.update`/`logprobs` on a scout layout: agreement with the sampler, update cost), `fp8_model.py` (FP8-storage trainer loader), `scout_common.py` (random adapters in the engine's format, GPU memory per process), `results/<tag>/` (reports).

## Part A — inventory (no GPU)

Installed stack: vLLM 0.34.0, torch 2.14.0+cu130, transformers 5.22.2, PEFT 0.22.0, accelerate 1.16.0, compressed-tensors 0.17.0, flashinfer 0.7.3. Not installed: bitsandbytes, `kernels` (the hub-kernel loader transformers needs for its own FP8 Linear and for the fast FLA / causal-conv1d kernels of Qwen3.5-architecture models), flash-linear-attention, causal-conv1d, auto-gptq/awq.

Models on the machine (chat/instruct, ≥ 4B):

| Model | Where | Text params | On disk | Format | Layers × hidden | Tokenizer / template | Thinking off |
|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | project HF cache | 4.0B (tied) | 7.6 GB | bf16 | 36 × 2560 | Qwen3 (151,669) | non-thinking model; template has no think block and ignores `enable_thinking` |
| Qwen3.8-27B-FP8 (official) | `/Data2/lzp/models` | 26.9B | 29 GB | block-128 e4m3, embeddings/head/norms/`in_proj_a,b` bf16 | 64 × 5120, hybrid: 48 Gated DeltaNet + 16 gated full attention | Qwen3.5 (248,320) | `enable_thinking=False` gives `<think>\n\n</think>\n\n`, as Qwen3 |
| Qwen3.8-27B-Uncensored-FP8 (third party, abliterated) | `/Data2/lzp/models` | same | 29 GB | same | same | same | same |
| Qwen3Guard-Gen-8B | `/Data2/lzp/models` | 8.2B | 16 GB | bf16 | 36 × 4096 | Qwen3 | a safety classifier, not a candidate |

Mirror (all ungated, safetensors): Qwen/Qwen3.8-27B 55.6 GB bf16, Qwen/Qwen3.8-27B-FP8 30.9 GB; the Qwen3.8 generation has no other dense size (Qwen3.8-Flash-Next is 360 GB, Qwen3.8-2.4T-A95B). Same architecture, smaller: Qwen3.5-9B 19.3 GB, Qwen3.5-4B 9.3 GB (also Qwen3.5/3.6-27B, 35B-A3B MoE). Qwen3 dense: 8B 16.4 GB, 14B 29.5 GB, 32B 65.5 GB. Downloaded (project HF cache): Qwen/Qwen3.8-27B (bf16) and Qwen/Qwen3-14B, 85 GB together.

Architecture consequences for the engine (Qwen3.8 / Qwen3.5 family):

- The engine's seven LoRA target names reach only the MLPs (all 64 layers) and q/k/v/o of the 16 full-attention layers (159M params at rank 32); the DeltaNet projections are `in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj` (all twelve names: 234M params).
- Linear-attention layers cannot take the engine's 4D packing mask, so `trainer.shared_prefix` must be false.
- vLLM's `Qwen3_5ForConditionalGeneration` expects adapter keys `model.language_model.layers...`; PEFT on transformers' `Qwen3_5ForCausalLM` names them `model.layers...`, so `save_adapter` would need a rename.
- Without `kernels`/FLA, transformers runs the DeltaNet recurrence in its pure-torch fallback (speed measured below).
- vLLM supports LoRA for Qwen3.5 (packed `in_proj_qkvz`, `in_proj_ba`), LoRA over quantised base layers (the LoRA wrapper calls the base layer's quant method) and LoRA under tensor parallelism (`--fully-sharded-loras` optional); `--generation-config vllm` still takes the eos list (`<|im_end|>`) from `generation_config.json`.
- transformers 5.22 can train LoRA over its own FP8 Linear (`FP8LinearFunction` has an input backward), but only with the `kernels` package; the scout uses its own loader instead (`fp8_model.py`: FP8 weights stay on the GPU, dequantised to bf16 per call, recomputed in backward).

Memory arithmetic, 46 GB (≈ 44.5 GiB usable) cards. Weights in GiB (bf16): 4B 7.5, 8B 15.3, 14B 27.5, 27B 50.1 (FP8: 27.4), 32B 61.0. Sampler ≈ weights + KV budget + about 3.5 GiB + LoRA slots (8 × 0.5 GiB at 27B); trainer ≈ weights + 16 bytes per LoRA parameter + activations (several GiB at 8k tokens).

| Layout | fits |
|---|---|
| sampler + trainer on one card (today) | 4B yes; 8B marginal (no headroom); 14B, 27B no |
| sampler on one card, trainer on another | 8B, 14B yes (14B sampler with a KV budget ≤ 10G); 27B only with 8-bit weights on both sides (bf16 50 GiB does not fit a card) |
| sampler TP2 and trainer split over the same two cards | 14B yes (tight); 27B no in bf16 (2 × 25 GiB per card) |
| 27B in bf16 | sampler TP2 on two cards + trainer over two more = one worker per four cards |

## Part B — measurements

(appended below as they are made)

### Log of runs

- 01:2x PAIRED-DONE seen; GPUs 2 and 7 free apart from a 2.7 GB process of another user on each.
- 01:30 27B-FP8 sampler, GPU 7, KV 8G, 8 LoRA slots, max_num_seqs 512: vLLM refused. Weights plus LoRA buffers took 32.1 GiB; the 8G cache gave 80k attention tokens but only 157 Mamba (DeltaNet state) blocks, one per running sequence, and vLLM requires max_num_seqs ≤ that. On this model the cache budget limits concurrency through the recurrent state (~51 MB per sequence), not through attention tokens.
- 01:35 retry with 6 slots, max_num_seqs 160 failed at startup: only 9.4 GiB free on GPU 7, because a 32.8 GB training job of the owner's other project (socio-evo, `learning.train ... jobs-01-gpu7.json`) started on GPU 7 at 01:33. GPU 7 is not usable while it runs; the scout continues on GPU 2 alone.
- 01:44–01:48 three more 27B-FP8 starts on GPU 7 failed: max_num_seqs 160 > 157 Mamba blocks at KV 8G; then max_num_seqs 152 at KV 8G ran out of memory while capturing CUDA graphs (weights + 6 LoRA slots 31.1 GiB, 8 GiB cache, ~41.4 GiB free at start, beside another user's 2.7 GB process).
- 01:49 (note for the coordinator) a 27B-FP8 sampler run of the scout is running on GPU 7 (KV 6G, max_num_seqs 112, 6 LoRA slots), started before GPU 7 was reassigned; it is left to finish (throughput then the 357-item probe at 4,096 tokens, roughly 30–40 minutes) and nothing further of the scout will start on GPU 7.
- 01:49 the KV 6G / max_num_seqs 112 server on GPU 7 started (ready in 85 s; 40.3 GB on the card; weights + 6 LoRA slots 31.1 GiB; 60,416 attention tokens). With prefix caching on (vLLM's default), vLLM sets the Mamba cache mode to `align` and the attention block size to 832 tokens; under 800 queued sequences only 27–28 ran at once with the cache 97–98% full, at 320–350 generated tokens/s. That is unusable; the run was stopped at 01:57 (GPU 7 released for the census) and is redone on GPU 2 with `--no-enable-prefix-caching`.
- Calibration of the scout's bench against the engine (Qwen3-1.7B, GPU 2): vLLM 4,923 completion tokens/s (800 sequences ≤ 1,024 tokens, mean 433); trainer update 2,311 completion tokens/s (128 rollouts, mean 409 tokens, peak 7.05 GiB), within the engine's measured 2,500–4,400 per second; 4 × 4,600-token sequences 1,634 tokens/s, peak 6.6 GiB. Agreement on 64 rollouts: per-token mean |Δ| 0.0115, Σ|ΔΣ|/tokens 0.0013 (the engine's check thresholds are 0.04 and 0.015); base model against the adapter's samples 0.083 and 0.030, so the random adapter (B ~ N(0, 0.01)) is well separated.
- Qwen3-4B-Instruct-2507 (GPU 2, KV 12G, one card): ready 176 s, 24.6 GB on the card; 2,093 completion tokens/s on 800 sequences ≤ 1,024 (mean 674, 36% hit the cap); probe set (357 items, 4,096 tokens, one sample, the base references' seeds): accuracy 0.713 (code 0.72, logic 0.70, math 0.72) against 0.416 for 1.7B, compliance 0.91, 9% truncated, mean 1,088 tokens, generation 329 s at 1,180 tokens/s. Trainer on one card: agreement 0.0140 / 0.0012 (base 0.080 / 0.025); update 1,162 completion tokens/s (128 rollouts, mean 666), peak 12.4 GiB; 4 × 4,600 tokens 1,112 tokens/s, peak 11.7 GiB. Sampler (24.6 GB) and trainer (13.2 GB reserved) fit one card together.

### 27B FP8 serving (for the census): what works on one 46 GB card

Command (vLLM 0.34, the engine's flags plus): `vllm serve /Data2/lzp/models/Qwen3.8-27B-FP8 --language-model-only --no-enable-prefix-caching --kv-cache-memory-bytes 9G --gpu-memory-utilization 0.8 --max-model-len 8192 --generation-config vllm --enable-lora --max-lora-rank 32 --max-loras 2 --max-num-seqs 64 --max-num-batched-tokens 8192`. Ready in 75–85 s with warm caches; weights + 2 LoRA slots 28.96 GiB (6 slots: 31.1 GiB); 40.4 GB on the card. A rank-32 adapter on all twelve projection names (keys `base_model.model.model.language_model.layers.N...`) loads in about 1 s. Without `--enable-lora` the footprint is smaller by the slots (about 0.4 GiB each).

Concurrency is set by the DeltaNet state, not by attention tokens: each running sequence holds one ~52 MB state page per group of 16 DeltaNet layers (three groups) plus one 832-token attention page per 832 tokens, so the cache budget buys about 4 pages per short sequence. Measured: KV 6G → 27–29 sequences running; KV 9G → 37–40 sequences (attention capacity reported as 111,537 tokens, which is misleading); 8G gave 157 state pages and vLLM refuses `max_num_seqs` above the page count. With prefix caching on (the default) vLLM switches the state to `align` mode, which costs more pages; turn it off. CUDA-graph capture needs about 2–3 GiB beyond weights + cache (KV 8G with 6 slots and 152 sequences ran out of memory while capturing).

Throughput with thinking disabled (engine prompts, temperature 1, through the adapter, 800 queued requests capped at 1,024 tokens): about 350–420 generated tokens/s at 28–40 running sequences, i.e. about 10 tokens/s per sequence. Final figures and a bf16-state variant below.

Measured on GPU 2 (02:03–02:42), 800 requests (200 probe prompts × 4) through the adapter, completions ≤ 1,024 tokens, thinking disabled:

| Config | Running sequences | Generated tokens/s (whole batch, incl. tail) | Mean length / at cap | Card memory |
|---|---|---|---|---|
| KV 9G, 2 LoRA slots, `max_num_seqs 64`, state fp32 (default) | 37–40 | 443 (wall 1,229 s) | 681 / 35% | 41.5 GB |
| same + `--mamba-ssm-cache-dtype bfloat16`, `max_num_seqs 128` | 44–75 | 607 (wall 901 s), 610–810 while the queue is full | 684 / 37% | 43.7 GB |

The bf16 state roughly doubles the sequences that fit; whether it moves the log-probabilities is checked against the trainer below. Answers at this cap are long: a third of them would exceed 1,024 tokens.

Base capability on the 357-item probe set (4,096 tokens, one sample, the base references' seeds; family-macro accuracy as the engine reports it, then the plain item mean):

| Model | Macro | Item mean | code / logic / math (item mean) | Mean / median tokens | Truncated | On items 1.7B failed / solved |
|---|---|---|---|---|---|---|
| Qwen3-1.7B (pilot C base) | 0.416 | 0.524 | 0.42 / 0.57 / 0.47 | 511 / 342 | 0.3% | 0 / 1 |
| Qwen3-4B-Instruct-2507 | 0.713 | 0.751 | 0.72 / 0.79 / 0.68 | 1,088 / 617 | 9.0% | 0.59 / 0.89 |
| Qwen3.8-27B-FP8, thinking off (bf16 state) | 0.688 | 0.784 | 0.64 / 0.85 / 0.70 | 1,134 / 653 | 7.6% | 0.64 / 0.91 |

By answer length the 27B solves 0.93 of answers under 512 tokens, 0.85 at 512–1,024, 0.80 at 1,024–2,048 and 0.38 beyond (64 items, most of them truncated). The 1.7B census strata do not carry over: the 27B (and the 4B) solve about 60% of the items the 1.7B failed, so a larger model needs its own census. Generation of the probe set took 684 s for the 27B (592 tokens/s) against 329 s for the 4B.

### 27B FP8 trainer on one card and its agreement with the FP8 sampler

Loader: `logs/scout-models/fp8_model.py` builds transformers' `Qwen3_5ForCausalLM` on the meta device, replaces the 400 quantised Linear layers by an `nn.Linear` subclass that keeps the e4m3 weight and its 128×128 block scales on the GPU and dequantises to bf16 inside an autograd function (input gradient through the re-dequantised weight; weights frozen), and copies every text tensor from the directory's `layers-N.safetensors`/`outside.safetensors` (vision and MTP skipped; it refuses if any parameter is left unloaded). PEFT wraps these layers like any Linear; the LoRA parameters stay fp32. Loading takes 11 s; weights + LoRA state (rank 32 on all twelve names, 234M parameters) + optimizer = 31.9 GiB peak while scoring. The engine's `Policy.logprobs`/`update` run unchanged on it (shared-prefix rows off, see above).

Agreement on the first rollout of each of 64 prompts (46k completion tokens), the engine's two measures (per-token mean |Δ|; Σ|ΔΣ|/tokens):

| Sampler | Trainer | adapter vs sampler | base vs the adapter's samples |
|---|---|---|---|
| vLLM FP8, fp32 DeltaNet state | FP8 storage, bf16 activations | 0.0096; 0.00095 | 0.043; 0.0083 |
| vLLM FP8, bf16 state | same | 0.0089; 0.00097 | 0.042; 0.0085 |
| either | same + emulated per-token-group FP8 activations | 0.0152; 0.0014–0.0017 | 0.044; 0.0086–0.0089 |

So the FP8 sampler and an FP8-storage bf16-activation trainer agree as closely as the 1.7B's bf16 pair (0.0115; 0.0013 in the same bench), and the bf16 recurrent state costs nothing measurable; emulating activation quantisation in the trainer makes it worse, so vLLM is evidently not quantising activations this way on the L20. Caveat: at this adapter scale (B ~ N(0, 0.01)) the base model differs from the adapter by only 0.0085 on the second measure, under the engine's 0.015 threshold, so the check's thresholds would need recalibrating for the 27B (the gap to the base grows as members train; the scout did not measure trained members). Forward-only scoring ran at about 1,020 row tokens/s.

Update cost, first attempt (token budget 8,192 per micro-batch, gradient checkpointing): 32 rollouts (18.7k completion tokens, mean 583) took 103 s, 181 completion tokens/s, peak 41.5 GiB allocated / 44.0 GiB reserved; the next update of 96 rollouts ran out of memory (38.7 GiB allocated, 4.8 GiB reserved but fragmented). Retried with token budget 4,096 and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (below).

Retry (token budget 4,096, expandable segments): 32 rollouts in 95 s (195 completion tokens/s, peak 35.6 / 39.3 GiB); 96 rollouts (65k completion tokens, mean 680, 25 micro-batches) in 322 s, 203 completion tokens/s, peak 37.3 GiB allocated / 39.7 reserved; 4 rows of 4,600 tokens in 119 s (155 tokens/s), peak 38.0 / 40.1 GiB. So the 27B trains on one card with FP8 storage up to at least 4,600-token rows, at about a tenth of the 1.7B's speed per token (2,311 in the same bench). The time is compute in the bf16 path plus transformers' pure-torch DeltaNet fallback (no FLA kernels installed), not memory; installing flash-linear-attention / causal-conv1d (not tried) is the obvious lever.

Two-card 27B layout, measured parts: sampler card (FP8, bf16 state, KV 9G, 2–6 LoRA slots) ≈ 600–800 generated tokens/s at full queue, ~10–15 tokens/s per sequence; trainer card (FP8 storage) ≈ 200 completion tokens/s per update, ≈ 1,000 tokens/s scoring. Estimated cost of one member step with answers like the probe's (mean ~1,100 tokens at a 4,096 cap) and about 60% of rollouts in the loss (as in pilot C): 48 × 4 = 211k generated tokens ≈ 5.5 min of sampler time (a 4,096-token straggler alone needs ~4–7 min at 10–15 tokens/s, so a step is never shorter) and ≈ 127k trained tokens ≈ 10.5 min of trainer time; 24 × 4 about half of each (≈ 2.7 + 5.3 min, straggler floor unchanged). Members' generation is shared and the engine runs generation and updates in sequence, so a worker of M members needs about M × (5.5 + 10.5) min per step at 48 × 4 and M × 8 min at 24 × 4.

### Qwen3-14B (bf16, GPU 2, sampler then trainer on the same card in sequence)

Sampler (KV 10G = 65,536 tokens, 6 LoRA slots, engine prompts with the empty think block): ready 181 s, 44.0 GB on the card; 1,264 completion tokens/s on 800 sequences ≤ 1,024 (mean 498, 15% at the cap). Probe set: macro 0.616, item mean 0.692 (code 0.61, logic 0.73, math 0.64), compliance 0.97, nothing truncated, mean 558 tokens, generation 257 s at 774 tokens/s; it solves 0.47 of the items the 1.7B failed. With thinking off the 14B is below Qwen3-4B-Instruct-2507 on this set, and its answers are half as long. Trainer (one card, bf16, shared-prefix rows as the engine): agreement 0.0136 / 0.0012 (base 0.031 / 0.0040: again too close to the engine's threshold to detect a swap at this adapter scale); update of 128 rollouts (62k completion tokens, mean 487) in 111 s, 563 completion tokens/s, peak 36.2 / 38.7 GiB; 4 × 4,600-token rows 426 tokens/s, peak 33.7 GiB. Sampler (44 GB) and trainer (39 GB) need a card each.

### 27B FP8 member-step generation, measured (GPU 2, 03:35–03:53)

One server (KV 9G, bf16 state, 2 LoRA slots, max_num_seqs 128), one adapter, n = 4 per prompt, 4,096-token cap: 24 prompts × 4 in 229 s (70k tokens, mean 726, one answer at the cap, 304 tokens/s); 48 × 4 in 443 s (217k tokens, mean 1,128, 15 at the cap, 490 tokens/s) and in 379 s on another draw (173k, mean 899, 4 at the cap). A step's generation is thus 4–7.5 minutes per member alone; several members generated together approach the full-queue rate of about 600 tokens/s.

## Final summary (03:55)

Per-card rates measured in the same bench (thinking off, rank-32 LoRA on all projections, gradient checkpointing):

| Model, layout | Sampler tok/s (800 seqs ≤ 1,024) | Trainer completion tok/s | Probe macro / item mean | Mean answer (4,096 cap) |
|---|---|---|---|---|
| Qwen3-1.7B, one card | 4,923 | 2,311 | 0.416 / 0.524 | 511 |
| Qwen3-4B-Instruct-2507, one card for both | 2,093 | 1,162 | 0.713 / 0.751 | 1,088 |
| Qwen3-14B bf16, a card each | 1,264 | 563 | 0.616 / 0.692 | 558 |
| Qwen3.8-27B FP8, a card each | 443 (fp32 state) / 607 (bf16 state) | 203 | 0.688 / 0.784 | 1,134 |

Not measured (needs two cards at once, or not downloaded): 27B bf16 with tensor parallelism and the trainer split over two cards, 14B sampler and trainer on separate cards under load, any 8–9B model, the trained-member log-probability gap, FLA kernels for the trainer.
