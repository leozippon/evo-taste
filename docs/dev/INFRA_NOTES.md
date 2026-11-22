# Infrastructure notes

Hand-off for the training engine: how the environment is built, how vLLM serves many LoRA adapters next to a PyTorch/PEFT trainer on one L20 (46 GB), what was verified, what it costs, and the pitfalls found on 2026-11-22. Scripts referenced here live in `scripts/dev/` and write to `runs/dev/`, `results/` and `logs/` on the compute host.

## Environment

Activate with `source scripts/env.sh` from any directory. It activates `.venv`, exports the variables in `.env` (HF token, HF mirror endpoint, GitHub token), and sets `HF_HOME=/Data2/lzp/hf-home`, `UV_CACHE_DIR=/Data2/lzp/uv-cache`, `HF_HUB_DISABLE_XET=1` (Xet downloads stall through the mirror), `VLLM_NO_USAGE_STATS=1`, and `CUDA_HOME=/usr/local/cuda-13.2` with its `bin` first on `PATH` (see pitfalls). Rebuild the venv with `uv sync` after sourcing; `pyproject.toml` pins only `vllm==0.34.0` and lets vLLM constrain torch and transformers, and `uv.lock` pins everything else.

The shared Hugging Face cache is `/Data2/lzp/hf-home` (layout `hub/`, `datasets/`); it holds `Qwen/Qwen3-4B-Instruct-2507`, the downloaded `Qwen/Qwen3-1.7B` (snapshot `70d244cc`), and the datasets. `/Data2/lzp/models` is not an HF cache: it only holds the owner's local 27B FP8 checkpoints.

| Component | Version |
|---|---|
| Python | 3.12.16 (uv-managed; vLLM's reference version) |
| torch | 2.14.0+cu130 (pinned by vLLM) |
| vLLM | 0.34.0 (released 2026-11-16) |
| FlashInfer | 0.7.3 |
| transformers | 5.22.2 (vLLM requires <5.23) |
| peft / accelerate / datasets | 0.22.0 / 1.16.0 / 5.2.0 |
| reasoning-gym | 0.1.25 (its `__version__` string wrongly says 0.1.19) |
| numpy / scipy / pandas / scikit-learn | 2.3.5 / 1.18.1 / 3.1.1 / 1.9.1 |
| httpx / openai / pytest | 0.28.1 / 3.57.0 / 9.2.0 |
| NVIDIA driver | 595.58.03 (CUDA 13.2) |

A newer small Qwen exists, `Qwen/Qwen3.5-2B` (2026-02), but it is a hybrid Gated-DeltaNet/attention vision-language model (`Qwen3_5ForConditionalGeneration`, 248k vocabulary), not a plain dense causal LM, so neither vLLM's LoRA path nor the PEFT trainer here was validated for it; it was not downloaded.

## vLLM server

`CUDA_VISIBLE_DEVICES=<gpu> scripts/dev/serve_vllm.sh <model> <port> <kv-size> [extra flags]` runs, in the foreground:

```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve <model> --host 127.0.0.1 --port <port> \
  --kv-cache-memory-bytes <kv-size> --max-model-len 8192 \
  --generation-config vllm --enable-prompt-tokens-details \
  --enable-lora --max-lora-rank 64 --max-loras 8 --max-cpu-loras 64 \
  --max-num-seqs 512 --max-num-batched-tokens 8192
```

The last line is passed as extra flags; without it vLLM uses its defaults for GPUs under 70 GB (`max_num_seqs=256`, `max_num_batched_tokens=2048`). `--kv-cache-memory-bytes` replaces `--gpu-memory-utilization` for sizing: it reserves a fixed KV budget and skips memory profiling, which makes the footprint predictable next to a trainer. It does not exempt the server from vLLM's startup check that free memory is at least `gpu_memory_utilization` × card (default 0.92), so a server started beside a loading trainer or any other process holding about 3.5 GiB dies with "Free memory on device … is less than desired GPU memory utilization". The engine therefore also passes `--gpu-memory-utilization 0.65`, about the server's real footprint (0.61 of an L20 at KV 20G); with a fixed KV budget the value is used only for that check. `--generation-config vllm` stops vLLM from applying the checkpoint's `generation_config.json` sampling defaults (Qwen3: `top_k=20, top_p=0.95`) to every request that does not override them. `--max-loras` is the number of adapters that can share one batch (GPU slots, about 0.1 GiB each for 1.7B and 0.3 GiB for 4B at rank 64); `--max-cpu-loras` is the LRU cache of loaded adapters. A request for a ninth distinct adapter waits until a slot frees, so raise `--max-loras` if more adapters per GPU must decode concurrently. Prefix caching is on by default, with 64-token blocks.

The first start on a fresh machine takes about 6 minutes (torch.compile, a 3-minute FlashInfer JIT build, CUDA-graph capture); later starts take 3.5–4 minutes. Measured footprints of the whole vLLM process (`nvidia-smi`), with the fraction of the card's 45.0 GiB:

| Model | KV budget | Scheduler | KV capacity | Footprint |
|---|---|---|---|---|
| Qwen3-1.7B | 12G | defaults (256 seqs, 2048 batched tokens) | 112k tokens | 18.9 GiB (0.42) |
| Qwen3-1.7B | 20G | 512 seqs, 8192 batched tokens | 187k tokens | 27.4 GiB (0.61) |
| Qwen3-4B-Instruct-2507 | 12G | 512 seqs, 8192 batched tokens | 87k tokens | 25.5 GiB (0.57) |

Beyond the KV budget, vLLM holds the weights plus LoRA slot buffers (4.55 GiB for 1.7B, 9.96 GiB for 4B at 8 slots of rank 64), CUDA graphs (0.9–1.5 GiB) and about 2–3 GiB of activations, sampler buffers and CUDA context.

## LoRA hot reload

Save every adapter version to a new directory and load it under a new name, for example `ind07@v12`, with `POST /v1/load_lora_adapter {"lora_name": "ind07@v12", "lora_path": "/abs/dir"}`; send new requests to the new name; unload the previous name (`POST /v1/unload_lora_adapter {"lora_name": "ind07@v11"}`) once its in-flight requests have finished. This is the procedure to use. It never changes the weights under a request that is running, and it keeps the prefix cache correct, because vLLM keys prefix-cache blocks by (base tokens, adapter name, adapter path).

`scripts/dev/lora_smoke.py` tested six ways to replace an adapter whose prompts were already in the prefix cache (Qwen3-1.7B). Which weights are served was measured with teacher-forced prompt logprobs, which are always computed from scratch; stale reuse was measured with `usage.prompt_tokens_details.cached_tokens` on the first request after the switch.

| Procedure | New weights served | Old-weight KV reused | Latency |
|---|---|---|---|
| new name, new path | yes | 0 tokens | 0.17 s |
| same name, new path, `load_inplace: true` | yes | 0 tokens | 0.16 s |
| unload, then load same name from new path | yes | 0 tokens | 0.16 s |
| same name, files overwritten at same path, `load_inplace: true` | yes | 128 tokens (stale) | 0.17 s |
| unload, then load same name from overwritten path | yes | 128 tokens (stale) | 0.16 s |
| files overwritten, no reload call | no (old weights stay cached in vLLM) | n/a | n/a |

Qwen3-4B-Instruct-2507 gave the same pattern, with latencies around 0.30 s for its 264 MB adapters. Never reuse a path for changed weights: the request still gets KV computed with the old weights for every cached 64-token block of its prompt, silently. `load_inplace` with a new path is correct for new requests, but it swaps the weights of the GPU slot under the same internal adapter id, so sequences already running would continue under the new weights (inferred from vLLM's worker code; not tested). Unloading is API-level bookkeeping only (7 ms); the engine evicts old versions from its GPU slots and CPU cache by LRU, so many versions over a run need no cleanup beyond unloading names. Save and load latencies are under Throughput.

## Prompt construction and sampling

Build prompts with the tokenizer's chat template and send token ids to `/v1/completions`, so the trainer re-uses exactly the same ids:

```python
messages = [{"role": "system", "content": reasoning_gym.utils.SYSTEM_PROMPTS["default"]},
            {"role": "user", "content": question}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
prompt_ids = tok(text, add_special_tokens=False).input_ids
```

For Qwen3-1.7B the result ends with `<|im_start|>assistant\n<think>\n\n</think>\n\n`: the empty think block is part of the prompt, and the model then answers without thinking. For Qwen3-4B-Instruct-2507 the template has no think tags and ignores `enable_thinking`, ending with `<|im_start|>assistant\n`. Qwen has no BOS token; `add_special_tokens=False` and the default give the same ids. vLLM tokenizes the same string to identical ids (checked for all prompts), so sending the string also works, but ids remove any doubt.

The request body for rollouts is `{"model": "<adapter name or base id>", "prompt": prompt_ids, "n": 8, "temperature": 1.0, "top_p": 1.0, "max_tokens": ..., "logprobs": 0, "return_token_ids": true}`. Each choice then carries `token_ids` (the sampled completion, including the final `<|im_end|>` id 151645 when `finish_reason` is `stop`; the trainer must score that token too) and `logprobs.token_logprobs` (log-probability of each sampled token under the full softmax: vLLM's default `logprobs_mode=raw_logprobs` is taken before temperature and top-p/top-k, which equals the sampling distribution only at temperature 1 with no truncation). `skip_special_tokens` hides `<|im_end|>` from `text` but not from `token_ids`. Size the async client's concurrency deliberately: httpx's default pool of 100 connections silently caps it, an unlimited pool fails beyond about 1000 simultaneous requests at the default soft limit of 1024 open files ("All connection attempts failed"), and requests queued inside httpx's pool cost time quadratic in the queue length (a 2,472-request probe took 377 s instead of 49 s). The engine admits at most 512 requests at a time through an asyncio semaphore in front of a 512-connection pool, which still fills vLLM's 512 sequence slots, and resends a request up to three times after a transport-level failure (one connection reset in about 9,000 requests otherwise ended a job).

## Trainer–sampler consistency

Part (c) of `scripts/dev/lora_smoke.py` checks that the trainer scores the policy that generated the rollouts. Adapter B (r=32, `lora_alpha=64` so that a scaling bug would show, all seven linear projections, B drawn from N(0, 0.01)) was loaded into vLLM; 16 reasoning-gym prompts × 4 samples were drawn at temperature 1, top_p 1, max_tokens 512 (64 sequences, 7,701 completion tokens, mean 120); every sampled token's log-probability was recomputed with transformers + PEFT on exactly the prompt ids plus completion ids, one unpadded sequence per forward pass. Qwen3-1.7B:

| Comparison | token mean abs Δ | token mean Δ | token p99 abs Δ | sequence mean abs ΔΣ | sequence p95 abs ΔΣ |
|---|---|---|---|---|---|
| vLLM vs HF+PEFT bf16 | 0.0127 | +0.0009 | 0.21 | 0.40 | 1.68 |
| vLLM vs HF+PEFT fp32 | 0.0099 | +0.0009 | 0.15 | 0.28 | 1.17 |
| HF bf16 vs HF fp32 (noise floor) | 0.0109 | 0.0000 | 0.16 | 0.35 | 1.48 |
| control: vLLM with adapter vs HF with adapter disabled | 0.113 | +0.064 | 2.08 | 7.84 | 27.9 |

The check passes: vLLM is as close to the fp32 reference as HF bf16 is, the control shows the adapter is applied (10× the noise), and there is no systematic offset from chat template, BOS, think tags, LoRA scaling or dtype. A repeat with fresh samples gave the same picture (0.0133 / 0.0110 / 0.0123 per token). The small positive mean Δ is expected rather than a bug: tokens drawn from vLLM's distribution are on average ones vLLM scores slightly higher than an independent bf16 evaluation, so it estimates KL(vLLM ‖ HF) ≈ 1e-3 nats per token. Consequence for the trainer: in bf16 the per-sequence log importance ratio between trainer and sampler is typically 0.3–0.4 and reaches about 2.5 for 500-token sequences, so keep a ratio clip or truncated importance weights even for nominally on-policy updates, and use the trainer's own recomputed log-probabilities wherever an exact ratio of 1 is assumed.

The same test on Qwen3-4B-Instruct-2507 (64 sequences, 19,398 tokens, mean 303) also passes: vLLM vs HF bf16 0.0113 per token (mean Δ +0.0008), vLLM vs HF fp32 0.0086, HF bf16 vs fp32 0.0100, control 0.083; per-sequence mean abs ΔΣ 0.64 against bf16 for these longer sequences.

## Throughput

Rollouts (`scripts/dev/bench_rollout.py`): 128 reasoning-gym prompts of 150–300 tokens (mean 200), n=8 (1024 sequences submitted at once), temperature 1, max_tokens 512. "Natural" lets sequences stop at EOS, so wall time is set by the few that run to 512 tokens; "full" sets `ignore_eos` so all 1024 sequences decode 512 tokens. LoRA runs use random r=32 adapters; ×4 assigns four different adapters round-robin within the same batch. Values are wall seconds / output tokens per second (mean completion length for natural runs).

| Model, server | base natural | LoRA×1 natural | LoRA×4 natural | base full | LoRA×1 full | LoRA×4 full |
|---|---|---|---|---|---|---|
| 1.7B, KV 12G, defaults | 11.2 s / 3968 (44 tok) | 8.9 s / 3279 (29) | 14.1 s / 2799 (39) | 88 s / 5950 | 107 s / 4895 | 101 s / 5205 |
| 1.7B, KV 20G, 512 seqs | 12.0 s / 4105 (48) | 8.8 s / 3281 (28) | 10.9 s / 3513 (37) | 95 s / 5547 | 91 s / 5785 | 84 s / 6211 |
| 4B, KV 12G, 512 seqs | 121 s / 3062 (362) | 115 s / 3187 (358) | 126 s / 2970 (366) | 123 s / 4281 | 162 s / 3228 | 176 s / 2974 |

Qwen3-1.7B answers these prompts in a few dozen tokens, so natural rollouts finish in about 10 s; Qwen3-4B-Instruct-2507 writes about 360 tokens and 40% of its samples hit the 512 cap. Full-length decoding is KV-bound: with 1024 sequences of up to 712 tokens the cache fills, running sequences are preempted and recomputed, and engine throughput swings between about 3k and 13k tokens/s (1.7B, KV 20G); doubling the KV budget and the sequence limit did not raise the 1.7B average. Serving different adapters in one batch costs little for 1.7B and about 25–30% of full-length throughput for 4B.

Trainer (`scripts/dev/bench_trainer.py`), with the vLLM server resident on the same GPU: one update over 512 sequences of 600 tokens (250 prompt + 350 completion, policy-gradient loss on the completion tokens), forward and backward on the LoRA parameters only (r=32, all linear layers, PEFT's fp32 adapter weights over the bf16 base, SDPA attention, no padding), one AdamW step at the end. "Score" is a no-grad forward over the same 512 sequences (for example to recompute old-policy log-probabilities). Peak is `torch.cuda.max_memory_allocated`; the process adds about 1.5 GiB of CUDA context and allocator slack.

| Model | Micro-batch | Gradient checkpointing | Update | Score | Peak allocated |
|---|---|---|---|---|---|
| 1.7B | 4 | yes | 109 s | 22 s | 6.9 GiB |
| 1.7B | 8 | yes | 71–86 s | 21–26 s | 10.6 GiB |
| 1.7B | 16 | yes | 80 s | 24 s | 17.5 GiB |
| 1.7B | 32 | yes | OOM | | |
| 1.7B | 2 / 4 | no | 92 s / 76 s | 21 s | 9.8 / 16.5 GiB |
| 4B | 2 | yes | 195 s | 45 s | 9.8 GiB |
| 4B | 4 | yes | 164 s | 51 s | 12.1 GiB |
| 4B | 8 | yes | 158 s | 48 s | 15.7 GiB |

At micro-batch 8 the 1.7B update runs at roughly a third of the L20's dense bf16 peak (about 41 TFLOPS counting forward, recomputation and input-gradient backward), so it is not grossly inefficient, but it dominates a training iteration (below). The obvious levers, none measured here, are computing log-probabilities without materializing fp32 full-vocabulary logits (chunked or fused linear-cross-entropy), packing sequences, and running the LoRA matmuls in bf16.

Adapter hand-off after an update: `save_pretrained` writes 139 MB (1.7B) or 264 MB (4B) of fp32 LoRA weights in 1.4–2.0 s to `/dev/shm` and 3–11 s to `/Data2`; `load_lora_adapter` then takes 0.17–1.6 s and the first request through the new version 0.03–0.5 s. Save hand-off versions to `/dev/shm` (188 GB free) and copy checkpoints to disk asynchronously.

## Recommended per-GPU layout

One vLLM server and one trainer process per GPU, started in that order, with the KV budget fixed by `--kv-cache-memory-bytes`. Measured `nvidia-smi` totals with both processes resident (card: 45.0 GiB):

| Model | vLLM | Trainer | Card total | Headroom |
|---|---|---|---|---|
| Qwen3-1.7B | KV 20G, 512 seqs, 8192 batched tokens: 27.4 GiB (0.61) | micro-batch 8, gradient checkpointing: 12.0 GiB | 39.5 GiB | 5.5 GiB |
| Qwen3-4B-Instruct-2507 | KV 12G, 512 seqs, 8192 batched tokens: 25.5 GiB (0.57) | micro-batch 4, gradient checkpointing: 13.1 GiB | 38.6 GiB | 6.4 GiB |

For 1.7B, KV 12G (18.9 GiB) loses nothing at today's short completions and frees 8.5 GiB, but the larger cache matters once training lengthens the answers. For 4B, micro-batch 8 also fits (42.7 GiB total) but leaves only 2.3 GiB for a 4% faster update, which is too little on a shared machine.

Per adapter and iteration (128 prompts × 8 samples, then one update over 512 sequences of 600 tokens), the cost is roughly: Qwen3-1.7B, 10–15 s of rollout at current answer lengths (up to about 90 s if every sample ran to 512 tokens), 70–86 s of update, 2–5 s of hand-off; Qwen3-4B-Instruct-2507, about 120 s of rollout, 160 s of update. The 4B figure understates its rollout cost: in the calibration half of its samples did not finish within 1024 tokens, so it needs a larger budget or a brevity instruction. The trainer is the bottleneck. Batching several adapters into one rollout costs little, so the engine should batch rollouts for all adapters on a GPU and spend its effort on the update path; overlapping one adapter's update with the next adapter's rollout is possible memory-wise but untested (the two compete for the same SMs).

## Pitfalls

- FlashInfer JIT-compiles its sampling kernels on the first vLLM start (about 3 minutes) with whatever `nvcc` it finds. The default `PATH` has CUDA 12.8's `nvcc`, whose build linked `libcudart.so.12` into a process that runs torch's CUDA 13 runtime; `scripts/env.sh` therefore points `CUDA_HOME` and `PATH` at `/usr/local/cuda-13.2`. The JIT cache is `~/.cache/flashinfer/0.7.3/` and vLLM's torch.compile cache is `~/.cache/vllm/` (shared with this user's other projects); with warm caches a server starts in about 3.5–4 minutes, mostly torch.compile and CUDA-graph capture.
- `--gpu-memory-utilization` sizes the KV cache from memory measured during startup. When another job allocated 32 GB on the same GPU during that window, vLLM attributed it to itself and died with "No available memory for the cache blocks". `--kv-cache-memory-bytes` avoids this, but the startup free-memory check against `--gpu-memory-utilization` remains (see vLLM server above).
- Other users' and the owner's jobs land on a GPU after vLLM has started (twice in this session: 32 GB, then 17–20 GB at 100% utilization on the card vLLM was using). Co-tenancy halved rollout throughput, made LoRA runs look faster than base, and caused OOMs in the trainer. Check `nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv` before trusting any timing; all timings in this note were measured with no other process on the card (logged every 5 s).
- Without `--generation-config vllm`, Qwen3's `generation_config.json` silently adds `top_k=20, top_p=0.95` (1.7B) or `top_k=20, top_p=0.8` (4B) to requests that only set temperature, and the trainer would score a different distribution than the one sampled. The flag keeps the checkpoint's EOS ids (`<|im_end|>` 151645 and `<|endoftext|>` 151643 both stop generation).
- Greedy text equality is not a test of which weights are served. In bf16, near-ties flip between a cold prefill and a prefix-cached one: on one prompt the fp32 reference gave log-probabilities −0.32/−1.31 for two candidate digits, HF bf16 −2.35/−0.10, and vLLM picked different tokens with and without the prefix cache. Compare teacher-forced log-probabilities (`"prompt_logprobs": 0`, which are never served from the prefix cache) instead.
- The prefix cache is keyed by adapter name and path, not content: overwriting an adapter directory in place and reloading reuses KV computed with the old weights. Overwriting without reloading changes nothing, because vLLM holds the loaded weights in memory.
- httpx's default connection pool (100) caps client-side concurrency without any error, and an unlimited pool breaks at the 1024 open-file limit and with long client-side queues; see Prompt construction and sampling for the engine's bounded client.
- vLLM's L20 defaults are `max_num_seqs=256` and `max_num_batched_tokens=2048`; long rollouts are KV-bound before they are compute-bound (see Throughput).
- PEFT keeps LoRA weights in fp32 (`autocast_adapter_dtype`), so a saved r=32 adapter for all linear layers is 139 MB for Qwen3-1.7B; `save_pretrained` takes 1.4–4 s and dominates the hand-off. vLLM casts the weights to bf16 on load.
- reasoning-gym: `reasoning_gym.__version__` reports 0.1.19 for the 0.1.25 wheel. `game_of_life_halting` scores any non-empty answer 1.0 (its scorer compares `bool(answer)` with `bool("False")`). `prime_factorization` raises on non-numeric answers and `game_of_life` raised `KeyError` on one model answer, so reward calls need a guard. `countdown`, `puzzle24` and `fraction_simplification` give 0.01 even for a missing answer, and datasets on the base-class scorer give `len(gold)/len(answer)` credit whenever the gold string is contained in the answer. `arc_agi` prompts reach 12.8k tokens and exceed `--max-model-len 8192`; `composite` needs an explicit dataset list. Details are in `docs/dev/REASONING_GYM_CALIBRATION.md`.
- `pgrep -f`/`pkill -f` patterns that also appear in your own shell command match the shell itself; stop servers by the PID saved at launch.
