# Scale notes

Internal current-state notes for the code of the larger-model stage. The design is `docs/design/scale.md`, the build specification `docs/dev/SCALE_SPEC.md`; the feasibility scout's measurements (forms of the 27B, memory, throughput) are in `docs/dev/SCALE_SCOUT.md`. So far this covers WP-M, the method census.

## Method census: code and commands

| File | Role |
|---|---|
| `evotaste/method.py` | the stages, their manifests, prompts, pairs, judge parsing, the reader sample |
| `configs/method.yaml` | subject models and their servers, item draw, sample counts, seeds, and the instrument's texts: the alternatives prompt and the judge's prompt and definitions |
| `evotaste/analysis/method.py` | the report: accuracy, method variation, method on request, cross-model pairs, judge quality, typicality |
| `tests/test_method.py` | item draw, pairs, judge parsing, construct profiles, typicality statistics (about 5 s once the parts cache exists) |

All commands run from the repository root after `source scripts/env.sh`. A stage that generates needs one card (`CUDA_VISIBLE_DEVICES=<g>`, or two with tensor parallelism in the model's server arguments) with the model's `need_free_mib` free; it starts and stops its own vLLM server. `--limit N` runs on the first N items of the draw, under `results/method/<name>-limitN/`; give it to every stage.

| Stage | Command |
|---|---|
| Item draw | `python -m evotaste.method items configs/method.yaml` |
| Attempts | `CUDA_VISIBLE_DEVICES=<g> python -m evotaste.method attempts configs/method.yaml --model <label> [--shard i/n]` |
| Alternatives | `CUDA_VISIBLE_DEVICES=<g> python -m evotaste.method alternatives configs/method.yaml --model <label> [--shard i/n]` |
| Pairs | `python -m evotaste.method pairs configs/method.yaml` |
| Judge | `CUDA_VISIBLE_DEVICES=<g> python -m evotaste.method judge configs/method.yaml [--shard i/n]` |
| Analysis | `python -m evotaste.analysis.method results/method/<name> [--out DIR] [--labels FILE ...] [--boot 1000]` |

Attempts of both models are independent and can run side by side; alternatives need the same model's attempts, pairs need every model's attempts and alternatives, the judge needs the pairs. Shards of one stage split its chunks by index modulo n and may run on several cards at once; the last judge process to find every chunk present writes the reader sample. `--kv-cache` and `--gpu-memory-utilization` override the config's server settings for one process (memory and speed only), and `--attach URL` uses a running server of the stage's model (development).

## Stages and files

Everything is under `results/method/<name>/` (outside git):

| Path | Content |
|---|---|
| `items.tsv`, `items.json` | the draw: uid, domain, source, the 1.7B census's passes and samples; its manifest |
| `attempts/<label>/`, `alternatives/<label>/` | `manifest.json`, `chunk-NNNNN.jsonl.gz`, `timing.jsonl` |
| `pairs.jsonl.gz`, `pairs.json` | the pairs to judge; manifest |
| `judge/` | `manifest.json`, chunks, timing |
| `reader-sample.jsonl`, `reader-key.jsonl`, `reader-instructions.txt` | the pairs for independent readers, the key, and what the readers are told |
| `logs/` | the vLLM server logs |
| `analysis/` | `report.txt`, `figures.json`, `pairs.csv` |

A manifest holds the settings a stage's records depend on (for a generation stage the model's label, weights and server arguments, and the stage's section of the config, chunk size included, and for the two stages that score answers the verifier's `RULES_VERSION`; for the judge its whole section, its own server arguments included) and the SHA-256 of the manifests of the stages it reads, so a change anywhere upstream changes every manifest below it. A stage refuses to continue, and a downstream stage to read, records whose manifest is not the one the current config gives; the analysis checks that the recorded upstream hashes are those on disk. Memory and speed settings (`kv_cache`, `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`) are left out. The config's `name` is in no manifest, so a census directory can be copied under another name and a stage rerun there with other settings (see the smoke run below for a second judge on the same pairs). Verifier processes run whatever verifier code is on disk, so every stage also saves `git diff HEAD` at launch when the working tree is dirty (`logs/method-<name>/uncommitted-<sha>.patch`); records scored while the verifier is being edited can be traced, though only a bumped `RULES_VERSION` makes the manifests refuse them.

Records of attempts and alternatives have the census record shape (`census.sample_records`: part, uid, family, domain, kind, wrapper, context, variant, sample, text, completion_tokens, finish_reason and every `Score` field) plus `model`, `logprob_sum` and `logprob_mean` (the sampled tokens' log-probabilities under the full softmax, end-of-turn token included), and for alternatives `reference`, the attempt shown. A pair is `{id, kind, uid, domain, a, b, of}` with sides `{model, stage, sample}`; a judge record adds `judge`, `prompt_tokens`, `truncated_a/b`, `replies` (text, tokens, finish reason of each reply), `completion_tokens`, `parsed`, `thought`, `approach_a`, `approach_b`, `relation` (all three None when unparsed).

## Design

**Items.** Only the train part is read: tasks come from `Suites.items(source, "train")`, a census row naming any other uid stops the draw, and every later stage checks each uid of `items.tsv` against the train part again. Within each cell of domain and census stratum the target is split over the sources in proportion to their items (largest remainder), each source's share drawn by a seeded sample; the drawn items are then put in a seeded order, so that `--limit N` is a small random sample of the draw (the first 60 hold 35 math, 16 code and 9 logic items, every cell represented). Proportional allocation makes the math frontier 62% DeepScaleR (186 of 300) and the math sweet spot 38%, because those are where the 1.7B fails; code is KodCode alone.

**Subject prompts.** Attempts use the training prompt, `tasks.prompt_ids` (the one-sentence system prompt, thinking disabled). Before generating, a stage refuses a template that does not end the prompt with the empty think block under `enable_thinking=False` (both Qwen3 and the Qwen3.5-family template of the 27B do), and refuses to write a chunk in which any sample contains `<think>` or `</think>` (none of the 1.7B census's 220,896 samples did). Sampling is as in training: temperature 1, top-p 1, no top-k (`--generation-config vllm` keeps the 27B's recommended top-k 20 and top-p 0.95 out).

**Alternatives.** One user turn, under the same system prompt: the problem, "Here is a correct solution to this problem:" and the model's first correct attempt (lowest sample index with reward 1) between `<solution>` tags, the request for a different method, and the item's own answer instruction last, as in every training prompt. A multi-turn form (the attempt as the model's own previous turn) was not used because the two templates render earlier assistant turns differently (Qwen3 drops the think block, the Qwen3.5 template keeps an empty one), and the shown solution is not said to be the model's own. The request defines method as the design does and names no particular technique; its text is `alternatives.prompt` in the config. Alternatives are verified by the item's normal verifier.

**Pairs.** Spontaneous pairs are disjoint (an item with two or three correct attempts gives one pair, with four or more two), so that an item's two pairs are independent draws of its method variation. The requested pair is the reference and the first correct alternative; the cross pair takes the first correct attempt of each model, which, the samples being exchangeable, is a uniformly random correct one. The side shown first is a seeded coin flip per pair; a repeat shows its pair's sides swapped.

**Judge.** Greedy decoding through vLLM's structured outputs (xgrammar), constrained to a compact JSON object with exactly `approach_a`, `approach_b` (each at most 160 characters; the prompt asks for at most 15 words) and `relation` in that order, so the approaches are named before the verdict and a well-formed reply is guaranteed unless it is cut off. vLLM 0.34 supports this on the completions endpoint; its request model accepts the request body and xgrammar compiles the schema against the 27B's tokenizer, accepting a well-formed verdict and rejecting misordered keys (checked offline, no GPU). The reply is still parsed strictly (exactly these keys in this order, non-empty approaches, a known relation; nothing stripped or repaired, so a fenced or prefixed object is unparsed). A reply that does not parse is asked for once more with twice the token budget, because the only way a constrained greedy reply fails is being cut off, and repeating the same greedy request would repeat it; if that fails too the pair is recorded as unparsed. The judge sees the problem as the solvers saw it (with its answer instruction) and the two solutions verbatim, labelled only A and B; a solution longer than `solution_tokens` (1,500) keeps its first two thirds and its last third of that budget, with a marker giving the number of tokens left out.

Thinking is off by default (`judge.thinking: false`): the verdict object already makes the judge state both approaches first, and with thinking at the template's default effort (`xhigh`) the judge would generate thousands of tokens per pair over some 8,000 pairs. `thinking: low | medium | xhigh` renders the prompt with `enable_thinking=True` and that `reasoning_effort` and needs `--reasoning-parser qwen3` in `judge.args` (the config refuses otherwise), so that the JSON constraint starts after `</think>`; `thought` records whether the reasoning was non-empty. `judge.args` are server flags added for the judge's server only, so that a judge setting never changes the subject model's manifests.

**Judge budget.** With the 1.7B's correct answers as solutions, judge prompts are 960 tokens on average for code, 1,720 for logic and 1,900 for math (at most 3,750), and 9.5% of the 1.7B's correct math answers, 1.9% of logic and none of code are longer than the budget. Prompts therefore stay below 4,000 tokens; with 256 reply tokens (512 on the retry) a context of 8,192 would do, and the configured 16,384 (needed for alternatives prompts, which hold a whole attempt: up to 4,400 tokens on the drawn items, plus 4,096 new tokens) leaves room for longer 27B answers or a thinking judge.

**Mechanical comparison on code.** The construct profile of a program (the verifier's extracted program) is a set: recursion, for and while loops, comprehensions, sorting, hashing containers, heap, deque, bisect, itertools, regex, math, string methods, index scanning (`for` over `range(len(...))` or a subscripting `while`), early return, and loop nesting depth (1, 2, 3 or more). A pair's distance is the Jaccard distance of the two sets; the analysis reports its Spearman correlation with the verdict (same < variant < different) and its mean per verdict. It is a cross-check with its own notion of method: the judge prompt counts a library routine in place of hand-written code as a variant, which the profile may count as a change of constructs.

**Analysis.** Every interval resamples items within the draw's cells, and shares of pairs are ratios of item sums (pair-weighted, the pairs of an item resampled together). Accuracies over a domain or overall are for the designed sample, which over-represents the 1.7B's frontier and sweet spot; read them by cell. Typicality is the per-token mean log-probability of an attempt: on items with correct and wrong attempts, the within-item standardised difference (mean of correct − mean of wrong, over the SD of the item's samples), accuracy by within-item tercile (sizes 3, 3, 2 of 8), the share of all correct answers in the least typical tercile against its chance value 3/8, and all of these again after removing a within-item slope on log length fitted per domain over all items (the slope is held fixed in the bootstrap). Cut-off attempts stay in as wrong answers.

## Serving the 27B

The FP8 weights take 27.9 GiB on an L20 (vLLM's "model loading" line, with `--language-model-only` so that the vision tower is not loaded), which leaves about 10 GiB of cache beside another user's resident 2.7 GB process; KV 10G fills the card (45.1 of 46 GB in use). The cache is shared by attention tokens and the DeltaNet states: vLLM makes one page size for both, here 832 attention tokens or the recurrent state of 16 DeltaNet layers (about 52 MB, the state being float32 as the model's config asks), so a running sequence holds 3 state blocks (48 DeltaNet layers) plus one attention block per 832 tokens, and 10G gives about 196 blocks. With prefix caching on, vLLM puts the DeltaNet layers in its "align" mode, which also keeps a state checkpoint every 832 tokens; our prompts share little (eight samples share one short prompt), so the 27B runs with `--no-enable-prefix-caching`. No LoRA support is enabled (the method census serves no adapters), which saves the adapter buffers. `max_num_seqs` must not exceed the number of state blocks (vLLM refuses at start); 64 slots are more than the cache can run.

Measured on GPU 7 (28 November, `results/method/_dev/serve27b.py`, the smoke's 60 items × 8 samples at 4,096 tokens, outside the census): the server is ready in about 2.5 minutes, runs 37 to 40 sequences at once with the cache at 98–100%, and generates 400–490 tokens/s, 434 tokens/s over the whole job (570,627 tokens in 1,316 s, the last minutes being the long tail). No sample contained a think tag. The 27B's answers are longer than the 1.7B's: 1,590 tokens on average for math (7% cut off at 4,096), 835 for logic, 510 for code (it explains around its program; the 1.7B writes the program alone, about 140).

Concurrency is set by the recurrent state, so more throughput would need a smaller state or more memory: a bfloat16 state (`--mamba-ssm-cache-dtype bfloat16`) would roughly halve the state blocks and an FP8 attention cache would halve the attention blocks, both changing the numerics against which the log-probabilities are measured, and neither is used. The model ships one multi-token-prediction layer that vLLM can use for speculative decoding; whether its log-probabilities and sampling stay exact at temperature 1 was not checked.

## Changes to shared code

- `census.run_chunks` is the census's pipelined chunk loop (two chunks in flight, atomic chunk files, the refusal of chunks whose verifier processes died, timing lines), now a function the census and the method census both call; `census.sample_records` is the record builder. The census's behaviour and its tests are unchanged.
- `sampler.Server` takes `max_lora_rank=None` (no adapter support), `max_model_len` (default 8,192 as before) and `extra_args` (further `vllm serve` flags); `sampler.Request` takes `temperature` (default 1) and `json_schema` (structured outputs). Engine calls are unchanged.
- `tasks.strata_rows` (formerly `_strata_rows`) is public.

## Spec points settled here

- "Up to 2 pairs of distinct correct attempts": disjoint pairs, so three correct attempts give one pair (above).
- "Retried once": with twice the token budget, since a greedy retry of the same request returns the same reply.
- "Refuses to mix records under different configs": per-stage manifests chained by hash, as described; the census itself has no such hash, only its strata-table check.
- `items.tsv` also has the census's `samples` column, so the file states its strata.
- The judge stage also writes `reader-instructions.txt`, generated from the judge's own method definitions and relations, so readers and judge read the same text.

## Dry run (28 November, no GPU)

- `tests/test_method.py`: 21 passed; `tests/test_census.py` and `tests/test_sampler.py` pass after the refactor.
- The full draw on the real census (`results/method/_dev/draw/`): exactly the targets per cell, all from the train part.
- Prompts for every drawn item on both templates (`results/method/_dev/prompts.py`, examples in `prompts.txt`): attempt prompts 120 (math) to 230 (code) tokens at the median; alternatives prompts up to 4,418 tokens; the lengths of judge prompts above.
- End to end on synthetic records (`results/method/_dev/synth/`: 82 items, the 1.7B's census answers as attempts and alternatives of both labels, fake log-probabilities and verdicts): the attempts and alternatives stages build and check every prompt and stop with nothing to generate, pairs are drawn (366: 180 spontaneous, 94 requested, 62 cross, 30 repeats), the judge stage builds every judge prompt and writes the reader sample, and the analysis runs with two fake label files. Changing the alternatives prompt made the alternatives stage, pairs and judge refuse; removing a judge chunk made the analysis refuse.

## Expected volume of the full run

From the 1.7B census's answers on the drawn items (1,800 items; the 27B's lengths and pass rates are not known yet):

| Stage | Sequences | Prompt tokens | Completion tokens |
|---|---|---|---|
| 1.7B attempts | 14,400 | 0.3 M (8 samples share a prompt) | 9.8 M (math 7.6, logic 1.5, code 0.8) |
| 1.7B alternatives | about 2,400 (1,200 items) | 1.0 M | 1.4 M |
| 27B attempts | 14,400 | 0.3 M | 10 to 15 M if its answers are as long as the 1.7B's or half again |
| 27B alternatives | about 3,400 | 1.5 M | 2.5 to 4 M |
| Judge (27B) | about 8,000 to 9,000 pairs | about 14 M (1,600 per pair) | about 0.5 M without thinking |

Pairs: the 1.7B gives about 1,750 spontaneous and 700 requested pairs, the 27B up to about 3,200 and 1,400, cross pairs about 1,150, plus 200 repeats. The 1.7B stages take one to two GPU-hours at the census's rates; the 27B stages depend on its throughput (12 to 19 M generated tokens and about 14 M of judge prompt).

## Smoke run (to do when a card is assigned)

```
source scripts/env.sh
python -m evotaste.method items configs/method.yaml --limit 60
for m in 1.7b 27b; do
  CUDA_VISIBLE_DEVICES=<g> python -m evotaste.method attempts configs/method.yaml --model $m --limit 60
  CUDA_VISIBLE_DEVICES=<g> python -m evotaste.method alternatives configs/method.yaml --model $m --limit 60
done
python -m evotaste.method pairs configs/method.yaml --limit 60
CUDA_VISIBLE_DEVICES=<g> python -m evotaste.method judge configs/method.yaml --limit 60
python -m evotaste.analysis.method results/method/method-1-limit60 --boot 200
```

A second judge on the same pairs (thinking at low effort, say): copy the directory without `judge/` and the reader files to `method-1t-limit60`, set `name: method-1t` and `thinking: low` with `--reasoning-parser qwen3` in a copy of the config, run the judge stage on it, and compare the verdicts pair by pair.

## Known limitations

- The 27B's form and server settings in the config (FP8 weights, KV budget, sequence count, free-memory threshold) are provisional until the scout's measurements; they are recorded in the manifests only through the weights path and the server arguments.
- Some KodCode problems prescribe the method ("reverse a list using recursion"), which bounds the variation that code can show; MMLU-Pro and ARC questions are often recall, where a line of argument is thin.
- Cut-off attempts count as wrong in the typicality statistics; a repetition loop is highly probable per token, which works against the hypothesis that correct answers are the typical ones; the length adjustment removes only a linear trend in log length.
- The construct profile is coarse and lexical (an import counts as use; a nested function's returns count for its parent).
