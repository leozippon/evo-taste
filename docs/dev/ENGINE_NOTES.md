# Engine notes

Internal operating notes for the first EvoTaste engine (`evotaste/`). The build specification is `docs/dev/ENGINE_SPEC.md`; the vLLM and trainer recipes it relies on are in `docs/dev/INFRA_NOTES.md`. This file holds the commands, the run-config schema, the on-disk layout and record schemas that analysis code is written against, measured costs, and known limitations.

## Commands

All commands run from the repository root after `source scripts/env.sh`, with exactly one GPU visible. Pick a GPU whose `memory.used` is below 500 MiB immediately before launching and pin it with `CUDA_VISIBLE_DEVICES`. Each command that needs generation starts its own vLLM server (1–4 minutes with warm compile caches) and stops it on exit, including on SIGTERM; the server also receives SIGTERM if its parent is killed with SIGKILL. `--attach http://127.0.0.1:<port>` makes a command use an already running server of the same model instead (development only; the server must have been started with runtime LoRA updating, see `scripts/dev/serve_vllm.sh`).

| Step | Command |
|---|---|
| Assign suites (once per project) | `python scripts/make_suites.py` (writes `configs/suites.yaml`) |
| Prepare a run | `CUDA_VISIBLE_DEVICES=<g> python -m evotaste.worker prepare configs/run/<name>.yaml` |
| Start a worker | `CUDA_VISIBLE_DEVICES=<g> nohup python -m evotaste.worker run runs/<name> --members m00,m01,m02 > /dev/null 2>&1 &` |
| Resume | the same `run` command again |
| Evaluate | `CUDA_VISIBLE_DEVICES=<g> python -m evotaste.evaluate run runs/<name> --suite selection --adapters base,m00,m01 --per-family 50 [--samples 1] [--shard i/n] [--tag t]` |
| Harvest | `python -m evotaste.evaluate harvest runs/<name> --tag n50x1`, then commit `docs/harvest/<name>.json` |

`prepare` freezes the run: it copies the config into `runs/<name>/config.yaml` (with `suites` and `scaffolds` pointing at copies of `configs/suites.yaml` and `configs/scaffolds.yaml` placed next to it), checks that every probe and seeding-pool gold answer earns reward 1, lets the base model answer the seeding pool once per teacher condition (plain, and each scaffold named by a member), and answers the probe set with the base model. It is idempotent and refuses to run if the run directory exists with a different config, suites or scaffolds. Workers only read these shared artefacts and refuse to start on an unprepared run, so several workers on different GPUs can share one run directory: give each a disjoint `--members` list. An exclusive lock per member slot (`runs/<name>/locks/<slot>.lock`) makes a second worker on the same member exit immediately.

A worker advances its members until each has finished its steps or its slot is exhausted. Killing it at any point and starting the same command again resumes every member from its last checkpoint. Logs go to `logs/<name>/`: one file per command invocation (`prepare-…`, `worker-<members>-…`, `eval-<suite>-…`) and one vLLM log per server.

`evaluate run` accepts `base`, a lineage id (its last checkpoint, which is the final adapter of a finished lineage) or `<lineage>@<step>`. Sharding splits the (adapter, instance) job list; each shard writes its own record file and every invocation rewrites the summary over all record files of the tag, so the summary is complete once the last shard has finished. On `blind1` or `blind2` the evaluator first calls the blind guard (below). Never evaluate a trained adapter on a blind vault outside the protocol's final comparisons.

## Run configuration

A run config is YAML; unknown keys and wrongly typed values are errors. Defaults are those of `docs/design/protocol.md` and the spec. Section by section:

| Key | Default | Meaning |
|---|---|---|
| `name` | required | run directory `runs/<name>/` and log directory `logs/<name>/` |
| `model` | required | Hugging Face id, e.g. `Qwen/Qwen3-1.7B` |
| `seed` | required | seed of the training stream and the seeding pool (probe and evaluation sets do not depend on it) |
| `suites`, `scaffolds` | `configs/suites.yaml`, `configs/scaffolds.yaml` | sources copied at prepare time |
| `server.kv_cache` | `20G` | vLLM `--kv-cache-memory-bytes` |
| `server.gpu_memory_utilization` | 0.65 | with a fixed KV budget only a startup check (free memory ≥ fraction × card); vLLM's default 0.92 fails whenever the trainer, loading in parallel, or anyone else holds memory |
| `server.max_num_seqs`, `server.max_num_batched_tokens` | 512, 8192 | vLLM scheduler limits |
| `server.max_loras` | 8 | adapters that can share a batch; raise it if a worker has more members |
| `lora.rank`, `lora.alpha` | 32, 64 | LoRA on all seven projection types |
| `trainer.token_budget` | 8192 | padded tokens per micro-batch (rows × longest row) |
| `trainer.shared_prefix` | true | one row per prompt holding the prompt once and all its completions |
| `trainer.grad_checkpointing` | true | needed to fit next to vLLM |
| `trainer.max_grad_norm`, `trainer.weight_decay` | 1.0, 0.0 | AdamW (betas 0.9/0.999, eps 1e-8) |
| `trainer.check_rollouts` | 8 | rollouts re-scored by the trainer before every update |
| `seeding.pool_prompts`, `seeding.samples`, `seeding.max_tokens` | 1536, 4, 640 | teacher sampling of the seeding pool (shared per teacher, not overridable per member) |
| `seeding.steps`, `seeding.batch`, `seeding.lr` | 30, 64, 1e-5 | positive warm-up updates |
| `seeding.perturb_std` | 0.01 | perturbation arm: LoRA B drawn from N(0, σ) |
| `amplification.steps`, `.prompts`, `.rollouts`, `.max_tokens`, `.lr` | 300, 64, 4, 640, 1e-5 | group updates; sampling is always at temperature 1, top-p 1 |
| `probes.every` | 20 | probe and checkpoint every E steps, and after the last step |
| `probes.train_per_family`, `.selection_per_family`, `.foundation_per_source` | 8, 12, 150 | probe set sizes |
| `probes.max_tokens` | 640 | also used by `evaluate` |
| `culling.grace` | 40 | no culling at probes before this step |
| `culling.max_replacements` | 2 | new lineages per slot before the slot is given up |
| `members` | required | list of `{id, arm, seed, scaffold?, overrides?}` |

`arm` is `plain`, `scaffold` (requires `scaffold`, a name from `control` or `scaffolds` in the scaffolds file, e.g. `cot` or `verify`) or `perturb`. Member ids use letters, digits, `_` and `-`. `overrides` may replace keys of `seeding` (except the pool keys), `amplification` and `culling` for one member; the long baseline is `overrides: {amplification: {steps: <N × steps>}}` and the wide one `overrides: {amplification: {prompts: <N × 64>}}`. Batch b of a member with `prompts: P` is items `[bP, (b+1)P)` of the run's training stream. `configs/run/toy.yaml` is the example.

## Mechanisms worth knowing

Instances are pure functions of (stream key, family, occurrence): stream item j of the training stream (`train/<seed>`) or seeding pool (`seedpool/<seed>`) takes its family from a seeded permutation of the train families per block of `len(train)` items, so every family appears once per block. A training or seeding item whose question equals a train-test probe question is redrawn deterministically (up to 16 draws). Probe sets use stream key `probe`, evaluation sets `eval`; neither depends on the run seed. External items are split by a seeded hash of their id: GSM8K train is foundation; MMLU-Pro is split three ways into foundation, blind1 and blind2; GSM8K test and MATH-500 two ways into blind1 and blind2. Probes take the first items of the foundation order, evaluation takes the last ones.

The reward is 1 if the completion stopped by itself, contains an `<answer>…</answer>` span (the last one counts), and the family's scorer gives at least 0.999; otherwise 0. Scorer exceptions and scorer runs longer than 10 s score 0 and are recorded in `score_error`. External scorers: GSM8K requires exactly one number equal to the gold number; MATH-500 uses math-verify equivalence; MMLU-Pro requires the answer to start with the gold option letter.

The update minimises −Σᵢ wᵢ Σₜ log π(yᵢₜ) / Σᵢ |yᵢ| over sequences with wᵢ ≠ 0, end-of-turn token included, with one AdamW step and gradient clipping. Group weights are (r − mean) / (population std + 1e-6) within each prompt's rollouts, zero for groups with identical rewards; a step without any non-zero weight leaves the weights and the optimizer untouched. Amplification starts from a fresh AdamW state. The trainer scores exactly the sampled token ids behind exactly the prompt ids sent to vLLM. Each adapter version is saved in bf16 to a new directory under `/dev/shm/evotaste/<run>/<pid>/` and loaded under a new name `<lineage>@<pid>-<counter>`; the previous version is then unloaded and its directory removed.

Before every update the worker re-scores the first rollout of the first `check_rollouts` prompts with the trainer and compares with vLLM's log-probabilities. It stops with `ConsistencyError` if the per-token mean |Δ| exceeds 0.04 or Σ|ΔΣ| / Σ tokens exceeds 0.015. Measured with the engine's own code (Qwen3-1.7B, random adapter, 128 rollouts of up to 640 tokens): 0.0057 and 0.0022 with the right adapter; 0.12 and 0.073 when the base model is served instead of the adapter or the reverse.

A member is culled at a probe at step ≥ `grace` (never at step 0) if fewer than half of its probe answers are parseable, more than half are truncated, its foundation accuracy is below half the base model's, or its train-test accuracy is below the base model's at this probe and at the previous probe at step ≥ `grace`. The culled lineage keeps its checkpoint; slot `s` continues with lineage `s~k` (k = 1, 2, …), seed `stable_seed("lineage", <slot seed>, k)`, which starts from seeding and runs the full course.

## On-disk layout

```
runs/<name>/
  config.yaml, suites.yaml, scaffolds.yaml   frozen at prepare time
  prepared.json                              {time, revision, teachers}
  seeding/<teacher>.jsonl.gz, <teacher>.json teacher = plain or a scaffold name
  base/probe_responses.jsonl.gz, probe_summary.json
  members/<lineage>/
    state.json, history.jsonl, probes.jsonl
    rollouts/step_TTTTTT.jsonl.gz            amplification step T (0-based)
    probe_responses/step_SSSSSS.jsonl.gz     probe after S updates (S = 0: after seeding)
    checkpoints/step_SSSSSS/                 adapter_config.json, adapter_model.safetensors (fp32), optimizer.pt
  eval/<suite>/<tag>/                        params.json, records-<key>-<i>of<n>.jsonl.gz, summary.json
  locks/<slot>.lock
docs/harvest/<name>.json                     harvest manifest (tracked)
docs/blind-ledger.md                         blind ledger (tracked)
```

Every checkpoint keeps its adapter, a PEFT directory vLLM can load (139 MB for 1.7B at rank 32); only the newest keeps `optimizer.pt` (278 MB; absent at step 0, where amplification starts with a fresh optimizer). `state.json` is rewritten only after a checkpoint is complete and is the resume point: on restart, history lines of amplification steps ≥ S, probe lines and files after S, rollout files of steps ≥ S and unfinished checkpoint directories are deleted, so redone steps replace their old records. A lineage without a checkpoint (status `new`) restarts from seeding with its directory cleared.

## Record schemas

Task uids are `<family>/<dataset seed>/<index>` for reasoning-gym instances (regenerate with `reasoning_gym.create_dataset(family, size=8192, seed=seed)[index]`) and `<source>/<item id>` for external items (GSM8K: row index in the split; MATH-500: `unique_id`; MMLU-Pro: `question_id`). All times are UTC ISO strings.

A *response record* has the fields `uid, family, question, answer` (gold, as text), `text` (completion text without special tokens), `completion_tokens` (sampled ids including the end-of-turn token), `finish_reason` (`stop` or `length`), `extracted` (content of the last answer span or null), `score` (raw scorer output; 0 when not scored), `reward` (0 or 1), `parsed`, `truncated`, `score_error` (null or message) and `logprob` (vLLM's summed log-probability of the sampled ids). Files built from it:

| File | One record per | Fields besides the response record |
|---|---|---|
| `rollouts/step_T.jsonl.gz` | rollout | `step` (T), `prompt` (index within the step's batch), `k` (rollout index), `weight` (group weight used in the update) |
| `probe_responses/step_S.jsonl.gz`, `base/probe_responses.jsonl.gz` | probe answer | `step` (S; null for the base model), `suite` (`train`, `selection` or `foundation`) |
| `eval/<suite>/<tag>/records-*.jsonl.gz` | sample | `adapter` (label as given), `adapter_sha256` (null for base), `suite`, `sample` (index) |

`history.jsonl`, one line per update:

- seeding (`phase: "seed"`): `step` (0-based warm-up update), `time`, `loss`, `grad_norm` (before clipping), `sequences`, `tokens` (completion tokens in the loss), `micro_batches`, `update_s`.
- amplification (`phase: "amp"`): the same plus `version` (adapter name published after the update), `reward` (mean over rollouts), `reward_by_family`, `completion_tokens` (mean), `parsed`, `truncated` (fractions), `groups` (prompts with non-identical rewards), `prompts`, `check` (`rollouts, tokens, token_mean_abs, seq_mean_abs, seq_abs_per_token`) and `timing` (`generate_s`: wall time of the rollout batch shared by `batch_members` members; `score_s, check_s, update_s, publish_s, write_s`: this member's own).

`probes.jsonl` and `base/probe_summary.json`: `step` (members only), `time`, `n`, `accuracy` (`train`, `selection`, `foundation`; each the mean over families of the family's accuracy), `by_family`, `parsed`, `truncated`, `completion_tokens`, and for members `cull` (null or the reason).

`state.json`: `id, slot, generation, arm, scaffold, seed`, `status` (`new`, `active`, `done`, `culled`), `step` (updates done at the last checkpoint), `probes` (the probe summaries up to it), `cull_reason`, `replaced_by`.

Seeding pool records (`seeding/<teacher>.jsonl.gz`, correct and stopped samples only): `uid, family, prompt_ids` (the plain prompt the member trains on), `completion_ids, text, teacher_logprob`. The summary `<teacher>.json` holds `teacher, time, generate_s, prompts, samples, correct, prompts_with_correct, by_family`.

Evaluation `summary.json`: `time, suite, per_family, samples` and `adapters` mapping each label to `adapter_sha256, n, n_by_family, macro` (mean of per-family accuracy over procedural families; the harvest key on the selection suite), `by_family` (external sources included), `parsed, truncated, completion_tokens`.

## Blind guard and harvest

Only `evaluate.py` opens a blind vault: the `Suites` class used by workers never reads the blind lists, and `tasks.blind_suite` and `external.items(..., part="blind1"|"blind2")` require the receipt returned by `evaluate.blind_guard`. The guard refuses unless `docs/harvest/<run>.json` is tracked in git, identical to `HEAD`, and lists every requested adapter's `adapter_model.safetensors` SHA-256 in its `ranking`. The base model is allowed without a manifest (calibration). Before returning, the guard appends `| time | vault | revision | manifest sha256 | adapters | per family x samples | run |` to `docs/blind-ledger.md`; commit the ledger with the results.

`harvest` ranks lineages with status `done` by `macro` on the selection evaluation of the given tag (every finished lineage must have been evaluated completely with its final adapter) and writes `{run, time, revision, tag, outlier, top3, ranking: [{lineage, slot, arm, scaffold, selection_macro, adapter, sha256, rank}]}`.

## Measured costs

Qwen3-1.7B on one L20 next to a resident vLLM server (KV 20G), no co-tenant on the card. The update benchmark (`scripts/bench_update.py`) uses one realistic amplification batch: the base model's 64 prompts × 4 rollouts on the first training step (prompts 273 tokens on average, completions 83 on average and at most 640), with every sequence given a non-zero weight so that all 256 enter the update.

| Update path | Seconds | Peak allocated |
|---|---|---|
| full fp32 logits over whole padded sequences, micro-batch 4 (8 runs out of memory), fp32 LoRA matmuls | 27.0 | 10.9 GiB |
| completion-only LM head, length-sorted 8192-token micro-batches, bf16 LoRA matmuls | 19.0 | 6.8 GiB |
| the same with shared-prefix rows (default) | 9.9 | 6.8 GiB |

Along the way: with fp32 LoRA matmuls and a 16384-token budget the engine's plain path took 27.2 s and the shared-prefix path 15.5 s; bf16 LoRA matmuls brought the latter to 12.6 s and the smaller budget (less padding) to 9.9 s. Budgets of 4096 and 6144 tokens measured 8.4–10.5 s, within noise of 8192. Without gradient checkpointing even a 4096-token budget runs out of memory next to vLLM. The bf16 LoRA path also agrees better with vLLM (per-token mean |Δ| 0.0057 against 0.014 with fp32 LoRA matmuls).

A default-size run with three members on one GPU (plain, verify scaffold, perturbation; 64 prompts × 4 rollouts, at most 640 tokens) spent, in its first amplification steps, 17–20 s on the joint rollout batch and per member 2.2–4.8 s on the update (only 5–15 of 64 groups had mixed rewards, so 20–60 sequences entered it), 0.2–0.5 s on the consistency check and 0.25–0.4 s on publishing: about 30 s per round, or 10 s per member step. Preparing the seeding pool took 164 s per teacher (1536 prompts × 4 samples; 17% of the prompts had a correct plain sample, 27% with the verify scaffold), the warm-up 5–6.5 s per positive update of 64 sequences (2.5–3.3 minutes per member), and a probe of three members (824 prompts each) about 50 s. The trainer process peaked at 10.7 GiB in `nvidia-smi` beside vLLM's 27.4 GiB.

## Known limitations

- The update is the plain on-policy policy gradient of the spec, with no importance correction for the remaining sampler–trainer mismatch (about 0.006 nats per token with the bf16 LoRA path).
- The consistency check detects a wrong member, a base/adapter swap, or a prompt or scaling mismatch once the adapter differs from the base model by more than the noise; it cannot tell two consecutive versions apart. Stale weights are excluded by the version naming, not by the check.
- Rollout sampling is not seeded, so a resumed step draws new rollouts. An interrupted seeding restarts from scratch (seeding has no checkpoints).
- Only training and seeding items are kept disjoint from the train-test probe. For families with tiny instance spaces, evaluation instances of the train suite can coincide with training prompts, and selection evaluation and probe instances can coincide.
- Shared-prefix rows are padded to the longest row of their micro-batch (about 85% useful tokens at the default budget); packing several rows per lane would need block-sparse attention.
- A worker with more members than `server.max_loras` still works, but generation for the extra adapters queues.
- Commands use paths relative to the repository root and must be started there.
- The harvest manifest and the blind ledger are written but not committed by the tools.
