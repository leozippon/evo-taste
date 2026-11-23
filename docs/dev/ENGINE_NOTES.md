# Engine notes

Internal operating notes for the EvoTaste engine (`evotaste/`). The build specification is `docs/dev/ENGINE_SPEC.md`, the vLLM and trainer recipes it relies on are in `docs/dev/INFRA_NOTES.md`, and the harvest rule is in `docs/design/protocol.md`. This file holds the commands and how to launch them on the shared machine, the run-config schema, the mechanisms whose details matter for results, the on-disk layout and record schemas that analysis code is written against, measured costs, and known limitations.

## Commands

All commands run from the repository root after `source scripts/env.sh`, with exactly one GPU visible (`CUDA_VISIBLE_DEVICES`). The script also sets `PYTHONHASHSEED=0`, which child processes inherit: some reasoning-gym generators iterate sets of strings, so an instance would otherwise differ between processes, and `evotaste.tasks` refuses to import under any other value. Each command that generates text starts its own vLLM server (1–4 minutes with warm compile caches) and stops it on exit; the server also dies if its parent is killed with SIGKILL. `--attach http://127.0.0.1:<port>` uses an already running server of the same model instead (development only; see `scripts/dev/serve_vllm.sh`).

| Step | Command |
|---|---|
| Assign suites (once per project) | `python scripts/make_suites.py` (writes `configs/suites.yaml`) |
| Prepare a run | `python -m evotaste.worker prepare configs/run/<name>.yaml [server overrides]` |
| Fork a run after seeding | `python -m evotaste.worker fork runs/<source> configs/run/<new>.yaml [--dry-run]` |
| Run or resume a worker | `python -m evotaste.worker run runs/<name> --members m00,m01 [--checkpoint-every 5] [server overrides]` |
| Evaluate | `python -m evotaste.evaluate run runs/<name> --suite selection --adapters base,population@120,wide-cot@120 --per-family 200 [--per-source 300] [--samples 2] [--shard i/n] [--tag final] [server overrides]` |
| Harvest | `python -m evotaste.evaluate harvest runs/<name> --tag final [--step S]`, then commit `docs/harvest/<name>.json` |

Server overrides are `--kv-cache SIZE`, `--max-loras N` and `--gpu-memory-utilization F` (see Run configuration). Resuming is the same `run` command again. In practice workers and evaluations are started through the scripts below rather than by hand.

## Launching on the shared machine

Other users start 8–30 GiB jobs on every card without warning, including cards our workers use, so no card stays exclusive. Three scripts handle this; each runs from the repository root after `source scripts/env.sh` and keeps its log and control files in `logs/<name>/`.

- `scripts/supervise_run.sh runs/<name>` is the way to train. It keeps every member list of a run alive: on each pass it starts a list that has unfinished members and no live worker on one of the allowed cards with enough free memory, and so restarts workers killed by a co-tenant, elsewhere if need be. It reads, on every pass, `lists` (one comma-separated member list per line, one worker each), `gpus` (allowed indices), `worker-args` (extra `worker run` arguments such as `--checkpoint-every 5 --kv-cache 12G --gpu-memory-utilization 0.45 --max-loras 10`), `need-free-mib` (default 40000) and `upgrade-after` (touch it to have every older worker stopped right after its next checkpoint and restarted on the current code and arguments, losing no steps). Stop it with SIGTERM; workers keep running.
- `scripts/queue_workers.sh runs/<name> m1,m2 m3,m4 …` is a one-shot launcher for a quiet machine: it starts each list on the next completely free card and withdraws a worker whose card another user claims within a minute. It does not restart anything; prefer the supervisor.
- `scripts/run_eval.sh runs/<name> <suite> <tag> <adapters> <per-family> <samples> <per-source> <shards>` runs one evaluation as shards, one per card with at least `NEED_FREE_MIB` (default 21000) free that hosts none of our workers or shards, each with `--kv-cache 12G --gpu-memory-utilization 0.42`; a shard that dies is started again (up to 6 attempts) and the script exits 0 when every shard has finished.

Every command writes whatever ends it to its own log before it exits: the traceback of an exception (`terminated by an exception`, or `exiting: <message>` for a refusal), or `received SIGTERM` / `received SIGINT` and `stopped by …` for a deliberate stop. Keep stderr (the scripts append it to `stderr-gpu<g>.log`), since an interpreter-level failure before logging starts can only appear there. A worker also logs whether the working tree was clean at launch; if not, it logs the modified files and the SHA-256 of `git diff HEAD` and keeps the diff as `logs/<name>/uncommitted-<hash>.patch`, so a run from uncommitted code can be reconstructed.

## Run configuration

A run config is YAML; unknown keys and wrongly typed values are errors. `prepare` freezes it into `runs/<name>/config.yaml`, with `suites` and `scaffolds` pointing at copies placed next to it, and every later command reads only that copy. Defaults follow `docs/design/protocol.md` and the spec.

| Key | Default | Meaning |
|---|---|---|
| `name` | required | run directory `runs/<name>/` and log directory `logs/<name>/` |
| `model` | required | Hugging Face id, e.g. `Qwen/Qwen3-1.7B` |
| `seed` | required | seed of the training stream and the seeding pool (probe and evaluation sets do not depend on it) |
| `suites`, `scaffolds` | `configs/suites.yaml`, `configs/scaffolds.yaml` | sources copied at prepare time |
| `server.kv_cache` | `20G` | vLLM `--kv-cache-memory-bytes` |
| `server.gpu_memory_utilization` | 0.65 | with a fixed KV budget only vLLM's startup check (free memory ≥ fraction × card) |
| `server.max_num_seqs`, `server.max_num_batched_tokens` | 512, 8192 | vLLM scheduler limits |
| `server.max_loras` | 8 | adapters that can share a batch; set it to the number of members of a worker |
| `lora.rank`, `lora.alpha` | 32, 64 | LoRA on all seven projection types |
| `trainer.token_budget` | 8192 | padded tokens per micro-batch (rows × longest row) |
| `trainer.shared_prefix` | true | one row per prompt holding the prompt once and all its completions |
| `trainer.grad_checkpointing` | true | needed to fit next to vLLM |
| `trainer.max_grad_norm`, `trainer.weight_decay` | 1.0, 0.0 | AdamW (betas 0.9/0.999, eps 1e-8) |
| `trainer.check_tokens`, `trainer.check_rollouts` | 3000, 8 | minimum completion tokens and rollouts re-scored by the trainer before every update |
| `seeding.pool_prompts`, `seeding.samples`, `seeding.max_tokens` | 1536, 4, 640 | teacher sampling of the seeding pool (shared per teacher, not overridable per member) |
| `seeding.steps`, `seeding.batch`, `seeding.lr` | 30, 64, 1e-5 | positive warm-up updates |
| `seeding.perturb_std` | 0.01 | perturbation arm: LoRA B drawn from N(0, σ) |
| `amplification.steps`, `.prompts`, `.rollouts`, `.max_tokens`, `.lr` | 300, 64, 4, 640, 1e-5 | group updates; sampling is always at temperature 1, top-p 1 |
| `amplification.truncation_penalty` | 0.0 | λ ≥ 0: group weights from the shaped return reward − λ·truncated |
| `probes.every` | 20 | probe and checkpoint every E steps, and after the last step |
| `probes.train_per_family`, `.selection_per_family`, `.foundation_per_source` | 8, 12, 150 | probe set sizes |
| `probes.max_tokens` | 640 | also used by `evaluate` |
| `culling.grace` | 40 | no culling at probes before this step |
| `culling.max_replacements` | 2 | new lineages per slot before the slot is given up |
| `members` | required | list of `{id, arm, seed, scaffold?, overrides?, role?}` |

`arm` is `plain`, `scaffold` (requires `scaffold`, a name from `control` or `scaffolds` in the scaffolds file, e.g. `cot` or `verify`) or `perturb`. Member ids use letters, digits, `_` and `-`. `role` is `member` (the default, a population member) or `baseline` (trains, probes and evaluates like a member but is never ranked). `overrides` may replace keys of `seeding` (except the pool keys), `amplification` and `culling` for one member; the long baseline is `{role: baseline, overrides: {amplification: {steps: <N × steps>}}}` and the wide one `{role: baseline, overrides: {amplification: {prompts: <N × prompts>}}}`. Batch b of a member with `prompts: P` is items `[bP, (b+1)P)` of the run's training stream. `configs/run/toy.yaml` is the example.

Launch-time settings do not alter the frozen run and never change results. `--kv-cache`, `--max-loras` and `--gpu-memory-utilization` replace the `server` keys for one process (`worker run`, `worker prepare`, `evaluate run`; logged at start); `--checkpoint-every` is described under Resilience. Measured vLLM footprints for Qwen3-1.7B at rank 32 are about the KV budget plus 6.8 GiB (26.8 GiB at 20G, 18.9 GiB at 12G, about 14.9 GiB at 8G; two more LoRA slots add about 0.15 GiB), and a worker's trainer holds 8.7–9 GiB (12.2 GiB for the wide baseline's larger batches) whatever its number of members, since only the active member is on the GPU. A worker therefore needs about 28–31 GiB at 12G and 24–27 GiB at 8G of the card's 44.4 GiB. Lower `--gpu-memory-utilization` with the KV budget (about 0.45 at 12G, 0.35 at 8G), or vLLM's startup check fails beside a co-tenant. Nothing else sizes itself from the KV budget: the client keeps up to 512 requests in flight and the trainer's token budget is independent; a small cache only makes vLLM pause and recompute sequences when answers are long.

## Data: streams, draws and the probe-set guard

Instances are pure functions of (stream key, family, occurrence). Stream item j of the training stream (`train/<seed>`) or the seeding pool (`seedpool/<seed>`) takes its family from a seeded permutation of the train families per block of `len(train)` items, so every family appears once per block. Probe sets use stream key `probe` and evaluation sets `eval`; neither depends on the run seed. Every position owns 16 fixed draws, and a position takes its first acceptable draw:

| Stream | A draw is skipped when |
|---|---|
| training stream, seeding pool | its generator raises (reasoning-gym's `maze` gives up on some seeds), or its question equals a train-test probe question |
| probe and evaluation sets | its generator raises, or its own scorer refuses reward 1 to its gold answer |

External items (GSM8K, MATH-500, MMLU-Pro) are taken in a fixed seeded order per part; an item whose gold answer is rejected is replaced by the next item of that order. Probes take the first items of the foundation order, evaluation takes the last ones. Every skip is logged once per process with family, seed and index; a set without such draws is exactly what it was before the rule existed, and if all 16 draws of a position fail the command stops. The training stream and the seeding pool deliberately do not check gold answers: members of finished runs trained on them as they are. Gold answers are checked with a 300 s scorer limit, and a timeout there is an error rather than a rejection, so a set never depends on the machine's load; `check_gold` remains the final guard before any generation. For main1-tp's sets, train at 100 per family redraws one instance (`base_conversion/1039087889/832`), selection at 200/300 and blind1/blind2 at 200/650 none.

External items are split into parts by a seeded hash of their normalised question text, so identical questions under different ids share a part in every source: GSM8K train is foundation; MMLU-Pro is split three ways into foundation, blind1 and blind2; GSM8K test and MATH-500 two ways into blind1 and blind2. The five MMLU-Pro items that repeat a MATH-500 problem are dropped; GSM8K train, GSM8K test and MATH-500 share no question text. Part sizes: GSM8K train 7473 (foundation); GSM8K test 657 / 662 and MATH-500 249 / 251 (blind1 / blind2); MMLU-Pro 4027 / 4081 / 3919 (foundation / blind1 / blind2).

The probe set is rebuilt by the code at every start, and members' probes are compared with the base model's. `prepare`, `fork` and every worker therefore check that the rebuilt probe set equals, in suite, uid and question, the set the base model answered at prepare time, and stop otherwise: a run prepared by code that built different instances cannot be continued.

## Rewards and scoring

The reward is 1 if the completion stopped by itself, contains an `<answer>…</answer>` span (the last one counts), and the family's scorer gives at least 0.999; otherwise 0. Scorers run in eight persistent scorer processes per engine process (about 4,500 answers per second). A scorer exception scores 0 with its message in `score_error`; a scorer call still running after 10 s of wall time (sympy on `1e999999999` would run for minutes inside C code, where no signal reaches it) is ended by killing its process, scores 0 with `score_error` `timeout`, and the process is replaced. Building a family's dataset the first time a scorer process meets it (`acre` precomputes 8,192 items) is outside that limit and has its own bound of 300 s.

`scripts/make_suites.py` keeps only families whose scorer, on 64 fresh instances, gives reward 1 to the wrapped gold answer and 0 to an empty or junk answer, and, checked after the deal so that no other family changes suite, rewards none of the constants 0, 1, nan and inf, no strict prefix that changes the gold answer (same numbers within 1%, or same letters and digits, count as unchanged), and the gold answer between two other gold answers on fewer than half of the instances (this removed `coin_flip`, `complex_arithmetic` and `time_intervals`). External scorers: GSM8K requires exactly one number equal to the gold number; MATH-500 uses math-verify equivalence; MMLU-Pro requires the answer to start with the gold option letter.

## Training

Seeding: the base model answers the seeding pool once per teacher condition (plain, and each scaffold named by a member) at prepare time, keeping correct, stopped samples; a member's warm-up is `seeding.steps` positive updates on samples drawn with its seed (a prompt uniformly among prompts with a correct sample, then one of its correct samples), always behind the plain prompt. The perturbation arm first draws LoRA B from N(0, `perturb_std`). Amplification starts from a fresh AdamW state.

The update minimises −Σᵢ wᵢ Σₜ log π(yᵢₜ) / Σᵢ |yᵢ| over sequences with wᵢ ≠ 0, end-of-turn token included, with one AdamW step and gradient clipping; a step without any non-zero weight leaves the weights and the optimizer untouched. Group weights are (r − mean) / (population std + 1e-6) within each prompt's rollouts, zero for groups with identical rewards. With `truncation_penalty` λ > 0, r is the shaped return reward − λ·truncated (λ = 1: correct 1, wrong but finished 0, cut off at `max_tokens` −1), standardised the same way; it gives a signal against running into the cap on prompts a member never solves. Only the weights change: recorded rewards, probes, culling and evaluation stay on the binary reward, and with λ = 0 the weights are exactly the reward-based ones. The trainer scores exactly the sampled token ids behind exactly the prompt ids sent to vLLM; on GPU its LoRA matmuls run in bf16 like vLLM's, over fp32 master weights. Each adapter version is saved in bf16 to a new directory under `/dev/shm/evotaste/<run>/<pid>/` and loaded under a new name `<lineage>@<pid>-<counter>`; the previous version is then unloaded and its directory removed, so no request can be served by a stale version.

Before every update the worker re-scores part of the step's rollouts with the trainer and compares with vLLM's log-probabilities: the first rollout of each prompt, then the second of each, and so on, until at least `check_tokens` completion tokens and `check_rollouts` rollouts are covered (all rollouts if the batch is smaller). The sample fails if the per-token mean |Δ| exceeds 0.04 or Σ|ΔΣ| / Σ tokens exceeds 0.015; a failed sample is repeated once on all rollouts of the step (both results logged), and only a failure there stops the worker with `ConsistencyError`. A wrong, stale or swapped adapter shifts every token and fails both, while an isolated bf16 near-tie token (2–3 nats) cannot. Measured with the engine's own code (Qwen3-1.7B, random adapter, 128 rollouts of up to 640 tokens): 0.0057 and 0.0022 with the right adapter; 0.12 and 0.073 when the base model is served instead of the adapter or the reverse.

A member is culled at a probe at step ≥ `grace` (never at step 0) if fewer than half of its probe answers are parseable, more than half are truncated, its foundation accuracy is below half the base model's, or its train-test accuracy is below the base model's at this probe and at the previous probe at step ≥ `grace`. The culled lineage keeps its checkpoints; slot `s` continues with lineage `s~k` (k = 1, 2, …), seed `stable_seed("lineage", <slot seed>, k)`, which starts from seeding and runs the full course.

## Resilience

A worker advances its members until each has finished its steps or its slot is exhausted; killing it at any point and starting the same command again resumes every member from its last checkpoint. Checkpoints are written at every probe and, with `--checkpoint-every N` (default `probes.every`), as light checkpoints without a probe every N steps, so a death costs at most N steps. Resume restores the fp32 adapter and the optimizer state of the checkpoint exactly, discards records of later steps, and continues at the same position of the training stream; it stops if a checkpoint after updates lacks its `optimizer.pt` rather than silently restarting AdamW. Several workers can share a run directory with disjoint `--members` lists; an exclusive lock per member slot (`runs/<name>/locks/<slot>.lock`) makes a second worker on the same member exit at once.

A CUDA out-of-memory error in the trainer (a co-tenant took memory on the card) is retried from clean state: cached memory is released, the card's per-process memory is logged, and the same member step (consistency check, update, or seeding update) is tried again every 100 s, up to 10 attempts, before the error ends the worker. A retry computes exactly the update an uninterrupted run would have: the same rollouts, and the member's weights and a private copy of its optimizer state are copied into the trainer and back only when an update has completed. An error inside vLLM still ends the worker at once, and a final out-of-memory error logs `nvidia-smi`'s per-process GPU memory before vLLM is stopped. A worker limits its torch CPU threads to 4: with torch's default of one OpenMP thread per core, the adapter's dtype conversion before each publish stalled for 10–80 s whenever several workers' parallel regions met on the loaded machine.

## Forking a run after seeding

`fork` creates a prepared run that shares the source run's seeding. It copies the seeding pools and base probe answers, and for every member id present in both configs with the same arm, scaffold, seed and effective seeding settings whose source lineage has a committed step-0 checkpoint, it copies that checkpoint (the adapter after seeding, no optimizer state), the seeding history and the step-0 probe, and writes a state at amplification step 0. Other members seed normally. It refuses unless model, run seed, `lora`, `seeding`, `probes`, `trainer` and the suites and scaffolds content are identical; amplification, culling, server settings and member overrides of those may differ. The source may be live: files are copied, never moved or linked. `prepared.json` of the fork records `fork: {source, source_revision, members}`, and `--dry-run` prints the plan without creating anything. A forked member's amplification is the same computation as an unforked one's from step 0; only adapter version names and the unseeded rollout sampling differ, as between any two runs.

## Evaluation and harvest

`evaluate run` evaluates adapters named as `base`, a lineage id (its last checkpoint), `<lineage>@<step>`, or `population@S`, which stands for exactly the labels a harvest at step S ranks (below) and expands the same way in every shard. It takes `--per-family` instances of every procedural family and `--per-source` items (default the `--per-family` value; more than a part holds takes the whole part) of every external source, both recorded in `params.json`; the default tag `n<per-family>x<samples>` gains `s<per-source>` when they differ. `--shard i/n` splits the (adapter, instance) job list; each shard writes its own record file. After each invocation the summary is recomputed over all record files of the tag and written only if every label comes from a single adapter (one content hash across all files) and covers the requested grid exactly, every (family, instance, sample) once; while records are only missing (shards still running) the summary is removed with a warning, and any other inconsistency stops with an error. Blind vaults get the same check.

`harvest` recomputes the selection summary of the tag from its records with the same checks and writes the manifest `docs/harvest/<name>.json`, to be committed before any blind evaluation. Without `--step` it ranks finished population lineages (role `member`, status `done`) by the selection `macro` of their final adapters. With `--step S` it follows the protocol's harvest rule: every population slot's current lineage (following replacements) with a step-S checkpoint is ranked by the macro of `<lineage>@S`, whatever its status. A slot whose lineage was culled and whose replacement has not reached S, or which has no replacement left, is not ranked but recorded under `culled_slots`; the harvest refuses, listing each gap, if a never-culled slot lacks the step-S checkpoint or a ranked adapter lacks a complete, matching evaluation, and prints how many were ranked, young and exhausted. Baselines are listed, never ranked, at every step at which they were evaluated under the tag. A run of baselines only (`role: baseline` for every member) harvests with an empty ranking and null `outlier` and `top3`; its manifest just lists the baseline adapters.

Only `evaluate.py` opens a blind vault: the `Suites` class used by workers never reads the blind lists, and `tasks.blind_suite` and `external.items(..., part="blind1"|"blind2")` require the receipt returned by `evaluate.blind_guard`. The guard refuses unless the manifest is tracked in git, identical to `HEAD`, and lists every requested adapter's content hash (over `adapter_config.json` and `adapter_model.safetensors`, both of which vLLM reads) under `ranking` or `baselines`; the base model is allowed without a manifest. Before any generation it appends a line to `docs/blind-ledger.md`, which notes when the vault had already been opened (time and run of the first opening); commit the ledger with the results.

## On-disk layout

```
runs/<name>/
  config.yaml, suites.yaml, scaffolds.yaml   frozen at prepare time
  prepared.json                              {time, revision, teachers, fork?: {source, source_revision, members}}
  seeding/<teacher>.jsonl.gz, <teacher>.json teacher = plain or a scaffold name
  base/probe_responses.jsonl.gz, probe_summary.json
  members/<lineage>/
    state.json, history.jsonl, probes.jsonl
    rollouts/step_TTTTTT.jsonl.gz            amplification step T (0-based)
    probe_responses/step_SSSSSS.jsonl.gz     probe after S updates (S = 0: after seeding)
    checkpoints/step_SSSSSS/                 adapter_config.json, adapter_model.safetensors (fp32), optimizer.pt
  eval/<suite>/<tag>/                        params.json, records-<key>-<i>of<n>.jsonl.gz, summary.json
  locks/<slot>.lock
logs/<name>/                                 one log per command (prepare-, fork-, worker-<members>-, eval-<suite>-),
                                             vllm-*.log, stderr-gpu<g>.log, uncommitted-<hash>.patch, and the
                                             scripts' supervisor.log, lists, gpus, worker-args, need-free-mib,
                                             upgrade-after, queue.log, eval-<suite>-<tag>.log, eval-stderr-gpu<g>.log
docs/harvest/<name>.json                     harvest manifest (tracked)
docs/blind-ledger.md                         blind ledger (tracked)
```

Probe-step checkpoints keep their adapter, a PEFT directory vLLM can load (139 MB for 1.7B at rank 32), for evaluation of `<lineage>@<step>`; a light checkpoint is deleted once a newer checkpoint is committed. Only the newest checkpoint keeps `optimizer.pt` (278 MB; absent at step 0). `state.json` is rewritten only after a checkpoint is complete and is the resume point: on restart, history lines of amplification steps ≥ S, probe lines and files after S, rollout files of steps ≥ S and unfinished checkpoint directories are deleted, so redone steps replace their old records. Until a stopped run is resumed, records past its `state.json` step remain on disk, so analysis must filter by it. A lineage without a checkpoint (status `new`) restarts from seeding with its directory cleared.

## Record schemas

Task uids are `<family>/<dataset seed>/<index>` for reasoning-gym instances (regenerate with `evotaste.tasks.procedural_task(family, seed, index)`, which also seeds the module-level `random` that `list_functions` draws from) and `<source>/<item id>` for external items (GSM8K: row index in the split; MATH-500: `unique_id`; MMLU-Pro: `question_id`). All times are UTC ISO strings.

A *response record* has the fields `uid, family, question, answer` (gold, as text), `text` (completion text without special tokens), `completion_tokens` (sampled ids including the end-of-turn token), `finish_reason` (`stop` or `length`), `extracted` (content of the last answer span or null), `score` (raw scorer output; 0 when not scored), `reward` (0 or 1), `parsed`, `truncated`, `score_error` (null or message) and `logprob` (vLLM's summed log-probability of the sampled ids). Files built from it:

| File | One record per | Fields besides the response record |
|---|---|---|
| `rollouts/step_T.jsonl.gz` | rollout | `step` (T), `prompt` (index within the step's batch), `k` (rollout index), `weight` (group weight used in the update) |
| `probe_responses/step_S.jsonl.gz`, `base/probe_responses.jsonl.gz` | probe answer | `step` (S; null for the base model), `suite` (`train`, `selection` or `foundation`) |
| `eval/<suite>/<tag>/records-*.jsonl.gz` | sample | `adapter` (label as given), `adapter_sha256` (SHA-256 over `adapter_config.json` and `adapter_model.safetensors`; null for base), `suite`, `sample` (index) |

`history.jsonl`, one line per update:

- seeding (`phase: "seed"`): `step` (0-based warm-up update), `time`, `loss`, `grad_norm` (before clipping), `sequences`, `tokens` (completion tokens in the loss), `micro_batches`, `update_s`.
- amplification (`phase: "amp"`): the same plus `version` (adapter name published after the update), `reward` (mean over rollouts), `return` (mean shaped return; present only when `truncation_penalty` > 0), `reward_by_family`, `completion_tokens` (mean), `parsed`, `truncated` (fractions), `groups` (prompts with non-identical rewards), `prompts`, `check` (`rollouts, tokens` actually re-scored, `token_mean_abs, seq_mean_abs, seq_abs_per_token`, and `first_attempt` with the failed sample's statistics when the check was repeated on the whole batch) and `timing` (`generate_s`: wall time of the rollout batch shared by `batch_members` members; `score_s, check_s, update_s, publish_s, write_s`: this member's own; `publish`: `save_s, load_s, unload_s, cleanup_s`, the parts of `publish_s`).

`probes.jsonl` and `base/probe_summary.json`: `step` (members only), `time`, `n`, `accuracy` (`train`, `selection`, `foundation`; each the mean over families of the family's accuracy), `by_family`, `parsed`, `truncated`, `completion_tokens`, and for members `cull` (null or the reason).

`state.json`: `id, slot, generation, arm, scaffold, seed`, `status` (`new`, `active`, `done`, `culled`), `step` (updates done at the last checkpoint), `probes` (the probe summaries up to it), `cull_reason`, `replaced_by`.

Seeding pool records (`seeding/<teacher>.jsonl.gz`, correct and stopped samples only): `uid, family, prompt_ids` (the plain prompt the member trains on), `completion_ids, text, teacher_logprob`. The summary `<teacher>.json` holds `teacher, time, generate_s, prompts, samples, correct, prompts_with_correct, by_family`.

Evaluation `params.json`: `suite, per_family, samples, per_source` (files written before `per_source` existed mean `per_source = per_family`). `summary.json`: the same plus `time` and `adapters`, mapping each label to `adapter_sha256, n, n_by_family, macro` (mean of per-family accuracy over procedural families; the harvest key on the selection suite), `by_family` (external sources included), `parsed, truncated, completion_tokens`.

Harvest manifest: `run, time, revision, tag`, `outlier` and `top3` (labels; null for a baseline-only run), `ranking` (entries `lineage, slot, arm, scaffold, selection_macro, adapter, sha256, rank`) and `baselines` (the same without `rank`). With `--step` also `step`, a `step` and an evaluation `label` in every entry, and `culled_slots` (entries `slot`, `culls` with each culled lineage's `lineage, step, reason`, `current` lineage and `current_step`, `status` `young` or `exhausted`). Ledger lines: `| time | vault | revision | manifest sha256 | adapters | per family x samples (per external source) | run <name>[; reuses <vault>, first opened <time> (run <name>)] |`.

## Measured costs

Qwen3-1.7B on one L20 next to a resident vLLM server (KV 20G), no co-tenant on the card. The update benchmark (`scripts/bench_update.py`) uses one realistic amplification batch: the base model's 64 prompts × 4 rollouts on the first training step (prompts 273 tokens on average, completions 83 on average and at most 640), with every sequence given a non-zero weight so that all 256 enter the update.

| Update path | Seconds | Peak allocated |
|---|---|---|
| full fp32 logits over whole padded sequences, micro-batch 4 (8 runs out of memory), fp32 LoRA matmuls | 27.0 | 10.9 GiB |
| completion-only LM head, length-sorted 8192-token micro-batches, bf16 LoRA matmuls | 19.0 | 6.8 GiB |
| the same with shared-prefix rows (default) | 9.9 | 6.8 GiB |

Along the way: with fp32 LoRA matmuls and a 16384-token budget the engine's plain path took 27.2 s and the shared-prefix path 15.5 s; bf16 LoRA matmuls brought the latter to 12.6 s and the smaller budget (less padding) to 9.9 s. Budgets of 4096 and 6144 tokens measured 8.4–10.5 s, within noise of 8192. Without gradient checkpointing even a 4096-token budget runs out of memory next to vLLM. The bf16 LoRA path also agrees better with vLLM (per-token mean |Δ| 0.0057 against 0.014 with fp32 LoRA matmuls).

A default-size run with three members on one GPU (plain, verify scaffold, perturbation; 64 prompts × 4 rollouts, at most 640 tokens) spent, in its first amplification steps, 17–20 s on the joint rollout batch and per member 2.2–4.8 s on the update (only 5–15 of 64 groups had mixed rewards), 0.2–0.5 s on the consistency check and 0.25–0.4 s on publishing: about 30 s per round. Preparing the seeding pool took 164 s per teacher (1536 prompts × 4 samples), the warm-up 5–6.5 s per positive update of 64 sequences, and a probe of three members (824 prompts each) about 50 s.

## Known limitations

- The update is the plain on-policy policy gradient of the spec, with no importance correction for the remaining sampler–trainer mismatch (about 0.006 nats per token with the bf16 LoRA path).
- The consistency check detects a wrong member, a base/adapter swap, or a prompt or scaling mismatch once the adapter differs from the base model by more than the noise; it cannot tell two consecutive versions apart. Stale weights are excluded by the version naming, not by the check.
- Rollout sampling is not seeded, so a resumed step draws new rollouts. An interrupted seeding restarts from scratch (seeding has no checkpoints).
- Only training and seeding items are kept disjoint from the train-test probe. For families with tiny instance spaces, evaluation instances of the train suite can coincide with training prompts, and selection evaluation and probe instances can coincide.
- Shared-prefix rows are padded to the longest row of their micro-batch (about 85% useful tokens at the default budget).
- A worker with more members than `server.max_loras` still works, but generation for the extra adapters queues.
- Commands use paths relative to the repository root and must be started there.
- The harvest manifest and the blind ledger are written but not committed by the tools.
