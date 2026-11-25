# Engine notes

Internal operating notes for the EvoTaste engine (`evotaste/` minus the task layer and the verifiers). The scientific design is `docs/design/round-2.md`, the build contract `docs/dev/ROUND2_SPEC.md`, the vLLM and trainer recipes `docs/dev/INFRA_NOTES.md`. The task layer — sources, suites, the probe set, the training stream and scoring — is described in `docs/dev/ENV_NOTES.md`, the verifiers and the sandbox in `docs/dev/VERIFY_NOTES.md`; the engine uses them only through the contract (`Task`, `Score`, `score_many`, `prompt_ids`, `Streams`, `fixed_tasks`). This file holds the commands and how to launch them on the shared machine, the run-config schema, the mechanisms whose details matter for results, the on-disk layout and record schemas that analysis code is written against, measured costs, and known limitations. Round 1's engine is reproducible from the tag `round-1`.

## Commands

All commands run from the repository root after `source scripts/env.sh`, with exactly one GPU visible (`CUDA_VISIBLE_DEVICES`). The script also sets `PYTHONHASHSEED=0`, which child processes inherit and `evotaste.tasks` requires. Each command that generates text starts its own vLLM server (1–4 minutes with warm compile caches) and stops it on exit; the server also dies if its parent is killed with SIGKILL. `--attach http://127.0.0.1:<port>` uses an already running server of the same model instead (development only; see `scripts/dev/serve_vllm.sh`).

| Step | Command |
|---|---|
| Prepare a run | `python -m evotaste.worker prepare configs/run/<name>.yaml [server overrides]` |
| Fork a run after seeding | `python -m evotaste.worker fork runs/<source> configs/run/<new>.yaml [--dry-run]` |
| Run or resume a worker | `python -m evotaste.worker run runs/<name> --members m00,m01 [--checkpoint-every 5] [server overrides]` |
| Evaluate | `python -m evotaste.evaluate run runs/<name> --suite selection --adapters base,population@60,m00@60 --per-family 50 [--per-source 300] [--samples 2] [--shard i/n] [--tag final] [server overrides]` |
| Harvest | `python -m evotaste.evaluate harvest runs/<name> --tag final [--step S]`, then commit `docs/harvest/<name>.json` |
| Context score | `python -m evotaste.evaluate context-score runs/<name> [--members m00,m01]` (no vLLM; the base model on the visible GPU) |

Server overrides are `--kv-cache SIZE`, `--max-loras N` and `--gpu-memory-utilization F` (see Run configuration). Resuming is the same `run` command again. In practice workers and evaluations are started through the scripts below rather than by hand.

## Launching on the shared machine

Other users start 8–30 GiB jobs on every card without warning, including cards our workers use, so no card stays exclusive. Three scripts handle this; each runs from the repository root after `source scripts/env.sh` and keeps its log and control files in `logs/<name>/`.

- `scripts/supervise_run.sh runs/<name>` is the way to train. It keeps every member list of a run alive: on each pass it starts a list that has unfinished members and no live worker on one of the allowed cards with enough free memory, and so restarts workers killed by a co-tenant, elsewhere if need be. It reads, on every pass, `lists` (one comma-separated member list per line, one worker each), `gpus` (allowed indices), `worker-args` (extra `worker run` arguments such as `--checkpoint-every 5 --kv-cache 12G --gpu-memory-utilization 0.45 --max-loras 10`), `need-free-mib` (default 40000) and `upgrade-after` (touch it to have every older worker stopped right after its next checkpoint and restarted on the current code and arguments, losing no steps). Stop it with SIGTERM; workers keep running.
- `scripts/queue_workers.sh runs/<name> m1,m2 m3,m4 …` is a one-shot launcher for a quiet machine: it starts each list on the next completely free card and withdraws a worker whose card another user claims within a minute. It does not restart anything; prefer the supervisor.
- `scripts/run_eval.sh runs/<name> <suite> <tag> <adapters> <per-family> <samples> <per-source> <shards>` runs one evaluation as shards, one per card with at least `NEED_FREE_MIB` (default 21000) free that hosts none of our workers or shards, each with `--kv-cache 12G --gpu-memory-utilization 0.42`; a shard that dies is started again (up to 6 attempts) and the script exits 0 when every shard has finished.

Put a clone in the same member list as its parent: a worker whose only unfinished members are clones waiting for a parent that another worker advances keeps its vLLM server and polls every minute until the parent has committed the clone point.

Every command writes whatever ends it to its own log before it exits: the traceback of an exception (`terminated by an exception`, or `exiting: <message>` for a refusal), or `received SIGTERM` / `received SIGINT` and `stopped by …` for a deliberate stop. Keep stderr (the scripts append it to `stderr-gpu<g>.log`), since an interpreter-level failure before logging starts can only appear there. A worker also logs whether the working tree was clean at launch; if not, it logs the modified files and the SHA-256 of `git diff HEAD` and keeps the diff as `logs/<name>/uncommitted-<hash>.patch`, so a run from uncommitted code can be reconstructed.

## Run configuration

A run config is YAML; unknown keys, missing required keys and wrongly typed values are errors. `prepare` freezes it into `runs/<name>/config.yaml`, with `suites`, `contexts` and `strata.path` pointing at copies placed next to it, and every later command reads only that copy. `configs/run/pilot-template.yaml` writes out the pilot defaults and shows every way a member can start; `configs/run/toy.yaml` is a small end-to-end run.

| Key | Default | Meaning |
|---|---|---|
| `name` | required | run directory `runs/<name>/` and log directory `logs/<name>/` |
| `model` | required | Hugging Face id, e.g. `Qwen/Qwen3-1.7B` |
| `seed` | required | seed of the training stream and the seeding prompts (probe and evaluation sets do not depend on it) |
| `suites`, `contexts` | `configs/suites-r2.yaml`, `configs/contexts.yaml` | copied at prepare time |
| `strata.path`, `strata.sha256` | none | the census strata table and its pinned SHA-256; required once any member has amplification steps; checked and copied at prepare time |
| `server.kv_cache` | `20G` | vLLM `--kv-cache-memory-bytes` |
| `server.gpu_memory_utilization` | 0.65 | with a fixed KV budget only vLLM's startup check (free memory ≥ fraction × card) |
| `server.max_num_seqs`, `server.max_num_batched_tokens` | 512, 8192 | vLLM scheduler limits |
| `server.max_loras` | 8 | adapters that can share a batch; set it to the number of members of a worker |
| `lora.rank`, `lora.alpha` | 32, 64 | LoRA on all seven projection types |
| `trainer.token_budget` | 8192 | padded tokens per micro-batch (rows × longest row), and the most a packed row holds |
| `trainer.shared_prefix` | true | a prompt once, followed by as many of its completions as fit in the token budget |
| `trainer.grad_checkpointing` | true | needed to fit next to vLLM |
| `trainer.max_grad_norm`, `trainer.weight_decay` | 1.0, 0.0 | AdamW (betas 0.9/0.999, eps 1e-8) |
| `trainer.check_tokens`, `trainer.check_rollouts` | 3000, 8 | minimum completion tokens and rollouts re-scored by the trainer before every update |
| `seeding.method` | `opcd` | `opcd` (on-policy context distillation), `imitation` (of correct teacher samples) or `none` |
| `seeding.steps`, `.batch`, `.lr` | 40, 64, 1e-4 | updates, prompts per update, learning rate |
| `seeding.max_tokens` | 4096 | length limit of the member's samples (opcd) or the teacher's (imitation) |
| `seeding.domains` | null (all) | domains of the seeding prompts, e.g. `[math]` |
| `seeding.samples` | 4 | imitation: teacher samples per prompt |
| `amplification.steps`, `.prompts`, `.rollouts`, `.max_tokens`, `.lr` | 60, 64, 4, 4096, 1e-4 | group updates; sampling is always at temperature 1, top-p 1 |
| `amplification.length.free`, `.slope` | 2048, 0.25 | the length deduction of the return (below) |
| `amplification.advantage` | `centered` | `centered` or `standardized` (Round 1's rule, for comparison) |
| `amplification.self_imitation` | 0.0 | β, the weight of the self-imitation term |
| `probes.every`, `probes.max_tokens` | 15, 4096 | probe and checkpoint every E steps and after the last step; the limit is also used by `evaluate` |
| `culling.enabled` | true | false: would-be culls are recorded and logged, nobody is culled or replaced |
| `culling.grace`, `culling.max_replacements` | 40, 2 | no culling at probes before this step; new lineages per slot before the slot is given up |
| `members` | required | list of `{id, seed, context?, parent?, overrides?, role?}` |

A member id uses letters, digits, `_` and `-`. `seed` is the lineage seed (LoRA initialisation, every sampling seed, imitation draws). `context` names an entry of the contexts file (`placebo`, a stance or a tactic); it is required for seeding by `opcd` or `imitation` and forbidden for `none`. `role` is `member` (the default, a population member) or `baseline` (trains, probes and evaluates like a member but is never ranked). `overrides` may replace keys of `seeding`, `amplification` and `culling` for one member; nested sections such as `length` are merged key by key. A member with `amplification: {steps: 0}` stops after seeding and its step-0 probe.

`parent: {member, step}` makes a member a clone: it starts from that member's checkpoint at that step and continues it (see Members). A clone takes its parent's context and effective settings, with its own `amplification` and `culling` overrides on top; it may not set `context` or seeding overrides, its parent may not itself be a clone, the step must lie within the parent's course and below the clone's own `amplification.steps`, and its seed must differ from its parent's and from every other clone of the same parent and step.

Launch-time settings do not alter the frozen run and never change results. `--kv-cache`, `--max-loras` and `--gpu-memory-utilization` replace the `server` keys for one process (`worker run`, `worker prepare`, `evaluate run`; logged at start); `--checkpoint-every` is described under Resilience. Measured vLLM footprints for Qwen3-1.7B at rank 32 are about the KV budget plus 6.8 GiB (26.8 GiB at 20G, 18.9 GiB at 12G, about 14.9 GiB at 8G; two more LoRA slots add about 0.15 GiB); the trainer's peak is in Measured costs. Lower `--gpu-memory-utilization` with the KV budget (about 0.45 at 12G, 0.35 at 8G), or vLLM's startup check fails beside a co-tenant. Nothing else sizes itself from the KV budget: the client keeps up to 512 requests in flight and the trainer's token budget is independent; a small cache only makes vLLM pause and recompute sequences when answers are long, which at 4,096 tokens happens often (a 12G cache holds about 112k tokens, some 25 full-length sequences).

## Members

A member starts in one of two ways.

**Seeding** (every non-clone member). The LoRA tensors are drawn from the lineage seed with B = 0, so the member starts as the base model; then `seeding.steps` updates, each on `seeding.batch` prompts of `Streams.seed_batch(i, batch, domains)`, the same prompts for every member with the same batch and domains:

- `opcd`: the member samples one completion per prompt behind its own prompt (no context), the trainer re-scores a sample of them (the consistency check below), and one AdamW step minimises the token-mean reverse KL over completion tokens, Σᵢ Σₜ KL(π(·|xᵢ, yᵢ<ₜ) ‖ base(·|cᵢ, yᵢ<ₜ)) / Σᵢ |yᵢ|, where xᵢ is the member's prompt, cᵢ the prompt that carries the context (the context text, framed by the file's preamble and closing, appended to the system prompt by `prompt_ids(..., context=...)`), yᵢ the member's own sample, and `base` the frozen base model: the same network with the adapter disabled and no gradient. The KL is over the full vocabulary at every completion position; the gradient does not flow through the choice of the sampled tokens (as in on-policy distillation generally). The new weights are published before the next update's sampling. The KL of every update is logged.
- `imitation` (Round 1's method with a context): the base model answers each prompt `seeding.samples` times behind the prompt that carries the context; the correct, stopped answers form the update's pool, and `batch` sequences are drawn from it with the lineage seed (a prompt with a correct answer uniformly, then one of its correct answers) for one positive update behind the member's own prompt. An update without any correct answer stops the worker.
- `none`: no update; the unseeded control.

All new members of a worker seed in lockstep: round i makes update i of every member that has one, with one joint generation. Amplification then starts from a fresh AdamW state, after the step-0 probe and checkpoint.

**Cloning.** A clone waits until its parent (the parent slot's first lineage) has committed the clone point, then loads the parent's checkpoint at that step, fp32 adapter and AdamW state, takes the parent's step counter, probes at that step with its own seed, commits its own copy of the checkpoint, and continues on the shared training stream: the same prompts as its parent and siblings at each step, different sampling seeds. Because the parent's state is copied exactly, two clones of one parent and step differ only in their sampling seeds (and in vLLM's batch-dependent numerics), and so does a clone from its parent's continuation. A parent writes a checkpoint at every clone point that a member spec names (a light one if it is not a probe step) and keeps its optimizer state; a parent culled before the clone point makes the clone's worker stop with an error. A clone at step 0 is a twin of the seeded member, which is how the template trains one seeded state under both update rules.

**Sampling seeds.** Every request carries an explicit seed, so that two lineages differ by construction and a rerun is as reproducible as vLLM allows (its draws are a function of seed and position, but batch composition changes bf16 numerics). Seeds are `stable_seed(tag, …)` of: `("opcd", lineage seed, update, prompt index)` and `("imitation", …)` for seeding; `("rollout", lineage seed, step, prompt index)` for amplification (vLLM gives the k-th of n samples seed + k); `("probe", lineage seed, step, item index)` for probes; `("probe", "base")` and `("probe", "base", context)` plus the item index for the base model's reference answers; and `("eval", adapter content hash or "base", uid)` for evaluation, the same in every shard and under any label of the same adapter. A replacement lineage after a cull gets the seed `stable_seed("lineage", slot seed, k)`.

## Returns, advantages and the update

The reward is the task layer's (`Score.reward`: 1 iff the answer is in the wrapper asked for, verified correct, and the completion stopped). The return adds the length slope: `return = reward − slope · clip((tokens − free) / (max_tokens − free), 0, 1)`, with `tokens` the completion's sampled ids (end-of-turn included), so nothing is deducted up to 2,048 tokens and 0.25 at the limit. With `centered` advantages a rollout's weight is (return − group mean) / 0.5; a group whose returns are all equal gets exactly zero, and a length difference alone gives weights proportional to it (two otherwise equal answers of 2,560 and 3,072 tokens: about ±0.06). Dividing by the group's spread instead (`standardized`) would make any such difference a full-size signal. Records, probes, culling and evaluation stay on the binary reward; only the update sees returns.

One AdamW step with gradient clipping minimises

  −Σ_{i: wᵢ≠0} wᵢ Σₜ log π(yᵢₜ) / Σ_{i: wᵢ≠0} |yᵢ|  +  β · (−Σ_{i: rᵢ=1} Σₜ log π(yᵢₜ) / Σ_{i: rᵢ=1} |yᵢ|),

each term normalised by its own token count, the second over every rollout of the step's batch with reward 1 whatever its advantage (β = `self_imitation`). With β = 0 the second term and its flags are not computed at all, so the update is exactly the advantage-only one. A step in which neither term has a sequence leaves the weights and the optimizer untouched. The history records both terms (`adv_loss`, `si_loss`) and the sequences and tokens of each.

The trainer scores exactly the sampled token ids behind exactly the prompt ids sent to vLLM; on GPU its LoRA matmuls run in bf16 like vLLM's, over fp32 master weights. Rows hold a prompt once followed by as many of its completions as fit in `token_budget`, with a mask that lets each completion see only the prompt and itself; a completion that does not fit starts a new row, so at 4,096 tokens most rows hold one completion and memory is bounded by the budget, not by the group. Output projections run only at completion positions, in chunks recomputed in the backward pass: 1,024 positions for sampled-token log-probabilities, 256 for the reverse KL, whose chunk holds a few fp32 [256, 151,936] tensors. In the distillation update the teacher's hidden states for a micro-batch are computed first (adapter disabled, no gradient) and only they are kept. Each adapter version is saved in bf16 to a new directory under `/dev/shm/evotaste/<run>/<pid>/` and loaded under a new name `<lineage>@<pid>-<counter>`; the previous version is then unloaded and its directory removed, so no request can be served by a stale version.

Before every amplification update and every `opcd` update the worker re-scores part of the samples with the trainer and compares with vLLM's log-probabilities: the first rollout of each prompt, then the second of each, and so on, until at least `check_tokens` completion tokens and `check_rollouts` rollouts are covered (all of them if the batch is smaller). The sample fails if the per-token mean |Δ| exceeds 0.04 or Σ|ΔΣ| / Σ tokens exceeds 0.015; a failed sample is repeated once on all samples of the step (both results logged), and only a failure there stops the worker with `ConsistencyError`. A wrong, stale or swapped adapter shifts every token and fails both, while an isolated bf16 near-tie token (2–3 nats) cannot. Measured in Round 1 with the engine's own code (random adapter, 128 rollouts of up to 640 tokens): 0.0057 and 0.0022 with the right adapter; 0.12 and 0.073 when the base model is served instead of the adapter or the reverse.

## Probes, culling and the base references

The probe set is the task layer's (`Streams.probe_tasks`: selection items spread over the selection units, plus selection items under held-out wrappers); it is rebuilt by the code at every start, and `prepare`, `fork` and every worker check that it equals, in uid and question, the set the base model answered at prepare time, and stop otherwise. At prepare time the base model answers the probe set plainly (every member's reference) and once behind the prompt of each context in the contexts file (the instructed references of the context score). A probe summary holds the accuracy (mean over families of the family's accuracy), the same by domain, per family, wrapper compliance overall and per wrapper, tolerant correctness (`correct_any`), the truncated share and the mean completion length.

A member is a cull candidate at a probe at step ≥ `grace` (never at step 0) if fewer than half of its probe answers are in the wrapper asked for, more than half are truncated, its probe accuracy is below half the base model's, or its probe accuracy is below the base model's at this probe and at the previous probe at step ≥ `grace`. With culling enabled the lineage is culled and keeps its checkpoints; slot `s` continues with lineage `s~k` (k = 1, 2, …), which starts from scratch (seeding, or for a clone slot a new clone of the same parent point) with seed `stable_seed("lineage", <slot seed>, k)`. With `culling.enabled: false` the reason is recorded in the probe summary with `culled: false`, a warning is logged, and the member carries on.

## Resilience

A worker advances its members until each has finished its steps or its slot is exhausted; killing it at any point and starting the same command again resumes every member from its last checkpoint. Checkpoints are written at every probe, at every clone point of the lineage, and, with `--checkpoint-every N` (default `probes.every`), as light checkpoints without a probe every N steps, so a death costs at most N steps. Resume restores the fp32 adapter and the optimizer state of the checkpoint exactly, discards records of later steps, and continues at the same position of the training stream; it stops if a checkpoint after updates lacks its `optimizer.pt` (for a clone, its parent's updates before the clone point count) rather than silently restarting AdamW. Seeding has no checkpoints: a member killed while seeding restarts seeding from scratch, with the same seeds. Several workers can share a run directory with disjoint `--members` lists; an exclusive lock per member slot (`runs/<name>/locks/<slot>.lock`) makes a second worker on the same member exit at once.

A CUDA out-of-memory error in the trainer (a co-tenant took memory on the card) is retried from clean state: cached memory is released, the card's per-process memory is logged, and the same member step (consistency check, update, or seeding update) is tried again every 100 s, up to 10 attempts, before the error ends the worker. A retry computes exactly the update an uninterrupted run would have: the same samples, and the member's weights and a private copy of its optimizer state are copied into the trainer and back only when an update has completed. An error inside vLLM still ends the worker at once, and a final out-of-memory error logs `nvidia-smi`'s per-process GPU memory before vLLM is stopped. A worker limits its torch CPU threads to 4.

## Forking a run after seeding

`fork` creates a prepared run that shares the source run's base references and seeded members. It copies `base/` (the plain and per-context probe answers), and for every member id present in both configs with the same context, seed and effective seeding settings, not a clone in either, whose source lineage has a committed step-0 checkpoint and complete seeding history, it copies that checkpoint (the adapter after seeding, no optimizer state), the seeding history and the step-0 probe, and writes a state at amplification step 0. Other members start normally; clones start from their parents in the new run. It refuses unless model, run seed, `lora`, `seeding`, `probes`, `trainer` and the suites and contexts content are identical; the strata table, amplification, culling, server settings and member overrides of those may differ. The source may be live: files are copied, never moved or linked. `prepared.json` of the fork records `fork: {source, source_revision, members}`, and `--dry-run` prints the plan without creating anything.

## Evaluation and harvest

`evaluate run` evaluates adapters named as `base`, a lineage id (its last checkpoint), `<lineage>@<step>`, or `population@S`, which stands for exactly the labels a harvest at step S ranks (below) and expands the same way in every shard, on the task layer's evaluation set of a part (`fixed_tasks`): `selection`, or the vault `blind2`. It takes `--per-family` instances of every procedural family and `--per-source` items (default the `--per-family` value; at most what the part holds besides the probe set) of every dataset source, both recorded in `params.json`; the default tag `n<per-family>x<samples>` gains `s<per-source>` when they differ. Answers are generated with `probes.max_tokens`. `--shard i/n` splits the (adapter, item) job list; each shard writes its own record file. After each invocation the summary is recomputed over all record files of the tag and written only if every label comes from a single adapter (one content hash across all files) and covers the requested grid exactly, every (family, item, sample) once; while records are only missing (shards still running) the summary is removed with a warning, and any other inconsistency stops with an error.

`harvest` recomputes the selection summary of the tag from its records with the same checks and writes the manifest `docs/harvest/<name>.json`, to be committed before any blind evaluation. Without `--step` it ranks finished population lineages (role `member`, status `done`) by the selection `macro` of their final adapters (the mean over procedural families, as in Round 1; Round 2's ranking key is decided with the population protocol). With `--step S` every population slot's current lineage (following replacements) with a step-S checkpoint is ranked by the macro of `<lineage>@S`, whatever its status. A slot whose lineage was culled and whose replacement has not reached S, or which has no replacement left, is not ranked but recorded under `culled_slots`; the harvest refuses, listing each gap, if a never-culled slot lacks the step-S checkpoint or a ranked adapter lacks a complete, matching evaluation. Baselines are listed, never ranked, at every step at which they were evaluated under the tag. A run of baselines only harvests with an empty ranking and null `outlier` and `top3`.

Only `evaluate.py` opens the vault: the task layer loads vault items only with the receipt returned by `evaluate.blind_guard`. The guard refuses unless the manifest is tracked in git, identical to `HEAD`, and lists every requested adapter's content hash (over `adapter_config.json` and `adapter_model.safetensors`, both of which vLLM reads) under `ranking` or `baselines`; the base model is allowed without a manifest. Before any generation it appends a line to `docs/blind-ledger.md`, which notes when the vault had already been opened (time and run of the first opening); commit the ledger with the results. The guard still knows Round 1's `blind1`, whose ledger lines it reads, but the task layer no longer evaluates it.

## Context score

`context-score` measures how much of each context a member's uninstructed answers carry. For every subject — the base model's plain probe answers (`base`), its answers under each context (`base:<context>`), and every committed probe of every lineage (`<lineage>@<step>`; `--members` restricts the lineages) — and for every context of the run's contexts file, it computes, by domain, the mean over completion tokens of log p_base(token | prompt with the context) − log p_base(token | plain prompt), on the stored sampled token ids, with the frozen base model on the GPU. The table `runs/<name>/context_score/table.csv` has the columns `subject, lineage, step, context, domain, responses, tokens, delta_sum, delta_mean` (token-weighted: `delta_sum / tokens`). Each subject's rows are cached in `runs/<name>/context_score/<subject>.json` with the SHA-256 of its responses file, so a rerun scores only new or rewritten probes. The index of the design (base model 0, instructed answers 1) is computed in analysis from the `base` and `base:<context>` rows. Probe files past a lineage's committed step are skipped.

## On-disk layout

```
runs/<name>/
  config.yaml, suites.yaml, contexts.yaml, strata.tsv   frozen at prepare time (strata only when configured)
  prepared.json                              {time, revision, contexts, fork?: {source, source_revision, members}}
  base/probe_responses.jsonl.gz, probe_summary.json     the base model's plain probe answers
  base/contexts/<context>.jsonl.gz, <context>.json      its probe answers under each context (summary written last)
  members/<lineage>/
    state.json, history.jsonl, probes.jsonl
    rollouts/step_TTTTTT.jsonl.gz            amplification step T (0-based)
    probe_responses/step_SSSSSS.jsonl.gz     probe after S updates (S = 0: after seeding; a clone's first is its start)
    checkpoints/step_SSSSSS/                 adapter_config.json, adapter_model.safetensors (fp32), optimizer.pt
  eval/<suite>/<tag>/                        params.json, records-<key>-<i>of<n>.jsonl.gz, summary.json
  context_score/                             <subject>.json (cache), table.csv
  locks/<slot>.lock
logs/<name>/                                 one log per command (prepare-, fork-, worker-<members>-, eval-<suite>-,
                                             context-score-), vllm-*.log, stderr-gpu<g>.log, uncommitted-<hash>.patch,
                                             and the scripts' control and log files
docs/harvest/<name>.json                     harvest manifest (tracked)
docs/blind-ledger.md                         blind ledger (tracked)
```

Probe-step checkpoints keep their adapter, a PEFT directory vLLM can load (139 MB for 1.7B at rank 32), for evaluation of `<lineage>@<step>`; a light checkpoint is deleted once a newer checkpoint is committed, unless it is a clone point. Only the newest checkpoint and the clone points keep `optimizer.pt` (278 MB; absent at step 0). `state.json` is rewritten only after a checkpoint is complete and is the resume point: on restart, history lines of amplification steps ≥ S, probe lines and files after S, rollout files of steps ≥ S and unfinished checkpoint directories are deleted, so redone steps replace their old records. Until a stopped run is resumed, records past its `state.json` step remain on disk, so analysis must filter by it. A lineage without a checkpoint (status `new`) restarts from seeding, or from its parent's checkpoint, with its directory cleared.

## Record schemas

Task uids and fields are the task layer's (`<source>/<item id>`; procedural instances `<family>/<dataset seed>/<index>`). All times are UTC ISO strings.

A *response record* has the task's `uid, family, domain, kind, wrapper, stratum, question`, the sample's `text` (completion without special tokens), `completion_tokens` (number of sampled ids, end-of-turn included), `finish_reason` (`stop` or `length`) and `logprob` (vLLM's summed log-probability of the sampled ids), and every field of the task layer's `Score`: `reward, correct_any, complied, channel, truncated, extracted, error`. Files built from it:

| File | One record per | Fields besides the response record |
|---|---|---|
| `rollouts/step_T.jsonl.gz` | rollout | `step` (T), `prompt` (index within the step's batch), `k` (rollout index), `return` (with the length deduction), `weight` (advantage used in the update) |
| `probe_responses/step_S.jsonl.gz` | probe answer | `step` (S), `token_ids` (the sampled ids) |
| `base/probe_responses.jsonl.gz`, `base/contexts/<c>.jsonl.gz` | probe answer | `step` (null), `context` (null or the context), `token_ids` |
| `eval/<suite>/<tag>/records-*.jsonl.gz` | sample | `adapter` (label as given), `adapter_sha256` (null for base), `suite`, `sample` (index) |

`history.jsonl`, one line per update:

- seeding (`phase: "seed"`): `method`, `step` (0-based update), `time`, `completion_tokens` (mean) and `truncated` (share) of the update's samples, the trainer's statistics (`loss`, `grad_norm` before clipping, `sequences`, `tokens`, `micro_batches`, `update_s`, and on GPU `peak_alloc_gib`), and `timing` (`generate_s` of the joint generation, `batch_members`, `update_s`, and for opcd `check_s`, `publish_s`). opcd adds `kl` (the token-mean reverse KL, equal to `loss`) and `check` (as below); imitation adds `correct` (correct teacher answers) and `prompts_with_correct`, and its trainer statistics include the self-imitation fields at zero.
- amplification (`phase: "amp"`): `step`, `time`, `version` (adapter name published after the update), `reward` and `return` (means over rollouts), `reward_by_family`, `completion_tokens` (mean), `complied`, `truncated` (shares), `groups` (prompts with non-zero weights), `prompts`, the trainer's statistics (`loss`, `adv_loss`, `si_loss`, `grad_norm`, `sequences` and `tokens` in either term, `si_sequences`, `si_tokens`, `micro_batches`, `update_s`, `peak_alloc_gib`), `check` (`rollouts, tokens` actually re-scored, `token_mean_abs, seq_mean_abs, seq_abs_per_token`, and `first_attempt` with the failed sample's statistics when the check was repeated on the whole batch) and `timing` (`generate_s`: wall time of the rollout batch shared by `batch_members` members; `score_s, check_s, update_s, publish_s, write_s`: this member's own; `publish`: `save_s, load_s, unload_s, cleanup_s`).

`probes.jsonl` and `base/probe_summary.json` (and `base/contexts/<c>.json`, with `context`): `step` (members only), `time`, `n`, `accuracy`, `by_domain`, `by_family`, `complied`, `correct_any`, `by_wrapper` (compliance), `truncated`, `completion_tokens`, and for members `cull` (null or the reason) and `culled` (whether it was acted on).

`state.json`: `id, slot, generation, context, seed, parent` (null, or `{lineage, step}` for a clone), `status` (`new`, `active`, `done`, `culled`), `step` (updates done at the last checkpoint), `probes` (the probe summaries up to it), `cull_reason`, `replaced_by`.

Evaluation `params.json`: `suite, per_family, samples, per_source`. `summary.json`: the same plus `time` and `adapters`, mapping each label to `adapter_sha256, n, n_by_family, macro` (mean of per-family accuracy over procedural families; the Round 1 harvest key), `by_family` (dataset sources included), `complied, correct_any, truncated, completion_tokens`.

Harvest manifest: `run, time, revision, tag`, `outlier` and `top3` (labels; null for a baseline-only run), `ranking` (entries `lineage, slot, context, selection_macro, adapter, sha256, rank`) and `baselines` (the same without `rank`). With `--step` also `step`, a `step` and an evaluation `label` in every entry, and `culled_slots` (entries `slot`, `culls` with each culled lineage's `lineage, step, reason`, `current` lineage and `current_step`, `status` `young` or `exhausted`). Ledger lines: `| time | vault | revision | manifest sha256 | adapters | per family x samples (per external source) | run <name>[; reuses <vault>, first opened <time> (run <name>)] |`.

## Measured costs

Qwen3-1.7B on one L20, measured on 2026-11-25. The trainer was benchmarked alone on a card (its peak adds to vLLM's fixed footprint above), on random token ids behind 300-token prompts, 64 prompts × 4 rollouts or 64 distillation sequences, LoRA rank 32, default trainer settings:

| Update | Tokens in the loss | Seconds | Peak allocated |
|---|---|---|---|
| amplification, β = 0.5, completion lengths uniform 100–4,096 (244 of 256 rollouts in a term) | 507k | 203–209 | 6.9–7.1 GiB |
| the same without shared-prefix rows | 507k | 184 | 6.8 GiB |
| amplification, β = 0.5, every rollout at 4,096 tokens (216 of 256 in a term) | 885k | 202–203 | 6.6 GiB |
| opcd, 64 completions of 4,096 tokens, teacher prompts 170 tokens longer | 262k | 93–105 | 5.1 GiB |

The peak stays at Round 1's level (6.8 GiB) because rows are capped at the token budget and the output projections are chunked; time grows with the tokens in the loss, at 2,500–4,400 trained tokens per second. At these lengths shared-prefix rows no longer pay: a prompt of 300 tokens is small beside its completions, and packed rows need a masked attention and padding (Round 1's short answers made them twice as fast). The default is kept until the census gives the real length distribution; `trainer.shared_prefix: false` changes speed only. With self-imitation most successful rollouts enter the update even when their group has no advantage, so an update costs up to the whole batch.

The end-to-end toy run (`configs/run/toy.yaml` on the Round 2 task layer, with a provisional strata table in place of the census's; one worker, KV 12G, on cards shared with other users) gave these costs. `prepare`: the 357-item probe set answered plainly and under each of the 12 contexts, 1–2.5 minutes each, 26 minutes in all. A worker spends about 6 minutes rebuilding and checking the probe set before vLLM starts (on a machine at load 55–65), and vLLM 3–5 minutes to start. opcd seeding with 32 prompts per update: generation 33–110 s, consistency check 1–4 s, update 14–25 s (13–20k tokens, peak 5.7–6.0 GiB), publish 1.3–1.6 s. Amplification of four members (16 prompts × 4 rollouts each, β = 0.5, answers 460–810 tokens on average): joint generation 77–111 s, update 12–38 s per member (28–43k tokens, 44–56 of 64 rollouts in a term, peak 6.8–7.1 GiB), consistency check 0.009–0.013 per token. A probe of 357 items: 276 s of generation for two members. An evaluation of 3 adapters on 67 selection items: 2.3 minutes of generation. The context score: about 6 minutes per subject of 357 answers (the plain pass and 12 context passes, some 200k tokens each, at about 20% GPU use on the loaded machine), 2 hours for the toy run's 21 subjects; a pilot with about 150 subjects needs `--members` lists split across cards.

## Known limitations

- The update is the plain on-policy policy gradient, with no importance correction for the remaining sampler–trainer mismatch (about 0.006 nats per token with the bf16 LoRA path).
- The distillation loss ignores how the member's own sampling depends on its weights (the gradient flows through the per-position distributions only), as is usual for on-policy distillation.
- The consistency check detects a wrong member, a base/adapter swap, or a prompt or scaling mismatch once the adapter differs from the base model by more than the noise; it cannot tell two consecutive versions apart, and at seeding update 0 the member is the base model. Stale weights are excluded by the version naming, not by the check.
- Sampling is seeded, but vLLM's numerics depend on batch composition, so a resumed step or a rerun reproduces samples only approximately; two clones in different batches are not bit-identical in the parts of their samples their seeds share.
- Seeding has no checkpoints; an interrupted seeding restarts from scratch.
- A clone needs its parent's first lineage to reach the clone point; with culling enabled a parent culled earlier stops the clone's worker.
- Shared-prefix rows are padded to the longest row of their micro-batch.
- A worker with more members than `server.max_loras` still works, but generation for the extra adapters queues.
- Commands use paths relative to the repository root and must be started there.
- The harvest manifest and the blind ledger are written but not committed by the tools.
