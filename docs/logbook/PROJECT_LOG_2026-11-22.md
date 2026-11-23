# Project log — 2026-11-22

## Setup

- Repository initialised; `origin` points at `git@github.com:leozippon/evo-taste.git`, which does not exist yet. The available token cannot create repositories, so commits are local until the repository is created.
- Compute: 8× NVIDIA L20 (46 GB), shared with another user and with the owner's other projects. Free GPUs varied between zero and five during the day. All jobs must be resumable and sized to one card.
- Environment: uv-managed, vLLM 0.34.0, torch 2.14, PEFT 0.22, reasoning-gym 0.1.25. Verified recipes are in `docs/dev/INFRA_NOTES.md`.

## Decisions

- **Protocol fixed before any training** (`docs/design/protocol.md`): four hypotheses with refutation criteria, train / selection / blind-vault suites, three seeding arms plus a generic-reasoning control, tolerant culling, harvest on the selection suite only, and a guarded, ledgered blind evaluation with two vaults.
- **Base model: Qwen3-1.7B, thinking disabled.** Reasons: about half the cost of the 4B model per token on this hardware; the literature notes report prompt-induced diversity helping at 1.7B and not at 4B (ICE), and the 4B Instruct model saturating within 20 steps on one paper's procedural task. The 4B calibration on our own tasks is still running; the choice is revisited if it contradicts this.
- **Carrier: LoRA rank 32 on all projections.** Members are adapters on a frozen base; many adapters share one vLLM server per GPU.
- **Environment: reasoning-gym at default configurations, binary exact-match reward.** Suites are assigned by a seeded draw over families, with near-duplicate families kept together and three whole categories held out for selection and the two blind vaults. Families the base model cannot solve at all stay in the training mix on purpose: unlocking one is exactly the kind of rare, self-reinforcing event the project is looking for.
- **External blind benchmarks**: GSM8K test, MATH-500, MMLU-Pro, split between the two vaults; GSM8K train and a third MMLU-Pro split serve as the non-blind foundation probe. `gsm_symbolic` is excluded from the procedural suites so that GSM8K stays out of distribution.

## Observations

- **Qwen3-1.7B is bimodal without thinking.** With reasoning-gym's default prompt, 66% of its answers are a bare `<answer>` with no reasoning (median 17 tokens); its mean score over 105 families at default difficulty is 0.21, counting partial credit, from 60 instances per family. Whether reasoning helps depends on the family: it raises accuracy on `gcd` (0.39 → 0.90), `gsm_symbolic` (0.01 → 0.61) and `base_conversion` (0.00 → 0.79), and lowers it on `number_sequence` (0.84 → 0.47). The base policy therefore already contains a "reason or answer directly" disposition that RL can amplify in either direction. A generic-reasoning control scaffold was added so that style-specific scaffolds are compared against members that reason, not against members that mostly do not.
- **The update, not generation, is the bottleneck.** An L20 trains the 1.7B model at about 4k tokens/s, so one update over 256 rollouts costs roughly 25–30 s, against about 10 s to generate them. Budget consequence: a population of about 30 members for 200 steps is roughly 50–70 GPU-hours, and single-model baselines can be matched to about a quarter of the population's compute within a day on one card each.
- **Literature** (`docs/references/literature-anchors.md`): three of the anchor labels do not match the papers' content (2607.01208 is a bias-auditing paper; 2603.14473 and 2611.02234 concern research-idea taste). The most relevant facts for the design are TAR's result that equally rewarded modes become under-sampled and rarely recover, the primacy-bias paper's finding that RL moves a model far less than supervised fine-tuning does, and the solved-set overlap and preserved/expanded/shrunk accounting used elsewhere as population diagnostics.

## Engine and audit

- Engine built to `docs/dev/ENGINE_SPEC.md` (operating notes in `docs/dev/ENGINE_NOTES.md`). A member step costs 35–45 s once members reason at length; generation, not the update, then dominates.
- Independent audit (`docs/dev/audits/ENGINE_AUDIT_2026-11-22.md`): update, policy consistency, seeding and resume hold. Two defects were found and fixed before the main run: scorers that reward wrong answers (`coin_flip` accepts any prefix of the gold value, `complex_arithmetic` accepts "nan", `time_intervals` accepts times off by a minute) and task instances that depended on the Python hash seed, so that separate processes scored different questions under the same id. The three families were removed from their suites without re-dealing the others; final suite sizes are 31 train, 21 selection, 18 and 20 blind. Scoring now runs in killable child processes, the evaluation summary checks adapter identity and grid completeness, identical question texts in external benchmarks stay in one split, and baselines can live in a run without being ranked.
- A transient HTTP read error killed the first pilot preparation; requests are now retried a bounded number of times.

## Pilot 0 (12 members, 50 amplification steps, 640-token cap)

Members differed in seeding bias and learning rates, so this was a calibration, not a population. Probe accuracies below are differences from the base model in points, with paired standard errors of about 2; the base scores 12.9 on fresh train-family instances, 23.2 on selection families and 49.3 on the foundation probe.

- **Seeding sets very different starting points.** Imitating correct solutions written under a generic "reason step by step" instruction lifts all three probes by 10–14 points before any RL; the `verify` habit lifts them by 7–10; plain imitation changes nothing; the `terse` habit produces bare answers and loses 31 points on the foundation probe; a random perturbation loses 12 there.
- **RL then pulls everyone to the same macro-strategy.** Within 25 steps every member, including the plain, perturbed and terse ones, moved from mostly answering directly to reasoning at length (share of direct answers from about 0.8 to 0), and by step 50 all but the terse member sit in one band on the selection probe (+12.5 to +17). The large initial differences in fitness are erased rather than amplified: "reason before answering" is a shared attractor of this environment.
- **Beneath that, lineages become more distinguishable.** A text classifier trained on a member's answers to train-family probes and tested on its answers to selection-family probes identifies the member 20% of the time after seeding, 42% at step 25 and 47% at step 50 (chance 8.3%), with answers and digits masked. The seeded `verify` habit is retained and strengthened without the scaffold: verification markers rise from 0.5–0.6 to 0.8 per 100 words (other members: about 0.2), and the classifier's recovery of the verify class rises from 23% to 56% between steps 25 and 50.
- **Learning rate.** 1e-5 is inert over 50 steps (reward slope +2.9 ± 2.4 points per 100 steps), 5e-5 learns steadily (+8 to +15), 2e-4 learns as fast but with more truncated and unparseable output early on; none collapsed. The main run uses 1e-4.
- **Token cap.** Once members reason, 25–33% of their answers hit the 640-token cap. The main run uses 1024 so that long-winded habits are not eliminated by the cap alone; accuracy under smaller budgets is reported from the same samples.
- **No regression that would call for targeted protection** appeared in 50 steps: foundation accuracy rose with the rest. The mechanism stays out of the engine until the main run shows a need.

Reading: the pilot already separates two things the thesis runs together. Fitness-relevant strategy converges under a shared reward; style diverges underneath it. Whether the divergent part carries any transfer benefit is exactly what the main run and the blind vault must decide.

## Main run 1 (started)

`configs/run/main1.yaml`: 36 members — 12 scaffolds × 2 seeds, 6 generic-reasoning controls, 3 plain, 3 perturbation — for 120 steps of 128 prompts × 4 rollouts at lr 1e-4, probes every 15 steps, plus two single-model baselines with six members' compute (720 steps; or 192 prompts × 16 rollouts). Full compute matching (36×) is not affordable on this hardware; the comparison is made at the six-member budget against the best of random six-member subsets, and the long baseline's checkpoints give the curve at smaller budgets. Thirty-two members started on four GPUs; the rest are queued for free cards.

## Main run restarted with a penalty for unfinished answers

- **Observation.** The long baseline of main run 1 runs several times faster than the population and so previews its dynamics. Over its first 50 steps the share of training rollouts cut off at the 1024-token cap rose from 15% to 37% and mean length from about 500 to 700 tokens, while reward still rose (0.28 → 0.34). The cut-off answers concentrate on families the model never solves (`survo`, `cryptarithm`, `kakurasu`, `modulo_grid`: every rollout), where it second-guesses itself until the cap. At a median of 12 steps, 22 of the 36 population members showed the same drift (cut-off share 11% → 18%, length 374 → 506 tokens). The pilot had shown the same expansion against a 640-token cap, so the length grows to whatever cap is set.
- **Cause.** With a group-relative update, a prompt whose rollouts all fail contributes nothing. Habits that pay on solvable prompts — rechecking, not committing — spread to unsolvable ones, where nothing counteracts them. This is frontier growth causing a regression elsewhere, the situation the protocol's targeted protection is meant for.
- **Remedy.** A rollout cut off at the cap now has return −0.5 instead of 0 in the group update (`amplification.truncation_penalty`); nothing else changes, and accuracy is still measured on the binary reward. Tested first on the two baselines: within 25 steps the cut-off share fell from 15% to under 1%, length from 350 to 330 tokens, parsed answers rose to 99%, and probe accuracy at step 15 matched the unpenalised run (train 27.4 vs 28.6, selection 31.0 vs 31.0, foundation 66.3 vs 66.0).
- **Decision.** The population was stopped at about step 12 and restarted with the penalty as `main1-tp`, forked from `main1` after seeding so that both share seeding pools, base probes and post-seeding adapters. `runs/main1` is kept as the record of the unpenalised dynamics. The decision used training statistics and in-loop probes only; no final selection evaluation and no blind evaluation had been run. `main1-tp` is the main run of the protocol.

## Running on a crowded machine

- **Worker deaths were GPU collisions.** Three workers of `main1-tp` died within an hour with nothing in their logs. One death coincided to the second with another user's job starting on the same card; the workers then left only 6–10 GiB free per card, and at that point the run held seven of the eight GPUs, so other users' launchers had nowhere else to go. A suspected false alarm of the sampler/trainer consistency check on very short answers was also removed (the check is now sized by tokens), although it was never observed to fire.
- **Changes.** The run now holds five cards instead of seven, with ten members on two workers and both baselines on one. Workers checkpoint every 5 steps, retry an update after an out-of-memory error instead of dying, log whatever ends them, and run with a 12 GiB KV cache (about 29 GiB per worker instead of 37), so that an 8–15 GiB neighbour fits. `scripts/supervise_run.sh` restarts any worker that still dies, on whichever allowed card has room. These are operational settings passed at launch; the frozen run configuration and the training computation are unchanged.
- **A 15–30% slowdown was our own.** Saving an adapter took up to 80 s in a fraction of steps because each worker ran PyTorch's CPU work on 96 threads and the workers starved each other. Workers now use 4 threads.
- **Cost.** Restarts and consolidation cost roughly two hours of population progress; at 22:10 the population is between steps 3 and 20.
- **Harvest step fixed in advance.** Because the run may not finish, the protocol now states the stopping rule and how the harvest step is chosen (`docs/design/protocol.md`, "Harvest rule"); the harvest tool can rank members at a common step.

## Open

- Final evaluation plan: selection and train suites at 200 instances per family × 2 samples for every final adapter and the long baseline's checkpoints, then harvest, then blind vault 1 with the same sizes and 650 items per external source.
- `origin` still does not exist; nothing has been pushed.
