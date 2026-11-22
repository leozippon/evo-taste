# Engine specification

Internal build specification for the first EvoTaste engine. It turns `docs/design/protocol.md` into concrete components; `docs/dev/INFRA_NOTES.md` holds the verified vLLM and trainer recipes this builds on. Numbers marked *default* are configuration values, not constants.

## Shape of the system

One *worker* process owns one GPU. It starts a vLLM server for the base model, holds the base model once more in PyTorch for training, and advances a list of members that it was given. Members are fully independent, so a population run is several workers with disjoint member lists and no communication. Everything a worker does is resumable from disk, because GPUs on this machine are shared and jobs can die.

```
evotaste/
  tasks.py      task instances, suites, sampling streams, scoring, prompt building
  external.py   external benchmarks behind the same interface
  sampler.py    vLLM server lifecycle, async generation client, adapter versions
  trainer.py    PyTorch/PEFT policy: scoring and the update
  member.py     member state: seeding, amplification step, probes, culling
  worker.py     CLI: one GPU, many members, resumable loop
  evaluate.py   CLI: final evaluation on a suite, with the blind guard
configs/        suites.yaml, scaffolds.yaml, run/*.yaml
scripts/        make_suites.py, launch helpers
tests/
```

## Tasks and suites

A task instance has a uid, suite, family, question text, gold answer and whatever the scorer needs. Procedural families come from reasoning-gym at their default configuration. The reward is binary: 1 if the family's scorer returns at least 0.999 for the extracted `<answer>` content, else 0; no extractable answer, a truncated completion or a scorer exception all give 0. The raw score and a `parsed` flag are recorded as well.

**Suite assignment** is produced once by `scripts/make_suites.py` from the Qwen3-1.7B calibration and written to `configs/suites.yaml`, which every other component reads. The script records the calibration statistics it used inside the yaml, so the file is reproducible from the repository alone. Rules:

1. Exclude a family if it needs special configuration (`composite`), its mean prompt exceeds 700 tokens, its scorer is defective (gold answer not scored 1.0, or junk scored as correct — `game_of_life_halting`), the base model is at ceiling (exact rate ≥ 0.9), or it duplicates an external probe (`gsm_symbolic`). Record each exclusion with its reason.
2. Near-duplicate families are assigned as one unit: {chain_sum, decimal_chain_sum}, {basic_arithmetic, decimal_arithmetic}, {simple_integration, intermediate_integration}, {simple_equations, polynomial_equations}, {sudoku, mini_sudoku}, {gcd, lcm}, {countdown, puzzle24}, {arc_1d, rearc, arc_agi}, {word_sequence_reversal, spell_backward}, {number_sorting, word_sorting}, {simple_geometry, advanced_geometry}, {string_manipulation, string_synthesis, string_splitting, ab}, {rotate_matrix, spiral_matrix, manipulate_matrix}.
3. Three whole categories are held out, one each for selection, blind vault 1 and blind vault 2, drawn at random from {algebra, cognition, graphs, logic}; the fourth joins the general pool. These give each of those suites a far-transfer stratum of a category never trained on.
4. The remaining units are stratified by category (categories with fewer than four units are pooled) and by tier (*floor*: no exact success in calibration; *reachable*: otherwise), shuffled with seed 20261122, and dealt to train, selection, blind 1, blind 2 in the ratio 2:1:1:1 within each stratum.

**Streams.** Instances are drawn deterministically from named streams, each a function of (suite, purpose, index) and the run seed where relevant: the training stream (step t gives the same prompts to every member), the seeding pool, and fixed probe sets. A training or seeding prompt whose question text equals a probe question is dropped and redrawn.

**External benchmarks** (`external.py`): GSM8K train (foundation probe) and test, MATH-500, MMLU-Pro. Items are split by a seeded hash: MMLU-Pro into foundation, vault 1 and vault 2; GSM8K test and MATH-500 into vault 1 and vault 2. They use the same `<answer>` format; scoring is numeric equality for GSM8K, normalized mathematical-expression equality for MATH-500, and letter match for MMLU-Pro.

**Prompts.** System message: reasoning-gym's default system prompt. User message: the question. Built with the chat template with thinking disabled, as token ids. A seeding teacher additionally gets the scaffold preamble and text appended to the system message.

## Sampler

Wraps the server recipe and hot-reload procedure of `INFRA_NOTES.md`: fixed KV budget, sampling defaults disabled, a new adapter name and a new directory for every adapter version (under `/dev/shm`), previous version unloaded and its directory removed. One call generates for many (adapter, prompt, n) requests concurrently and returns text, token ids, finish reason and summed log-probability per sample. The server is started and stopped by the worker and never outlives it.

## Trainer

Holds the base model in bf16 with one LoRA structure (*default*: rank 32, alpha 64, all seven projection types). A member's LoRA tensors and AdamW state are swapped in and out.

The update is one optimizer step over a set of (prompt ids, completion ids, weight) triples with loss −Σᵢ wᵢ Σₜ log π(yᵢₜ) / Σᵢ |yᵢ| over the sequences with non-zero weight, including the end-of-turn token, with gradient-norm clipping at 1.0 (*default*). Two weightings are needed:

- *group*: within each prompt's K rollouts, w = (r − mean) / (std + ε); groups with identical rewards get zero weight.
- *positive*: w = 1 (reward-filtered imitation; used for seeding).

The update is the throughput bottleneck (about 4k tokens/s on an L20). Compute the LM head only at completion positions, never materialize full-vocabulary fp32 logits for whole sequences, and batch by token budget after sorting by length. A further gain is available because the K rollouts of a prompt share its prefix: run the prompt once with gradient and reuse its keys and values for the K completions. Implement this only with a test showing loss and gradients equal to the plain path within numerical tolerance; otherwise keep the plain path.

## Member life cycle

A member is `{id, arm, scaffold, seed}` plus per-member overrides of run settings (this is how the long and wide baselines are expressed).

**Seeding.** For each distinct teacher condition (no scaffold, or one scaffold) the base model answers the seeding pool once (*default*: 1536 prompts × 4 samples); correct, untruncated samples are cached per run. A member's warm-up is a number of *positive* updates (*default*: 30 steps of 64 sequences) on samples drawn with its seed: first a prompt uniformly among prompts with a correct sample, then one of its correct samples. The training prompt never contains the scaffold. The perturbation arm first initializes LoRA B from a zero-mean normal (*default* σ = 0.01) with the member's seed.

**Amplification.** Step t: take the stream's prompts for t (*default* 64), sample K rollouts each (*default* 4) at temperature 1 with at most 640 new tokens, apply one *group* update, publish the new adapter version. Rollouts for all of a worker's members are generated in one batch; updates run one member after another.

**Probes.** A fixed probe set — fresh instances of every train family, instances of every selection family, and the foundation probe (*default*: 8 per train family, 12 per selection family, 150 per foundation source), one sample each at temperature 1 — is answered by the base model once and by every member after seeding and every E steps (*default* 20). All responses are stored; they are also the raw material for the style diagnostics.

**Culling.** After a grace period (*default* 40 steps), a member is culled when its train-test probe accuracy is below the base model's on two consecutive probes, or less than half of its probe answers are parseable, or more than half are truncated, or its foundation accuracy is below half the base model's. The replacement is a new lineage of the same arm and scaffold with a new seed; it starts from seeding and runs its own full course of steps.

**Persistence.** Per member: a history line per step (reward overall and per family, completion length, parsed and truncation rates, loss, gradient norm, number of contributing groups, timing); every rollout and probe response (compressed jsonl); and at every probe step a checkpoint of adapter, optimizer state and member state. A restarted worker resumes each member from its last checkpoint.

## Final evaluation and the blind guard

`evaluate.py` scores a list of adapters (or the base model) on a suite with a given number of instances per family and samples per instance, writes per-sample records and a summary, and can be sharded across GPUs.

For a blind vault it refuses to run unless a harvest manifest for the run is tracked in git and unmodified, and every adapter it is asked to evaluate is listed there with a matching content hash; base-model evaluation is exempt. Each blind invocation appends a line to `docs/blind-ledger.md` (time, vault, revision, manifest hash, adapters, sizes) before any generation starts.

## Tests

Keep tests to conditions that must hold: scoring guards (no answer, truncation, scorer exception); stream determinism and disjointness from probes; suite disjointness including near-duplicate units; group weights; plain versus shared-prefix update equivalence on a tiny random model; resume from checkpoint; blind-guard refusal. Finish with a real end-to-end run on one GPU at toy size — three members (plain, one scaffold, perturbation), seeding, a few steps, a probe, a kill and resume, then a final evaluation on the selection suite.
