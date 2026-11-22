# Engine audit, 2026-11-22

Audit of the engine at `e9baab8` against seven frozen conditions. Scope: reward integrity, the update, policy consistency, seeding, data isolation and determinism, culling/resume/records, final evaluation and the blind guard. The audit scripts live under `runs/audit/` on the compute host; that directory is untracked. Commands assume `source scripts/env.sh` from the repository root. One GPU was used once, for about five minutes: GPU 5, which showed 0 MiB in use beforehand. That run checked the shared-prefix path in the production configuration (real model, CUDA, bf16 autocast, gradient checkpointing). The existing test suite passes (32 tests).

## Verdicts

| # | Condition | Verdict |
|---|---|---|
| 1 | Reward integrity | violated: the `coin_flip` scorer (selection) accepts wrong answers; the gate itself holds |
| 2 | Update correctness | holds |
| 3 | Policy consistency | holds |
| 4 | Seeding | holds |
| 5 | Isolation and determinism of data | violated: some instances depend on `PYTHONHASHSEED`; small MMLU-Pro text overlap across parts |
| 6 | Culling, resume, records | holds |
| 7 | Final evaluation and blind guard | holds, with two fragilities (one hash per label, completeness of blind summaries) |

### 1. Reward integrity

The gate holds. `tasks.score` (`evotaste/tasks.py:107-114`) returns reward 1 only when all of these are true: `finish_reason == "stop"`, `extract_answer` finds a span (the last one counts), and the raw score is ≥ 0.999. Every reward in the engine goes through it: rollouts and probes through `member.response_record`, the seeding pool through `worker.build_pool`, evaluation through `evaluate.evaluate`. Scorer exceptions return 0 together with an error string (`tasks.py:100-101`). A scorer that swallows the timeout exception also returns 0: `n_queens` and `spiral_matrix` given `9**9**9**9` returned 0 after 10 s with no error recorded. Re-scoring every rollout in `runs/toy` and `runs/scale-test` reproduced every stored reward, and the truncation flags matched (`runs/audit/records_check.py`).

Adversarial answers were tested on six instances of every train and selection family (`runs/audit/scorers.py`). Constant junk answers were tested on 30 evaluation instances per family (`runs/audit/constants.py`). In the train families, the only non-gold answers accepted were harmless lenience: case (`needle_haystack`, `bitwise_arithmetic` hex), a trailing period, a sentence prefix (`survo`, `kakurasu`), and extra trailing words (`letter_jumble`, which still requires every gold word in position). No accepted variant gives an RL policy a cheaper route to reward than solving the task. One selection family is defective: `coin_flip` (see D1). Blind-family scorers were tested with constant answers on freshly generated instances (seed 777, not the vault's evaluation stream; data only, no model; `runs/audit/blind_scorers.py`). None accepted a constant junk answer.

### 2. Update correctness

Holds.

- Loss and sign match ENGINE_NOTES. The loss is −Σ wᵢ Σₜ log π / Σ|yᵢ| over active sequences (`trainer.py:228-262`), and `test_group_update_direction` passes.
- Group weights are computed per request: `asyncio.gather` and the sort of choices by index keep the order (`sampler.py:176,186`), and `member.py:233-241` zips tasks with sample groups. In the artefacts every stored weight equals `group_weights` of its own prompt's rewards, and `history.sequences/tokens` equal the non-zero-weight rollouts and their token counts.
- Scored tokens are the sampled ids behind the cached prompt ids that were sent to vLLM (`member.py:224,237-239`). Completions end with `<|im_end|>`, verified on the seeding records.
- Masks and positions: rows are right-padded; the plain path uses causal attention with no mask; the shared-prefix mask is causal ∧ (same segment ∨ prompt), and positions restart at `n` for every completion (`trainer.py:73-82,196-201`). On the GPU in fp32, shared-prefix log-probabilities and gradients are identical to the plain path. In bf16 both paths differ from fp32 by the same amount: per-token mean |Δ| 0.014 for plain and 0.015 for shared. A negative control with segment isolation removed gives 0.78 (`runs/audit/gpu_shared2.py`).
- Trainable state: only PEFT LoRA tensors require gradients, and the optimizer is built over `self.params` alone. Each update starts by copying the member's tensors in (`_activate`), loads that member's AdamW state, and ends by copying both out. Resume reproduces the next update bit-exactly (`test_resume_reproduces_training_state`).

### 3. Policy consistency

Holds. Rollout and probe requests use `Member.version`, which is set only by `Sampler.publish`. Each publish uses a fresh name and a fresh directory, and the previous version is unloaded after the new one is loaded (`sampler.py:127-136`). Within a round the order is: rollouts on version v → consistency check on the pre-update weights → update → publish v+1 → probe on v+1 (`member.py:227-264`, `worker.py:195-215`). A resumed or fresh member is published before any request (`worker.py:199-201`). The base model is addressed only in `prepare` and for the `base` evaluation label. The online check cannot tell consecutive versions apart; this is documented.

### 4. Seeding

Holds.

- Student prompts are plain. In both runs every seeding record's `prompt_ids` equals the plain prompt of its uid, and no decoded prompt contains the scaffold text.
- Only correct, untruncated samples are kept: reward 1 implies `stop`, and every record re-scores to reward 1.
- Warm-up sampling follows the spec: a prompt is drawn uniformly among uids with a correct sample, then one of its correct samples, using an RNG seeded by (lineage seed, step) (`member.py:197-202`).
- Perturbation arm: A is Kaiming-uniform and B ~ N(0, σ), both from the lineage seed (`trainer.py:144-157`). It differs between seeds, and replacement lineages get new seeds.

### 5. Isolation and determinism of data

Violated (see D2 and R4). What holds:

- The four suites are pairwise disjoint, and every near-duplicate unit sits in one suite.
- Workers read only train and selection (`tasks.Suites`). External vault parts need a receipt.
- Training-stream questions (20 steps × 64) and seeding-pool questions (1536) are identical across processes.
- Training and seeding items are redrawn against train-test probe questions (`tasks.py:215-221`).
- External item ids map to exactly one part each (`external.part_of`), and GSM8K train and test share no question text.

What fails: some generators iterate Python sets, so instances depend on `PYTHONHASHSEED`. Separately, MMLU-Pro contains distinct ids with identical question text across parts (R4).

### 6. Culling, resume, records

Holds.

- `cull_reason` (`member.py:316-331`) implements the documented rules in order, including the requirement that the previous probe be at step ≥ grace.
- Resume rolls back to `state.json`: it deletes history at amplification step ≥ S, probes and probe files after S, rollouts at step ≥ S, and `.tmp` and later checkpoints. It then reloads the fp32 adapter and the optimizer of checkpoint S; resume tests pass.
- History, rollout, probe, seeding and state records in `runs/toy` and `runs/scale-test` agree with the schema. Rewards, weights, `groups`, `sequences`, `tokens` and the history reward means were recomputed from the records.
- The toy evaluation hashes equal the SHA-256 of the final checkpoints.

Caveat (R5): `scale-test` currently holds three uncommitted amplification steps per member. These are rollout files and history lines past `state.step = 0`, and they disappear only when the run is resumed.

### 7. Final evaluation and blind guard

Holds.

- The guard runs before any task of a vault is built. It requires the manifest to be tracked and equal to `HEAD` (`git diff --quiet HEAD`), and every trained adapter's `adapter_model.safetensors` hash to be in `ranking` (`evaluate.py:47-81`).
- The CLI restricts `--suite` to known names. Only the literal `base` maps to no adapter.
- The hashed file is the one served from the same directory (`evaluate.py:132-154`).
- `macro` is the mean over non-external families (`evaluate.py:108-123`).
- Shards `[i::n]` over one job list partition it; overlapping invocations are caught by the duplicate-key check.

Fragilities are listed as R2 and R3.

## Findings

### D1 (defect): the `coin_flip` scorer rewards wrong answers, including the constant `0`

reasoning-gym's `CoinFlipDataset.score_answer` gives "partial" credit equal to the matching prefix length divided by the length of the shorter string. Any answer whose formatted value is a prefix of the gold value therefore scores 1.0. Reproduction: `python runs/audit/coin_flip.py` prints:

```
answer '0' rewarded on 189.0 / 200; one-decimal rounding rewarded on 101.0 of the non-exact ones
```

Example: the gold is `0.205078125`, and both `0.2` and `0` earn reward 1. The engine's own junk check (`zzz`, empty) and `make_suites.py` missed it. Impact: `coin_flip` is one of 23 selection families. Selection macro is the harvest key and the selection probe statistic. A member that rounds or gives short numeric answers gains up to about 3 macro points (1/23 × (1 − 0.27 base exact rate)) on this family alone. That can reorder a harvest ranking whose spread is of the same order, and it confounds H2 and H3 with answer-format habits. The base calibration of this family (exact rate 0.27) is inflated too. Smallest fix: exclude `coin_flip` as a defective scorer. Alternatively, wrap its scorer so that reward requires `|float(Fraction(a)) − gold| ≤ 1e-4`. Either way, extend the `make_suites.py` junk check with the answers `0` and `1`.

### D2 (defect): instances of `polynomial_multiplication` (selection), `isomorphic_strings` and `ransom_note` (blind 1) depend on `PYTHONHASHSEED`

These generators iterate Python sets of strings, so the same uid yields a different question in each process. Python randomises the hash seed per process by default.

Reproduction: `for s in 3 4; do PYTHONHASHSEED=$s python runs/audit/det5.py; done` shows that only `polynomial_multiplication` differs among train and selection families. `runs/audit/det_blind.py` shows the two blind families (data only). `knight_swap` changes only its gold answer string; a cross-process gold still scores 50/50.

Real artefacts are affected: `python runs/audit/records_check.py runs/scale-test` reports the base model's and the members' `polynomial_multiplication` probe questions differing for 9 of 9 uids. Training prompts are unaffected.

Impact: probe instances of this selection family differ between the base model, each worker, and each restart. Selection evaluation instances differ between `evaluate` invocations and between shards, so members ranked by the harvest are not compared on the same items for this family. On blind 1, the outlier and the baselines would be scored on different instances of two families unless they share one unsharded invocation, which breaks the paired bootstrap. Uids are also not regenerable as documented.

Smallest fix: export `PYTHONHASHSEED=0` in `scripts/env.sh`, and fail fast at import of `evotaste.tasks` when `os.environ.get("PYTHONHASHSEED") != "0"`. Existing `toy`/`scale-test` probe comparisons for this family are invalid; re-prepare runs created before the fix.

### R1 (risk): a scorer can hang a worker beyond the 10 s alarm

`SIGALRM` only interrupts Python bytecode. The answer `1e999999999` keeps `simple_integration` busy for more than 400 s (killed at 400 s), and `intermediate_integration` and `polynomial_multiplication` more than 40 s (`runs/audit/bomb1.py`). All three are selection families, scored in every probe and in evaluation. No train family hung, and no false reward results, but a drifted member that emits such a number stalls its worker with no error. Trigger: any selection probe answer containing a huge exponent. Fix if it matters: score in a child process with a hard kill, or reject answers containing exponents above a few hundred digits before calling sympy-based scorers.

### R2 (risk): the evaluation summary assumes one adapter per label

`summarize` takes `adapter_sha256` from the first record of a label (`evaluate.py:117`). A synthetic check returns the first of two different hashes. If shards of one tag run at different times, `m00` can resolve to different checkpoints (for example before and after the lineage finishes). Their records then mix under one label, and `harvest` can still pass its hash check. Low-cost fix: raise when a label has more than one hash. The same function should also flag selection families missing from `n_by_family`, since `harvest` only checks families that are present. Blind summaries have no completeness check at all; require the expected count per family before a blind summary is reported.

### R3 (risk): the guard hashes only the weights file

The guard does not hash `adapter_config.json`, which vLLM also reads (rank, alpha, target modules). Reaching it needs deliberate edits inside `runs/`. Hashing both files would close the gap at no cost.

### R4 (risk): MMLU-Pro has distinct `question_id`s with identical question text across parts

Normalised exact matches: foundation–blind1 80, foundation–blind2 88, blind1–blind2 86, out of about 4000 per part (`runs/audit/external_split.py`, counts only). Foundation items are never trained on, so this cannot leak training signal. It does give about 2% overlap between the two vaults, which matters when vault 2 is used to verify designs informed by vault 1. Fix: split by a hash of the normalised question text instead of the id.

### R5 (risk): records past the resume point stay on disk until the next restart

Analysis that reads `rollouts/` or `history.jsonl` of a stopped run must filter by `state.json`'s `step`, as `scale-test` shows. Document this beside the schema, or make analysis code filter.

### R6 (risk): the harvest ranks every lineage with status `done`

This includes long and wide baseline members if they are configured in the same run, so a baseline could become "the outlier". Run baselines as a separate run, or exclude members with overrides from the ranking.

### S1 (suggestion): exercise the shared-prefix mask in the online check

The online consistency check re-scores one rollout per prompt, so it never exercises the shared-prefix mask in production. Re-scoring two rollouts of the first prompt would cover that path at negligible cost. The fp32 GPU equivalence above shows the path is correct today.

### S2 (suggestion): align the threshold comment

The comment above the consistency thresholds (`member.py:31-33`) quotes older measurements (0.013 / 0.003 and 0.11 / 0.065) than ENGINE_NOTES (0.0057 / 0.0022 and 0.12 / 0.073). Align it with ENGINE_NOTES.
