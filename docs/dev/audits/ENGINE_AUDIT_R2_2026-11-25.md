# Engine audit, Round 2, 2026-11-25

Scope (frozen): the engine at `c226a52` and `0c35e58` (`evotaste/trainer.py`, `member.py`, `worker.py`, `sampler.py`, `config.py`, `records.py`, `evaluate.py` and their tests), against `docs/dev/ROUND2_SPEC.md` (WP-E and the contract) and `docs/design/round-2.md`. None of these files has changed between `0c35e58` and `HEAD` (`98366d3`). The task layer and the verifiers are out of scope. Library versions read: vLLM 0.34.0, PEFT 0.22.0, transformers 5.22.2, torch 2.14.0.

Method: I read the code, and ran the engine tests and my own throwaway checks on the CPU with the tiny Qwen3 of `tests/conftest.py`, through stdin scripts with no files left behind. I did not use a GPU. The engine tests (`test_trainer`, `test_engine_flow`, `test_resume`, `test_fork`, `test_config`, `test_consistency`, `test_sampler`, `test_evaluate`, `test_blind_guard`, `test_harvest_step`, `test_logging`) gave 112 passed in 230 s.

## Verdicts

| # | Property | Verdict |
|---|---|---|
| 1 | Return with the length slope; records, probes and culling on the reward | holds |
| 2 | Centred advantage; equal groups contribute nothing; `standardized` as in Round 1 | holds |
| 3 | Self-imitation term, its normalisation, β = 0, empty cases | holds |
| 4 | Distillation: reverse KL, positions, teacher, alignment, chunking, gradients, modes | holds (CPU; the GPU bf16 path was read, not run) |
| 5 | Clones: bit-identical start, step counter, same prompts, only the seed differs | holds |
| 6 | Sampling seeds and vLLM's `n` | holds for every derived seed; the config does not keep lineage seeds distinct (D1) |
| 7 | Sampler/trainer consistency check under the new prompt path | holds |
| 8 | `context-score` | holds |
| 9 | Blind guard, ledger and manifest unchanged since `round-1`; no new path to a blind item | holds |
| 10 | Config refusals, strata hash, `culling.enabled: false` | holds, except duplicate lineage seeds (D1) |
| 11 | Bit-exact resume, including self-imitation, clones and explicit seeds | holds |

### 1. Return

`length_returns` (`trainer.py:54-57`) is `r − slope·min(max((n − free)/(cap − free), 0), 1)`. It is called with `cap = amp.max_tokens` and `n = completion_tokens = len(sample.token_ids)`, end-of-turn included (`member.py:157`, `402-403`). A completion cut off at the cap has `n = max_tokens`, so the clip is 1, and the task layer's reward is 0 unless the completion stopped (`tasks.py:415`). Check: `length_returns([0],[4096],4096,2048,0.25) == [-0.25]`. The return goes only into `weight` and into the record's `return` field (`member.py:404-410`); history `reward`, `reward_by_family`, probe summaries (`member.py:135`) and culling (`member.py:506-521`) read `reward`.

### 2. Advantage

`advantages` (`trainer.py:60-74`) returns exact zeros when `max == min`, and otherwise `(r − mean)/0.5` (centred) or `(r − mean)/(std + 1e-6)` (standardized). Round 1's rule (`round-1:evotaste/trainer.py:47-53`) tested `std == 0.0`; the only difference is that equal values whose float mean is off by rounding now give exact zeros instead of weights near 1e-11, which is an improvement. The normaliser of the advantage term counts only sequences with `w ≠ 0` (`trainer.py:339-340`), so an all-equal group adds nothing to it. Checks: equal-return groups at 0.0, 1.0 and −0.25 (all truncated) gave `[0,0,0,0]`; lengths 100–2,048 with reward 1 gave `[0,0,0,0]`. The existing test `test_length_difference_alone_gives_small_proportional_weights` covers proportionality.

### 3. Self-imitation

`imitate = reward == 1.0` (`member.py:408`). `update` (`trainer.py:328-368`) builds `active` from sequences with `w ≠ 0` and, only if β > 0, those with `imitate`. Each term is divided by its own token count, there is one backward per micro-batch on `adv/adv_tokens + β·si/si_tokens`, and one `_step`. Checks I ran:
- **No mixed group, three successes, β = 0.5:** `si_loss` was 4.625201, the same as a hand-computed token-mean NLL (4.625201), with `adv_loss` 0.0 and the weights changed.
- **No success and no mixed group:** 0 sequences, 0 micro-batches, weights unchanged, optimizer still `None`.
- **Mixed groups but no success:** at β = 0.5 the weights were bit-identical to β = 0.
- The existing tests cover β = 0 equalling the advantage-only update and the sum of the two normalised terms.

### 4. Distillation seeding

- **Loss.** The loss is `Σ p_s (log p_s − log p_t)` over the full vocabulary, from KL_CHUNK-row chunks of hidden states, recomputed in backward (`trainer.py:152-165`). It is divided by the batch's total completion tokens (`trainer.py:380`, `393`).
- **Positions.** `build_rows` (`trainer.py:108`) puts the target `y_t` at source position `n−1` for t = 1 and at the previous completion position after that. So the positions are exactly those that predict each completion token, including the one that predicts the end-of-turn token; nothing after the end-of-turn token is included.
- **Teacher.** The teacher pass runs under `torch.no_grad()` and `disable_adapter()` (`trainer.py:387-389`) on `Sequence(teacher_prompts[i], seqs[i].completion)`. Its rows have their own positions (`range(n_teacher, …)`), and teacher and student outputs are matched by sequence index through `by_sequence` (`trainer.py:385-392`).
- **Gradients.** Only the LoRA tensors require gradients: `trainer.py:194-196` refuses anything that is not fp32, the bf16 base is frozen and `lm_head` is tied and frozen.
- **Modes.** The model is always in train mode (`trainer.py:191`). `lora_dropout = 0` (`trainer.py:189`) and Qwen3-1.7B's `attention_dropout` is 0.0 (its `config.json`), so neither pass is stochastic.
- **Rollouts.** The member samples behind `run.prompt(question)` with no context (`member.py:297-299`, `313`); the teacher reads `run.prompt(question, context)` (`member.py:319`).
- **Check.** Four prompts with teacher prompts 3–9 tokens longer, one prompt packed with a second completion, `token_budget = 30` (several micro-batches) and `KL_CHUNK = 3`. The engine's KL was 0.0172001 with gradient checkpointing off and on, against 0.0172000 from per-sequence full forwards with explicit slicing. With gradient checkpointing on and off, the weights after the step differed by at most 0.0, and the gradient norms were equal.
- **Not run.** The GPU bf16 autocast path was not run.

### 5. Clones

- **Start state.** `start_clone` (`member.py:365-372`) loads the parent's fp32 adapter and `optimizer.pt` from the clone-point checkpoint and sets `lin.step = p`. The parent keeps that checkpoint with its optimizer state (`member.py:231`, `475`, `538-539`), and writes it before committing `state.json`, which is what the clone polls (`member.py:354`).
- **Settings.** The clone's settings are the parent's effective settings plus its own amplification and culling overrides (`config.py:153`). The run has no learning-rate schedule (AdamW `lr` is reset from the config, `trainer.py:309-310`), and the trainer has no per-member randomness: no dropout, and `new_member` is never called for a clone.
- **Inputs.** Probe set: `Run.probe_tasks`, the same for all. Prompts: `train_batch(lin.step, prompts)`, a pure function of (run seed, step) (`tasks.py:723-738`).
- **Seeds.** Rollout seeds come from the clone's own seed (`member.py:391`).
- **Tests.** The existing tests check byte-identical adapter files, identical AdamW moments and step counters, and an identical next update (`test_clone_continues_the_parent_bit_exactly`, `test_clones_start_…`).

### 6. Sampling seeds

- **Derivation.** Seeds are `stable_seed(tag, lineage seed, step or update, index)` for rollouts, opcd, imitation and probes (`member.py:164`, `299`, `301`, `391`); `("probe","base"[,context], i)` for the references (`worker.py:175`); and `("eval", adapter sha256 or "base", uid)` for evaluation (`evaluate.py:200-201`).
- **vLLM's `n`.** vLLM 0.34 gives the k-th child of an n-sample request seed + k (`vllm/v1/engine/parallel_sampling.py:69-81`). Its Gumbel noise is `murmur3(seed, position, token)` (`vllm/v1/worker/gpu/sample/gumbel.py:172-212`). So the samples of one prompt are distinct draws. Two lineages share noise only if their seeds coincide.
- **Probes and evaluation.** Probe seeds include the lineage seed, so a clone and its parent probe the identical adapter at the clone point with independent noise. Evaluation seeds follow the adapter's content: byte-identical adapters (a parent at the clone point and the clone's own copy) get the same draws, and any two distinct adapters get independent ones.
- **Gap.** The config only ensures that a clone's seed differs from its parent's and from its same-point siblings (D1).

### 7. Consistency check

Both amplification and opcd re-score `Sequence(run.prompt(question), token_ids)`, which is exactly the prompt sent to vLLM, under the member's current weights. These are the weights of the published version, so the comparison is with vLLM's `token_logprobs` (`member.py:313-317`, `406-414`, `457-467`). `check_sample` assumes prompt-major order, which `amplify` produces (`member.py:46`, `400-410`); opcd passes `rollouts = 1`.

### 8. Context score

`context_deltas` (`evaluate.py:387-395`) scores the stored `token_ids` with `base_logprobs` (adapter disabled, `trainer.py:297-301`) behind `run.prompt(q, context)` and behind `run.prompt(q)`. Positions align through `build_rows`. The deltas are summed per response and token-weighted by domain (`score_rows`). The subjects are `base`, `base:<context>`, and committed probes `step <= lin.step` (`evaluate.py:371-384`). The existing test `test_context_score_of_a_plain_prompt_context_is_zero` covers alignment.

### 9. Blind guard

The body of `blind_guard` is byte-identical to `round-1` (both md5 `027d7e90…`). The diff against `round-1` removes the `train` and `blind1` suites, routes items through `fixed_tasks(..., receipt)`, adds evaluation seeds and adds `context-score`, which reads only probe answers on selection items. The only call that loads vault items, `fixed_tasks(…, VAULT, receipt)`, follows the guard (`evaluate.py:177-179`).

### 10. Config

- **Unknown keys and types.** `_build` refuses unknown and missing keys and wrong types at every level, overrides included (`config.py:203-215`, `168-173`).
- **Consistency checks.** Context versus method, clone rules, the bounds on the length slope and β, and the need for a strata table when any member trains are all checked (`config.py:218-262`, `265-284`).
- **Strata hash.** It is checked at prepare time (`worker.py:152-153`) and again whenever the stream loads (`tasks.py:598`, through `Run.streams`), which happens at worker start through `check_probe_set`.
- **Culling disabled.** With `enabled: false`, `culled` is false and `cull()` is never reached (`member.py:489`, `493`).
- **Gap.** Duplicate lineage seeds are accepted (D1).

### 11. Resume

End-to-end check: the `test_engine_flow` configuration (opcd, imitation and none seeding; β = 0.5; length slope; two clones at step 1; `checkpoint_every = 1`) was run once uninterrupted and once killed inside `amplify` of `a` at step 2 and of `a1` at step 2, each kill followed by a fresh `FakeRun`, `Policy` and sampler. The fake sampler draws from the request seed, so both runs see the same samples. Result: across all 15 checkpoints the adapter files, AdamW `step`, `exp_avg` and `exp_avg_sq` were identical, there were no extra or missing checkpoints, the rollout files were identical, and the history was identical apart from times and version names. A unit check of checkpoint → reload → update at β = 0.5 against an uninterrupted update was also bit-exact.

## Defects

### D1 (low impact, low cost): lineage seeds are not required to be distinct across the run

`_check_clone` (`config.py:281-284`) refuses only a clone whose seed equals its parent's or a sibling's at the same parent and step. Nothing refuses two non-clone members with the same seed, two clones of different parents or clone points with the same seed, or a clone that reuses an unrelated member's seed.

A shared seed means shared LoRA A initialisation (`trainer.py:207-221`), the same opcd, rollout and probe seeds, and the same replacement seeds (`member.py:562`). Because vLLM's noise is a function of (seed, position, token), members with shared seeds sample with common random numbers. Two `none` members with one seed are the same lineage, the same up to batch numerics. Two members seeded from different contexts but with one seed draw correlated rollouts at every step. Either way, "unrelated" pairs would look more alike than they are, which biases the heritability comparison (clones against unrelated members) in the direction that hides heritability. The rationale that the config gives for refusing a clone's seed ("or the two would draw the same samples") applies to every pair of members. The template and toy configs use distinct seeds, so no existing config is affected.

Reproduction: `config.parse({... "members": [{"id":"a","seed":7,"context":"c"}, {"id":"b","seed":7,"context":"d"}, {"id":"t1","seed":9,"parent":{"member":"a","step":0}}, {"id":"t2","seed":9,"parent":{"member":"b","step":0}}, {"id":"k1","seed":9,"parent":{"member":"a","step":10}, "overrides":{"amplification":{"steps":70}}}]}, {"c","d"})` is accepted.

Smallest fix: in `parse`, refuse any repeated `seed` among `cfg.members`, which makes the clone-specific seed check redundant, and add one case to `test_config.py`.

No other defect was found in scope.

## Suggestions

- **Length-only groups dilute the advantage normaliser.** Groups that differ only in length have unequal returns, so their tokens enter the advantage term's normaliser (by design). Under β > 0 this shifts the balance between the advantage and self-imitation terms with a member's answer lengths: members with many long all-correct groups put relatively more weight on self-imitation. The history logs `tokens`, `si_tokens` and `groups`, so analysis can control for this; it is worth a look when comparing the rules.
- **Self-imitation ignores the length deduction.** A 4,000-token success is imitated at full weight (as specified), so the self-imitation arm partly offsets the slope. Compare completion lengths by rule before attributing length differences to anything else.
- **Clipping applies to the summed gradient.** Clipping at 1.0 (`trainer.py:319`) scales the advantage component down whenever the self-imitation term pushes the norm over. Report how often `grad_norm > 1` in each arm.
- **The KL uses bf16 logits.** The KL is computed from bf16 logits (`trainer.py:153-154`, the same head precision as the log-probabilities). I did not measure whether this sets a noise floor that matters for small KLs; it is a candidate if opcd's KL stalls near 1e-3.

## Accepted limitations (as documented, relevant to the pilots)

- **A clone cannot be cloned** (`config.py:271`). In the template the self-imitation twin is the clone, so clone pairs exist only under the advantage-only rule. Clone pairs under self-imitation would need the self-imitation arm to be a non-clone (for example in a `fork` with an amplification override), whose rollout seeds then equal the source member's.
- **Rerun and clone samples match only approximately.** vLLM's batch-dependent bf16 numerics make samples of a rerun or of clones in different batches only approximately reproducible.
- **The consistency check cannot catch a swap at seeding update 0**, because the member is then the base model.
- **A clone's culling history starts with its own first probe.** This matters only with culling enabled.
