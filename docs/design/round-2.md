# Round 2 design

Round 1 ([findings](../findings/main-run-1.md)) showed that independent members do diverge, but only where the reward is indifferent: layout and notation. Everything the reward had a say in was decided the same way for all of them, the biases they were seeded with were gone within tens of steps, and the one drifting trait that moved scores, answer format, paid off only on a selection suite that was strict about it. Round 2 keeps the paradigm — independent members, no communication or merging, bias amplification, tolerant culling, harvest of the outlier, a sealed blind test — and rebuilds the three things that result points at: the environment, the reward, and the way a bias is put in and reinforced. It proceeds by small pilots; a population run and a blind opening follow only if the pilots show divergence that is substantive and heritable.

## What the pilots must show

1. **The environment is healthy.** Members improve on tasks they did not train on, without losing answer-format compliance, without forgetting what the base model already does reliably, and without answers growing to the token limit.
2. **Divergence is not cosmetic.** Members come to solve different problems, not just to write differently: after allowing for each member's overall level and each problem's difficulty, which problems a member solves depends on the member by more than sampling noise.
3. **Divergence is heritable.** Two copies of one member, trained further with different random draws, stay more alike than two unrelated members, in accuracy, in which problems they solve and in how they work. Without this, harvesting selects luck, which is what happened in Round 1.
4. **A seeded bias can survive.** Some kind of seeded disposition is still measurably present after training, and it matters which kind: the pilots compare epistemic stances, mechanical tactics and a neutral control.

If the second and third fail under every setting tried, the conclusion is that a shared reward leaves nothing substantive for members to differ in, and a population run would repeat Round 1.

## Environment

Training moves from short synthetic puzzles to real datasets with automatic verification, in three domains.

| Domain | Trained on | Held out as unseen sources |
|---|---|---|
| Math | MATH (train), DeepScaleR, GSM8K (train), AQuA-RAT | TheoremQA, SAT-Math, SVAMP, Omni-MATH problems not in DeepScaleR |
| Code | KodCode (functions with unit tests) | HumanEval+, MBPP+, LeetCode |
| Logic and science | LogiQA, ProofWriter, ARC-Challenge, a third of MMLU-Pro | FOLIO, LSAT logical and analytical reasoning, FLD, twelve BigBench-Hard tasks, MedMCQA, ScienceQA, OpenBookQA, QASC |

**Difficulty.** In Round 1 only one prompt in six produced a learning signal, because the model either always failed or always succeeded on the rest. Before training, the base model attempts every training item eight times. Items it solves one to six times form the sweet spot and make up 70% of training prompts. Items it never solves (the frontier) and items it nearly always solves (mastered) each make up 15%, so that both new ground and retention stay in view and can be measured separately.

**Verification by meaning.** A math answer is correct if it is mathematically equal to the reference, whatever its notation. A multiple-choice answer is correct if it identifies the right option. Code is correct if it passes the item's tests in a sandbox. The same verifiers score training, selection and the blind test, so a member cannot win on notation.

**Two ways to answer a math problem.** A problem with a numeric or symbolic answer may be answered directly, or by a short program whose printed output is the answer; the program is run and its output verified like any answer. This gives the environment one clear-cut strategic choice — reason it out or compute it — with similar reward on many problems and different strengths elsewhere. Which way a member leans is a substantive trait that can be counted.

**Format guardrail.** By default the final answer goes in a box (or a code block). In 15% of training prompts the problem instead asks for a different, randomly chosen answer wrapper, and the reward requires it. Following a stated output instruction therefore stays rewarded throughout training. Compliance is measured on wrappers never used in training.

**Length.** The token limit rises from 1,024 to 4,096. The flat penalty for unfinished answers is replaced by a gentle slope: nothing up to 2,048 tokens, then a deduction growing linearly to 0.25 at the limit. Checking and second attempts cost nothing; only running on toward the limit does.

## Suites and the blind line

Every item belongs to exactly one of three parts.

- **Training.** The training sources, minus their held-out items.
- **Selection.** Used for monitoring, for all pilot measurements and, later, for ranking the population. It holds held-out items of the training sources (near transfer) and about half of the unseen sources (far transfer). It also holds the spent halves of Round 1's first vault.
- **Blind.** The other half of the unseen sources and the remaining held-out items of the training sources. Round 1's second vault is not replaced but extended with these: its procedural families, its half of GSM8K test and MATH-500 and its third of MMLU-Pro stay sealed inside it. No model, not even the base model, is run on any blind item until a harvest manifest is committed. The vault is opened once.

Which unseen source goes to selection and which to blind is a seeded draw within each domain. Several public math sets contain copies or rewrites of the sealed questions (all of MATH-500 is in the MATH test split; GSM-Plus and GSM-Symbolic are rewrites of GSM8K test). Rewrites are excluded outright, and question texts are de-duplicated across all sources before the split, with blind taking precedence over selection and selection over training.

## Seeding

A member's starting bias is distilled from a context: a short instruction shown to a frozen teacher and never to the member. The member writes its own answers without the context and is pulled, token by token, toward what the teacher would write with it (on-policy context distillation). The teacher is the base model itself, so a member gains a disposition and no extra ability; a larger teacher from the same family is the fallback if the base model does not act on a context.

Contexts come at two levels on five shared themes.

| Theme | Stance (a disposition) | Tactic (a fixed procedure) |
|---|---|---|
| Falsification | try to break every answer before accepting it | plug the answer back into each condition |
| Economy | prefer the account that needs the least | fewest steps, fewest words |
| Foundations | rebuild from definitions and given facts, not remembered templates | number every definition and fact, cite them in each step |
| Evidence | work concrete cases first, let rules follow | solve a smaller version first |
| Structure | attend to how the parts relate before the parts | tabulate objects and their links, solve from the table |

A stance says what to do and how it plays out in a calculation, a program and an argument; a tactic is one mechanical routine carried out the same way everywhere. No context names a school of thought, and each forbids naming or describing the method; a term list checks that members never recite. A neutral context and an unseeded member are the controls, and the one Round 1 habit that survived training is kept as a reference.

How much of a context a member carries is measured the same way for every context: how much better the teacher with the context predicts the member's uninstructed answers than the teacher without it, scaled so that the base model scores 0 and the teacher's own instructed answers score 1. Tracked over training this is the wash-out curve; measured on code and logic after seeding on math alone it is cross-domain coherence.

## Amplification

Members learn from their own attempts as before: several attempts per problem, each reinforced by how much better it did than the others on that problem. Two variants are compared.

- **Relative only.** As in Round 1, except that the advantage is no longer divided by the spread of rewards within a problem. That division would turn the gentle length slope into a full-size signal whenever all attempts are otherwise equal.
- **Relative plus self-imitation.** In addition, a member imitates its own successful attempts, whatever they looked like. This is the rich-get-richer term: whatever a member happened to do when it succeeded becomes more likely, so accidents of sampling compound instead of averaging out. Theory on the role of baselines in policy gradients predicts exactly this: the less of a success is explained away by the baseline, the more committal the learner.

## Pilots

**A. Census (no training).** The base model attempts the training pool eight times and the selection suite four times under the new prompts and verifiers. This yields the difficulty strata, the length distribution that the slope is checked against, an audit of the verifiers for false credit, and a first look at strategy: on which math problems each way of answering works, and how the base model behaves under each seeding context.

**B. Seeding (no reinforcement learning).** Each context is distilled once on prompts from all domains and once on math prompts only; two contexts are also seeded by Round 1's method, imitation of the teacher's correct answers, for comparison. Measured straight after seeding: how much of the context was absorbed, whether it shows up outside the domain it was seeded in, accuracy, and recitation.

**C. Amplification and twins.** Fourteen seeds (five stances, six tactics, two controls, one unseeded) are each trained under both update rules for 60 steps of 64 problems × 4 attempts. A subset is then cloned: each clone pair continues from the same weights for another 60 steps with different random draws. The four requirements above are read off this pilot: health from the selection suite, format wrappers and strata; wash-out from the context measure; substance from the member-by-problem analysis and from counted traits such as the share of math answered by program; heritability from clone pairs against unrelated pairs.

Culling is switched off in the pilots so that failures can be observed. The population design, the harvest rule and the stopping rule are written after pilot C, in a protocol committed before the population run starts.
