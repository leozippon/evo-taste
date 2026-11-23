# EvoTaste experimental protocol

This document fixes what the experiments are meant to decide and how, before results exist. Hyperparameters marked *tentative* may be changed by pilots up to the moment the main run starts; every change is recorded in the logbook. Nothing in the harvest rule or the blind comparisons may change after the first main-run adapter is trained.

## Question

Standard RL from verifiable rewards treats early lock-in and finite-rollout mode sharpening as defects of a single training run. EvoTaste asks whether the same dynamics, run independently in many copies of one model, act as a bias amplifier that occasionally produces a model with a sharp, coherent and transferable reasoning style — a model worth harvesting even if most of the population is mediocre.

"Taste" is used here in an operational sense: a model-specific way of reasoning that is (a) sharp — the model's own outputs agree with each other in style, (b) coherent — the same style shows on task families of very different content, and (c) consequential — it changes which unseen problems the model solves.

## Hypotheses and what would refute them

| | Claim | Refuted if |
|---|---|---|
| H1 Divergence | Independent members become more distinguishable from each other over training while each becomes more self-consistent; induced initial biases diverge further than seed noise alone. | Member identity cannot be recovered from responses on unseen families better than for the seed-only arm, or distinguishability decays during amplification. |
| H2 Heavy tail | The spread of transfer performance across members exceeds evaluation noise and is larger for biased arms than for the seed-only arm. | Between-member variance on the selection suite is within evaluation noise, or no larger than in the seed-only arm. |
| H3 Transfer | The member harvested on the selection suite beats, on the blind suite, the base model, the population median, and single-model RL given the population's whole compute. | The harvested member's blind advantage is within noise of any of these, or selection and blind scores are uncorrelated across members (winner's curse). |
| H4 Taste mediates transfer | Members that share an induced bias end with similar styles and similar transfer; style explains blind performance beyond in-distribution fitness. | Lineages with the same induced bias are no more alike than unrelated lineages, or style carries no information about blind performance. |

A clean refutation is a result. The protocol is built so that each failure mode has a diagnostic that names it: biases washed out (H1), divergence without consequence (H2), selection on noise (H3), or cosmetic style (H4).

## Units

A *member* is one LoRA adapter on a frozen base model, trained only on its own rollouts. Members never exchange rollouts, gradients or weights. A *lineage* is a member together with its induced bias and seed. The *population* is the set of members that share one base model and one environment.

## Task suites

All procedural tasks come from generators with exact verifiers, so instances are unlimited and the unit of distribution shift is the task family. Families are excluded only for practical reasons — an unreliable verifier, an overlong prompt, a task the base model already solves almost always — and are then assigned to suites by a seeded draw, so that no suite is hand-picked. Families that are variants of one another are assigned together. Three whole categories are held out, one for selection and one for each blind vault, so that each of those suites also tests transfer to a kind of task never trained on; the remaining families are dealt across all four suites, stratified by category and by whether the base model can solve them at all. Families the base model cannot solve stay in every suite, including training: unlocking one is the kind of rare, self-reinforcing event the project looks for. The assignment lives in one configuration file, which is the single source for every script.

| Suite | Content | Used for |
|---|---|---|
| Train | A broad mix of families at their default difficulty. Every member sees the same full mix and the same prompt stream. | Rollouts and updates. |
| Train-test | Fresh instances of the train families. | In-distribution fitness, the collapse floor, forgetting. |
| Selection | Families never trained on, plus a natural-language foundation probe. | The collapse floor, harvest ranking, all development decisions. |
| Blind, vault 1 and vault 2 | Families disjoint from train and selection, plus external benchmarks never touched elsewhere. Each vault holds different families and different benchmark items. | Verification only. |

## Life of a member

**Seeding (symmetry breaking).** Every member starts with a short reward-filtered warm-up in which it imitates correct solutions produced by the base model. The arms differ only in how those solutions were biased:

- *Seed arm*: the base model is prompted plainly; members differ only by sampling seed. This arm is the null for everything else.
- *Scaffold arm*: the base model is given a strategy scaffold — a short description of one reasoning habit — while the member is trained to reproduce the resulting correct solutions without the scaffold. Each scaffold is given to two members with different seeds, which separates the effect of the bias from seed noise.
- *Perturbation arm*: the adapter starts from a random low-rank perturbation and then receives the same plain warm-up as the seed arm.

**Amplification.** After seeding, no member ever sees a scaffold again. Each step samples a batch of prompts from the full train mix, draws a small number of rollouts per prompt, and applies one on-policy policy-gradient update with a group-relative advantage. There is no KL penalty to the base model and no trust region beyond the step size: a member is compared only with itself. Few rollouts per prompt are a deliberate choice, since finite-rollout skew is the amplifier under study. The return of a rollout is 1 for a correct answer, 0 for a wrong one and −0.5 for an answer cut off at the token cap. The last term was added after the first attempt at the main run: without it, a prompt on which every attempt fails gives no signal, and answers lengthened until about a third hit the cap, mostly on families a member never solves. Accuracy everywhere is still the plain share of correct answers. Settings of the main run: 128 prompts × 4 rollouts per step, 120 steps, LoRA rank 32, at most 1024 new tokens.

**Targeted protection.** Each member tracks its accuracy per train family and on the foundation probe against its own earlier checkpoints. Protection is applied only where a regression is detected, by replaying the member's own earlier correct solutions for the regressed family. Whether this is needed at all is decided by a pilot; the unprotected and KL-leashed variants are kept as ablations.

**Tolerant culling.** A member is removed only for collapse: in-distribution fitness below the untrained base model after a grace period, mostly unparseable or truncated output, or the foundation probe falling below half the base model's score. A removed member is replaced by a new lineage of the same arm. Poor but alive members are kept.

## Baselines

- The untrained base model.
- Single-model RL with the population's compute, in two forms: *long* (same batch, as many steps as all members together; its intermediate checkpoints give the comparison at every smaller budget) and *wide* (same number of steps, the batch enlarged by the population size).
- The population itself: the median member, the mean, and the best member of the seed arm chosen by the same harvest rule.

## Harvest rule

At the end of the main run, members are ranked by macro-averaged accuracy over the selection families, using the adapters of one common step only. The top member is "the outlier"; the top three are reported alongside.

The common step is fixed by a rule set before any final evaluation, because the machine is shared and the run may not finish: training stops when every member has completed its 120 steps or at 14:00 local time on 23 November 2026, whichever comes first, and the harvest step S is the last probe step (a multiple of 15) that every population member has reached by then. The wide baseline is evaluated at S and the long baseline at S, 2S, 4S and 6S, or at its last probe step if it has not got that far. Members that are behind are not waited for and members that are ahead are not given credit for later steps. The ranking, the adapter hashes and the code revision are written to a harvest manifest and committed before any blind evaluation.

## Blind protocol

The blind evaluator refuses to run without a committed harvest manifest and appends every invocation to a ledger in the repository. A vault is opened once, for the comparisons below, and is then spent: results from it may inform later designs, and those designs are verified only on the unopened vault. Base-model calibration on blind families is permitted beforehand because it involves no trained model.

Primary comparisons on the blind suite, each with a paired bootstrap interval over problems:

1. Outlier against base model, long and wide single-model RL, and the median member.
2. Rank correlation between selection and blind scores across members, and the retained fraction of the selection differential, (blind outlier − blind mean) / (selection outlier − selection mean).
3. Blind variance between and within scaffolds, and best-of-arm for each arm.

## Diagnostics of Taste

- *Sharpness and divergence.* Style features of responses to a fixed probe set, tracked over training: within-member variance, between-member variance, and their ratio.
- *Coherence.* A classifier trained to recognise members from responses on train families is tested on responses to selection and blind families; accuracy relative to chance and to the seed arm measures a style that survives a change of content.
- *Persistence of the induced bias.* Whether the scaffold a member was seeded with can still be recovered at the end of amplification.
- *Niches.* Per-problem solve sets: population coverage, unique solves per member, and blind problems the outlier solves repeatedly that no baseline solves in any sample, set against the reverse count.
- *Compression.* Tokens per correct solution and accuracy under a shortened token budget.
- *Confounds.* Answer-format validity is reported separately everywhere, so that learning the output format is not mistaken for transfer.
