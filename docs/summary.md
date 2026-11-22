# EvoTaste

EvoTaste tests whether the dynamics that make single-model reinforcement learning collapse — early lock-in and the self-reinforcement of whatever a model happened to sample — can be used across a population to grow models with a sharp, coherent and transferable reasoning style, here called Taste. Compute efficiency is not a goal: the population is allowed to be mostly mediocre if a few members turn out exceptional.

## How it works

Many copies of one small language model, each a low-rank adapter on a frozen Qwen3-1.7B, are trained independently in the same mixed environment of procedurally generated, automatically verified reasoning tasks. They never share data or weights.

1. **Seeding.** Each member is nudged toward a different starting bias. Some imitate solutions the base model wrote while following a reasoning habit (check every condition, work backwards, enumerate cases, and so on); the habit is described only to the base model, so the member absorbs it into its weights. Others differ only by random seed or by a random perturbation, and serve as controls.
2. **Amplification.** From then on every member learns only from its own attempts: a few attempts per problem, reinforced relative to each other, with nothing pulling it back toward the base model. Small differences compound.
3. **Tolerant culling.** A member is replaced only if it collapses — worse than the untrained model, or producing mostly unusable output. Weak but functioning members stay.
4. **Harvest.** At the end, members are ranked on task families none of them trained on, and the best is harvested.

## How the result is kept honest

Picking the best of many runs invites a lucky winner. The task families are therefore split, by a seeded draw, into training, selection and two sealed blind vaults, each blind vault also holding external benchmarks. The harvest ranking is committed before a vault is opened, the blind evaluator refuses to run otherwise, and every opening is logged. A vault is used once.

The harvested member is compared, on blind tasks, with the untrained model, with the median member, and with single models trained using the population's compute. Beyond accuracy, the project measures whether members can be told apart by how they reason on unfamiliar tasks, whether that signature strengthens over training, whether it predicts which unseen problems a member solves, and whether solutions become more economical.

The full protocol, including what would count as a refutation, is in [design/protocol.md](design/protocol.md). Progress and decisions are recorded in the [logbook](logbook/).

## Status

The environment and serving stack are verified; the training engine is being built. No training results exist yet.
