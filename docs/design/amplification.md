# Amplifying endogenous bias

The second direction of Round 2. The first, in [round-2.md](round-2.md), put a bias into a member from outside, by distilling an instruction, and asked whether training would keep it. This one asks the question the project started from: which training paradigm makes a member's own small deviations grow, so that it stops behaving like the average of what the base model has memorised and becomes sharply itself.

## What the pilots already say

- **Relative-only reinforcement learning does not amplify.** It removes within about 30 steps whatever costs reward and leaves everything else where it was. A harmless seeded disposition is still there after 45 steps, at about its seeded strength, not stronger.
- **Self-imitation is not the amplifier it was meant to be.** Imitating one's own successes halved the seeded dispositions and amplified one trait, answer length, up to the token limit.
- **A member that starts from the bare base model is fragile under relative-only training.** It lengthens, loses held-out answer formats and loses mastered problems within 15 steps, while members whose adapter had first been through any distillation, even of a neutral instruction, do not.

So nothing tried so far reinforces a lineage's own direction. The mechanisms below are built to do exactly that.

## What is to be amplified

A member's endogenous bias is the way its behaviour differs from the base model's on the same input, whatever produced the difference: its initialisation, the accidents of its own sampling, its own history of successes. Average behaviour is the base model's. A paradigm amplifies endogenous bias if, over training, a member's deviation from the base model grows along the direction it already has, stays functional, and differs from lineage to lineage. Growth that is the same for every lineage is a property of the environment, not a taste; growth that destroys accuracy is collapse.

## Mechanisms

Each adds one thing to relative-only training. All members start unseeded and differ only by seed.

1. **Exaggerated self-imitation.** On its own successful answers a member is trained toward a caricature of itself: its own next-token distribution pushed further away from the base model's (member plus a multiple of member minus base, in logit space). Standard practice penalises distance from a reference model; this rewards it, but only in the direction the member has already taken and only where the answer was right. A member identical to the base model feels no push, so the symmetry has to be broken by the member's own sampling first, and whatever breaks it is then reinforced.
2. **Self-chosen practice.** A member practises more of the kind of problem it currently solves best. All sources stay in the mix within bounds, so the environment remains whole, but the proportions follow the member's own record: early luck on one kind of problem becomes more practice on it, and more skill.
3. **Weight-space extrapolation.** The adapter is the member's deviation from the base model in weight space. Scaling it up exaggerates the member directly, with no training. It is tried first as a probe on the adapters that already exist: does a member at 1.5 or 2 times its own deviation become more itself, or just worse?

Because the previous amplifier fed on length, the imitation terms apply only to successful answers that stay within the free length.

## Measurements

- **Deviation.** How much more likely a member finds its own answers than the base model does, in nats per token: its distance from average behaviour.
- **Distinctness.** How much better a member predicts its own answers than another member's, and how often the author of an answer is identified by likelihood alone. This needs no classifier and no features.
- **Coherence.** Whether a member is recognisable by the same measure on domains and sources other than those the comparison was fitted on.
- **Substance.** Which problems a member solves, relative to its overall level and each problem's difficulty; its profile across domains; accuracy on unseen sources; and health: answer formats, mastered problems, length.
- **Heritability.** Clone pairs, as before: do two continuations of one member stay closer than two unrelated members, in all of the above.

A mechanism is worth a population run if it raises deviation and distinctness above relative-only training, the difference is heritable, and accuracy on unseen sources does not fall.

## Pilot D

Four arms of four unseeded lineages each — relative-only, exaggerated self-imitation, self-chosen practice, and both together — for 45 steps of 48 problems × 4 attempts, with the two unseeded self-imitation members of pilot C as a fifth reference. Clone pairs follow for the arm that deviates most. Everything runs on at most four GPUs.
