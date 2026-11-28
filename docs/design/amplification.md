# Amplifying endogenous bias

The second direction of Round 2. The first, in [round-2.md](round-2.md), put a bias into a member from outside, by distilling an instruction, and asked whether training would keep it. This one asks the question the project started from: which training paradigm makes a member's own small deviations grow, so that it stops behaving like the average of what the base model has memorised and becomes sharply itself.

## What the pilots already say

- **Relative-only reinforcement learning does not amplify.** It removes within about 30 steps whatever costs reward and leaves everything else where it was. A harmless seeded disposition is still there after 45 steps, at about its seeded strength, not stronger.
- **Self-imitation is not the amplifier it was meant to be.** Imitating one's own successes halved the seeded dispositions and amplified one trait, answer length, up to the token limit.
- **Members do diverge by themselves, and substantively.** After 45 steps, which problems a member solves differs between members by about 14 points per problem beyond what level and difficulty explain, and a member's standing carries over to unseen sources at about half strength. The largest single difference is how long members' answers have become.
- **A member that starts from the bare base model is fragile under relative-only training.** It lengthens, loses held-out answer formats and loses mastered problems within 15 steps, while members whose adapter had first been through any distillation, even of a neutral instruction, do not.

So lineages drift apart, but nothing tried so far reinforces a lineage's own direction.

## What is to be amplified

A member's endogenous bias is the way its behaviour differs from the base model's on the same input, whatever produced the difference: its initialisation, the accidents of its own sampling, its own history of successes. Average behaviour is the base model's. A paradigm amplifies endogenous bias if, over training, a member's deviation from the base model grows along the direction it already has, stays functional, and differs from lineage to lineage. Growth that is the same for every lineage is a property of the environment, not a taste; growth that destroys accuracy is collapse.

## Undirected amplification fails

The first mechanisms pushed a member further along everything it had become, and the first checks were enough to judge them.

- **Exaggerated self-imitation.** On its own successful answers a member is trained toward a caricature of itself: its own next-token distribution pushed further away from the base model's. The push grows with the deviation it produces, so it has no resting point: at the planned strength a member collapsed within five updates (answers three times as long, accuracy down by a third), and a weaker push only postpones the take-off.
- **Scaling the adapter.** Multiplying an existing member's deviation in weight space by 1.5 or 2, with no training, made it less accurate, less compliant and far more verbose. Training had already taken the member about as far along its own direction as pays.
- **Self-chosen practice,** in which a member practises more of the kind of problem it solves best, mostly followed which sources are easy for every member.

The lesson is that a member's deviation is a mixture. Part of it is what the reward shaped, part is habit that the reward is indifferent to, and the largest part by size is length. Amplifying the whole mixture amplifies length first.

## Selective amplification

The next mechanisms act only where the reward is indifferent, and with a strength that does not grow with the deviation.

1. **Taste-shaped reward.** Among a member's correct answers, the ones most characteristic of itself earn more. An answer's characteristic score is how much more probable the member finds it than the base model does, per token; scores are standardised across the step's correct answers and clipped, and a correct answer's reward is scaled by between 0.5 and 1.5 accordingly. A wrong answer earns nothing however characteristic it is, so deviation cannot be bought with accuracy, and because the scale is relative within a step the pressure stays the same size as the member moves away. Where two answers are both right, the member is steered toward its own way of being right: this turns the drift that the reward ignores into a self-reinforcing direction. Run with the sign reversed, the same term steers a member back toward the base model, which gives a control that differs in nothing else.
2. **Consolidating one's own frontier.** Each member keeps a record of the training problems it has attempted. Part of every batch is then drawn from its own frontier: problems it has solved at least once but not reliably, the ones the base model rarely solves first. It attempts them afresh and learns from the outcome as usual; optionally it also imitates the answer with which it first solved them. What is reinforced is the member's own record of discoveries, which differs from member to member by the accidents of its sampling, and it is substance by construction: which problems a member can crack. The control revisits problems the member has seen before, chosen without regard to its own successes and matched in difficulty, so that the only difference is whose discoveries are consolidated.
3. **Practice relative to the base model.** A member practises more of the sources on which it has gained most over the base model's own pass rate, so that what is reinforced is the member's own progress, not the common difficulty of a source.

These are applied to members that training has already shaped, by cloning members at step 45 into the variants of a mechanism, so that the variants start from identical weights.

The first check of the taste-shaped reward showed what its score favours among correct answers to one problem: the shorter answer, because a member's deviation is diluted over a long one, and a particular phrasing, with the method unchanged. It is therefore expected to amplify presentation and brevity; the pilot measures whether it does anything more. Consolidating one's own frontier does not have this weakness, because it never looks at how an answer is written.

## Measurements

- **Deviation.** How much more likely a member finds its own answers than the base model does, in nats per token: its distance from average behaviour.
- **Distinctness.** How much better a member predicts its own answers than another member's, and how often the author of an answer is identified by likelihood alone. This needs no classifier and no features.
- **Coherence.** Whether a member is recognisable by the same measure on domains and sources other than those the comparison was fitted on.
- **Substance.** Which problems a member solves, relative to its overall level and each problem's difficulty; its profile across domains; accuracy on unseen sources; and health: answer formats, mastered problems, length. Because presentation is the cheapest way to be characteristic, the share of a member's recognisability that survives stripping layout and notation is reported with every result.
- **Heritability.** Clone pairs: do two continuations of one member stay closer than two unrelated members, in all of the above.

A mechanism is worth a population run if it raises deviation and distinctness above the neutral variant, the gain is not presentation or length, the difference is heritable, and accuracy on unseen sources does not fall.

## Pilots

- **Pilot D**: eight unseeded lineages on two GPUs — relative-only, weak exaggerated self-imitation, and self-chosen practice — to characterise the undirected mechanisms at low strength and to add unseeded baselines.
- **Pilot E**: two members of pilot C at step 45, each cloned into the taste-shaped reward toward its own direction, the neutral rule, and the reversed sign, for 45 further steps; the reversed sign was then repeated on seven more lineages.
- **Pilot F**: six members at step 45 cloned into consolidation of their own frontier, the matched control, and plain training.

Everything ran on at most four GPUs. The results are in [findings/round-2-pilots.md](../findings/round-2-pilots.md); none of the mechanisms met the criterion above on the 1.7B model, and the question is carried to a larger one in [scale.md](scale.md).
