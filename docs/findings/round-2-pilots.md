# Round 2 pilots

Round 2 rebuilt the environment on real, verified datasets and then asked two questions in small pilots: does a bias put into a member from outside survive training, and can a training rule make a member's own deviation from the base model grow into something useful. This page gives the results up to 27 November 2026. All runs use Qwen3-1.7B with low-rank adapters; none touches the sealed blind vault.

The short answer: the rebuilt environment works, and members trained in it differ in real, partly transferable ways. But every rule that pushed a member further from the base model, in whatever form, cost accuracy in proportion to the distance gained, and the one rule that pulled members back toward the base model did best. Sharp, lineage-specific identity turned out to be easy to produce and to be a different thing from better reasoning.

## The rebuilt environment

Training draws on nine sources of math, code and logic problems, filtered by a census in which the base model attempted 27,612 problems eight times each. Answers are verified by meaning, a share of prompts asks for a varying answer format, and answers may run to 4,096 tokens with a gentle deduction beyond 2,048.

- **The learning signal is several times denser.** About half the problems of a step have attempts with different rewards, against one in six in Round 1.
- **Format compliance holds.** On real-data problems, strict and format-forgiving accuracy agree within 0.1 point for every trained member; Round 1 lost about ten points here.
- **Members gain on new ground.** After 45 steps: +3 to +11 points on problems the base model never solved, and for most members +4 to +7 on held-out problems of the training sources. On sources never trained on, members that started from a distilled adapter gain 2 points on average (from −2 to +6).
- **A member's standing carries over to unseen sources at about half strength.** Ranking 28 members on half the unseen sources and testing on the other half, the top member leads by 3.8 points where it was selected and by 2.0 where it was not; the rank correlation between halves is 0.54. In Round 1 the harvested member kept none of its lead. The blind vault is further away than a held-out half, so this is an upper bound.
- **Answers still lengthen.** Mean length grows from about 600 tokens to as much as 1,700 within 45 steps, and under some rules a tenth or more of answers are cut off. The gentle deduction does not hold it.
- **A member started from the bare base model is fragile**: under plain training it lengthens, loses answer formats it was not trained on and loses mastered problems within 15 steps, and it gains nothing on unseen sources. Members whose adapter first went through any distillation, even of a neutral instruction, do not show this.

## A bias put in from outside

Twelve instructions were distilled into members that never saw them: five epistemic stances (for example, try to break every answer before accepting it), five mechanical tactics on the same themes, a sixth tactic kept from Round 1, and a neutral control. How much of an instruction a member carries is measured on a scale where 0 is the base model and 1 the base model following the instruction.

- **Stances are small nudges that distil cleanly; tactics are large changes that distil partly or not at all.** Stances reach 0.25 to 0.34, tactics −0.07 to 0.21. In absolute terms two tactics were absorbed more than any stance; the stances simply ask for much less. Seeded by imitation instead of on-policy distillation, a tactic is absorbed as completely as a stance (0.34 against 0.10), so the gap is a property of the method.
- **What is absorbed appears in every domain.** Stances distilled on math alone show up on logic problems at two thirds or more of their full strength, and so do the tactics that were absorbed.
- **After 45 steps of plain training the stances are still there** (0.16 to 0.33), and in each of the five themes the stance outlasts its tactic. Five themes cannot give a p-value below 0.06, and they give 0.06.
- **Abstraction is probably not the reason.** The two tactics that vanish by step 30 are the two instructions that lower the base model's accuracy; the two tactics that do not lower it persist as well as the stances. The reward removes what costs it and leaves the rest.
- No member recites the vocabulary of its stance.

## What members become under plain training

- **They come to solve different problems.** After removing each member's level and each problem's difficulty, a member's solve rate on a problem still varies across members with a standard deviation of 14 points (interval 13.3 to 14.7; a model against itself reads zero). Answer length and truncation account for about 9% of this.
- **They become recognisable, and not by layout.** A classifier names the author of an answer to a problem from an unseen source 56% of the time among 28 members; 48% with layout and notation stripped.
- **They move steadily away from the base model**: by 0.13 nats per token at step 45 and 0.17 at step 90, measured as how much more probable a member finds its own answers than the base model does.
- **A lineage inherits its manner and its level, not its set of solved problems.** Two clones of one member, trained 45 steps further with different random draws, write alike (their answers are less than half as far apart by likelihood as those of unrelated members), and their answer length, coding habits and layout follow their parent's with slopes near one. But in which problems they solve, the two clones are as far apart as two unrelated members were after their own first 45 steps. The set of solved problems keeps drifting at the same pace whatever the ancestry.
- **Distance from the base model goes with less transfer.** Across the 28 members, deviation from the base model is rank-correlated −0.43 with the gain on unseen sources and −0.58 with accuracy on mastered problems, and not at all with the gain on frontier problems. Members learn the frontier equally well whether they move little or much; the extra movement buys only loss.

## Amplifying a member's own direction

Six rules were tried. Each adds one mechanism to plain training.

| Rule | What happened |
|---|---|
| Imitate one's own successes | Answer length grew to the token limit (5 to 14% of answers cut off) and seeded stances were halved. Members drifted further from the base model and from each other, with no clear change in accuracy. |
| Train toward an exaggerated version of oneself | Collapse: within 5 updates at the planned strength, and within 15 at one fiftieth of it, where over 90% of answers ended at the token limit. The push grows with the deviation it causes. |
| Scale the adapter by 1.5 or 2, without training | Accuracy fell from about 0.5 to 0.4 and 0.3 or lower, compliance fell and answers doubled in length. |
| Practise more of the sources one solves best | The proportions of different members moved together (87 to 94% in common): this follows which sources are easy, not who the member is. |
| Reward the most characteristic of one's correct answers | Deviation grew fivefold in 45 steps, each lineage in a direction of its own, and accuracy fell by 5 and 14 points in the two lineages tried. With the sign reversed, deviation shrank to a fifth and accuracy was 1 to 2 points above plain training. |
| Revisit the hard problems one has cracked oneself | The revisited problems were solved more often (about 49% against 32% before). On held-out problems the in-run probes show no gain over plain training, and none over revisiting the problems another member cracked; the two-sample evaluation is being run. |

The fifth rule is the cleanest test, because it never pays for a wrong answer and its strength does not grow as the member moves. A reader compared the resulting members with their plain siblings on the same problems. The method of solution was the same wherever both solved a problem. What had changed was manner: one lineage opens two answers in three with the same phrase and reopens its first step repeatedly, the other writes hedged, compressed prose without any markup, and their changes in vocabulary are uncorrelated. The cost came from the working: the weaker lineage often no longer wrote a derivation before asserting an answer, and both began to leak fragments foreign to the base model's habits.

## What this says about the thesis

- **Identity can be amplified; it is not taste in the sense the project is after.** A lineage's own direction away from the base model is real, inherited and easy to strengthen. At this model size it consists of manner and length, not of method, and strengthening it damages the reasoning it is wrapped around.
- **The base model's competence sits where its behaviour is typical.** Three interventions and the comparison across members agree: distance from the base model is a price paid for training, not a product of it. The best members are those that learned the frontier while moving least.
- **Useful differences between members exist but are not self-reinforcing.** Which problems a member solves differs substantially between members and half of a member's standing transfers to unseen sources, which is enough for selection to pay modestly. But that difference is accumulated drift: it is not kept by a member's continuations and no rule tried here made a member build on it.
- **The constructive result points the other way.** Pulling a member's correct answers toward the base model preserved what plain training began to lose after step 45. This rests on two lineages; seven more are being trained to test it.

## Caveats

- One base model of 1.7 billion parameters. A model with several ways of solving a problem could have something other than manner to amplify.
- Pilots, not population runs: the taste-shaped reward rests on two lineages and the frontier rule on six, each compared within lineage against siblings from identical weights; effects are large against item-level intervals but cannot reach conventional significance across so few lineages.
- Most accuracy figures for single members come from two samples on 1,840 problems (about ±2 points); in-run probes are single samples on about 360.
- The verifiers credit an unevaluated arithmetic expression as its value. It occurs in about 1% of credited math answers, nearly all of them answers that are legitimately expressions.
- The harvest simulation stays inside the selection suite. Nothing here has been tested on the blind vault.

## Where the evidence is

The design documents are [round-2.md](../design/round-2.md) and [amplification.md](../design/amplification.md); the day-by-day record with all figures is in the logbook for 25 to 27 November.
