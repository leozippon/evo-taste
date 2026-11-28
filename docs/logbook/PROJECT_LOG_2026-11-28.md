# Project log, 28 November 2026

## The anchoring test: reversed taste-shaped reward on nine lineages

The reversed taste-shaped reward (among a member's correct answers, the least characteristic earn most) had come out 1.2 and 1.8 points above plain training in the two lineages of pilot E. Seven more lineages were trained to test it: the step-45 members `placebo-adv`, `ground_up-adv`, `small_case-adv`, `observe-adv`, `verify-adv`, `procedure-adv` cloned into the reversed rule (their plain clones already existed from pilot F), and `unseeded-b-adv` cloned into both the plain and the reversed rule. All ran 45 further steps. Evaluation tag `r90`: 1,861 selection items (one per procedural family, 80 per real-data source), two samples; the comparison uses the 1,840 items every clone answered.

Reversed against plain, same parent, nine parents (interval over parents and items; p from the exact sign-flip test over parents):

| Measure | Difference | Interval | Parents agreeing | p |
|---|---|---|---|---|
| Accuracy, all items | −0.6 | −2.2 to +0.8 | 5 of 9 | 0.38 |
| Held-out items of training sources | −0.5 | −3.0 to +1.8 | 4 of 9 | 0.66 |
| Unseen sources | −0.7 | −2.0 to +0.9 | 6 of 9 | 0.32 |
| Problems the base model never solved | −3.4 | −5.2 to −1.7 | 9 of 9 | 0.004 |
| Problems the base model solves sometimes | −1.2 | −4.0 to +1.2 | 5 of 9 | 0.28 |
| Problems the base model has mastered | +2.2 | +0.6 to +4.0 | 8 of 9 | 0.027 |
| Math | −2.7 | −4.4 to −0.7 | 8 of 9 | 0.008 |
| Code | +0.2 | −3.2 to +3.2 | 5 of 9 | 0.92 |
| Logic | +0.1 | −1.7 to +2.0 | 5 of 9 | 0.96 |
| Answer length, tokens | −292 | −388 to −187 | 8 of 9 | 0.008 |
| Answers cut off, points | −0.9 | −1.3 to −0.5 | 8 of 9 | 0.008 |
| Deviation from base, nats per token | −0.133 | −0.172 to −0.098 | 9 of 9 | 0.004 |

Per parent, accuracy difference and change in deviation: relations +1.3 (−0.221), unseeded +1.9 (−0.175), unseeded-b +0.6 (−0.218), verify +2.0 (−0.125), observe −1.0 (−0.121), ground_up −2.4 (−0.106), small_case −2.3 (−0.093), procedure −2.6 (−0.078), placebo −3.4 (−0.062). The four parents that gained are the four in which the rule removed the most deviation; rank correlation between change in deviation and change in accuracy −0.83 (permutation p 0.008). The two unseeded parents gain 1.3 on average, the seven distilled ones lose 1.2.

For comparison, the forward rule against plain in its two lineages: accuracy −9.5 (−5.1 and −13.9), deviation +0.89.

Reading:

- The gain reported on 27 November from two lineages does not hold. Over nine lineages the reversed rule changes nothing on balance.
- It is a trade. It keeps mastered problems, shortens answers and removes truncation; it learns less of the frontier in every lineage, and loses on math.
- The frontier cost is close to mechanical. A correct answer to a problem the base model never solves is an answer the base model finds improbable, so a rule that pays least for a member's least base-like correct answers pays least for its discoveries.
- This qualifies the earlier statement that distance from the base model is only a cost. Across members under plain training, deviation did not correlate with frontier gain; but removing deviation by intervention removes frontier gain. Movement away from the base model is how a member learns what the base model could not do, and also how it loses what the base model could. What neither direction of the rule does is separate the two.
- Anchoring is useful as a repair: where a lineage had drifted far (the unseeded ones, `relations`), pulling it back paid.

The findings page, the summary and the amplification design were corrected accordingly.

## Larger base model: start of the stage

The user's direction of 27 November (a larger base model, up to Qwen3.8-27B) is taken up as a separate stage, designed in `docs/design/scale.md`. The order of questions puts first the one that needs no training and decides whether the rest is worth its cost: whether the larger model's own correct answers to one problem differ in method, where the 1.7B's did not. The build spec is `docs/dev/SCALE_SPEC.md` (work package WP-M, the method census).

Running in parallel since 01:00:

- A feasibility scout: models available in trainable form, layouts on 46 GB cards, measured sampling and training speed, and sampler/trainer agreement. Its downloads (float weights of the 27B and the 14B) were interrupted by a session restart and resumed. Because GPUs 1 and 5 are lent to the owner's other session until about 30 November, the scout was also asked whether one 27B worker fits on two cards with the weights kept in 8 bits on both sides.
- A developer building the method census, without GPU use until the scout's measurements are done.
- A developer closing the known verifier loophole (an unevaluated arithmetic expression credited as its value), now that no run is in progress. This bumps the rules version, so the finished 1.7B pilots stop being evaluable under the current code; their records stay.

The user did not answer the question of whether to ask the other session for GPUs 1 and 5 back and said to continue. The default taken: work on GPUs 2 and 7 until that session's arm ends, and ask it first if a run needs more.
