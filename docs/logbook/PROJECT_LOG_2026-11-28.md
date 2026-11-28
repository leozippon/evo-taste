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

Coordination with the other session (02:00): one of its nightly fits fell back to GPU 7 at 01:33 while the card was briefly empty and a scout measurement failed to start. At my request its fits now stay on GPUs 3 and 4. It reported that its first arm has ended, that GPUs 0 and 6 will be free after about half a day of scoring, and that GPUs 1 and 5 stay with its second arm until the afternoon of 29 November.

## Verifier: the expression loophole and three parsing defects closed

Rules version `2026-11-28.literal-forms` (details and measurements in `docs/dev/VERIFY_NOTES.md`).

- **Unevaluated expressions.** When the reference is a number in final form (integer, decimal, fraction of two integers, mixed number, scientific notation), an answer that is not a number in final form is no longer credited. A decimal reference still accepts an exact irrational form (`\pi` for 3.14). Re-judging every stored math record: 29 of 45,334 credited census answers lose credit and 4 of 12,586 in the pilot F evaluation; all 33 were read and all are unevaluated (`389 \cdot 463`, `8!`, `\binom{16}{7}`, `2^{18}`); nothing gains credit. About 1,500 credited answers containing an operator keep credit because their reference is itself an expression.
- **Three parsing defects found on the way**: clock times read as ratios (any `h:00` credited for `6:00`), statement-label lists such as `(2)(3)` read arithmetically, and thin-space thousands read as zero. Fixed; no stored verdict changes.
- **Consequences.** The eight Round 2 run directories made under the old rules are refused by the current code for training, resumption and new evaluation tags; their records remain readable. Census outputs now record the rules version and a census made under another version cannot be resumed; the 1.7B strata table stays readable (9 of 27,612 rows would change stratum). The suites file keeps its part assignment; only the rules line changed, and two references that used to fail their own check joined the train part.
- A full regeneration of the suites differs in nine code references whose own check depends on machine state (eight library-documentation and LeetCode items pass now, one that asserts a timing fails). The committed assignment was kept; the build is therefore not exactly reproducible on another machine state, which is recorded as a limitation.

## Scale scout: what a larger model costs

Full notes: `docs/dev/SCALE_SCOUT.md`. All measurements on single 46 GB cards with the engine's prompts, thinking disabled, rank-32 adapters.

Qwen3.8-27B is not a larger copy of the 1.7B: it is a hybrid of 48 linear-attention (Gated DeltaNet) layers and 16 full-attention layers. Consequences for the engine: five more projection names for the adapter (234M adapter parameters), no packed shared-prefix rows in the trainer, renamed adapter keys for the server, and a server whose concurrency is limited by a per-sequence recurrent state (about 52 MB), not by attention tokens.

| Model | Sampling, tokens/s | Training, tokens/s | Layout | Probe accuracy (item mean) | Mean answer, tokens |
|---|---|---|---|---|---|
| Qwen3-1.7B | 4,923 | 2,311 | one card | 0.52 | 511 |
| Qwen3-4B-Instruct-2507 | 2,093 | 1,162 | one card | 0.75 | 1,088 |
| Qwen3-14B | 1,264 | 563 | one card each | 0.69 | 558 |
| Qwen3.8-27B, 8-bit weights | 607 | 203 | one card each | 0.78 | 1,134 |

- **The 27B trains on two cards.** The official 8-bit weights are served by the sampler on one card and held by the trainer on the other, kept in 8 bits and expanded layer by layer at use. Sampler and trainer log-probabilities agree to 0.009 per token and 0.0010 per token summed, as closely as the 1.7B pair (0.0115 and 0.0013). In 16-bit form the weights alone exceed a card, and one worker would need four.
- **Cost of a member-step on the 27B**: about 4 minutes of sampling and 5 of training with 24 problems × 4 attempts; about 7 and 10.5 with 48 × 4. Answers that run to the token limit set a floor of about 4 minutes. The trainer is slow partly because the linear-attention recurrence runs in a pure-torch fallback; faster kernels exist and were not installed.
- **Accuracy with thinking disabled does not rise much with size on our probe set**: the 4B instruct model is close to the 27B, and the 14B is below both. The 27B writes answers twice as long as the 1.7B, and 8% reach the token limit.
- **The 1.7B's difficulty strata do not carry over**: larger models solve about 60% of the problems the 1.7B never solved. Each model needs its own census.

Decisions:

- The 27B in 8-bit form is the model of this stage, with one worker per two cards. With four cards that is two workers, about 290 member-steps a day at 24 × 4.
- The 4B instruct model is added to the method census as a third size, because it would run the same pilots at a quarter of the cost with no engine change if the answer to the first question does not depend on going all the way to 27B.
- The 14B is dropped: slower than the 4B and less accurate with thinking disabled.
