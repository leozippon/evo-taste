# Project log — 2026-11-26

## Amplification pilot (pilot C), steps 0 to 45

Twenty-eight members: twelve seeded states and two unseeded controls, each cloned at step 0 into the relative-only rule and into the rule with self-imitation of the member's own successes; 45 steps of 48 problems × 4 attempts. All finished between 22:45 and 03:45. Reading below is from the in-run probes (one sample per item, about 360 items) and from context scores on 80 probe items; the two-sample evaluation of 3,520 selection items was still running when this was written.

### Training

- Training reward rose from about 0.44 to about 0.52 in both rules. About half the problems of a step have attempts with different rewards throughout.
- **Answers lengthen again.** Mean answer length in training went from about 600 tokens to between 500 and 1,600 depending on the member. Under self-imitation six of fourteen members exceed 1,000 tokens and 5 to 9% of their answers are cut off at 4,096; under the relative-only rule the unseeded members are the long ones. The slope from 2,048 tokens does not hold length back, and self-imitation works against it, because it imitates a success whatever its length. The accuracy lost to cut-off answers is estimated at about one point under self-imitation.
- Probes at step 45 against the base weights: +6 to +8 points on frontier items in both rules, +1 to +2 on unseen sources (not distinguishable from zero), no loss on mastered items or on held-out answer formats — except for the unseeded members, which lose 4.5 points on mastered items and 9 on held-out formats. Every member that went through distillation, the neutral context included, keeps both.

### Wash-out of the seeded contexts

How much of its context a member carries, on the scale where 0 is the base model and 1 the teacher's instructed answers, with the drift common to all contexts and the controls' drift removed. Noise of one value is about 0.03.

| Context | Level | After seeding | Step 45, relative only | Step 45, with self-imitation |
|---|---|---|---|---|
| `ground_up` | stance | 0.31 | 0.33 | 0.17 |
| `relations` | stance | 0.27 | 0.28 | 0.14 |
| `parsimony` | stance | 0.36 | 0.30 | 0.19 |
| `observe` | stance | 0.25 | 0.19 | 0.14 |
| `refute` | stance | 0.33 | 0.16 | 0.13 |
| `verify` | tactic | 0.09 | 0.09 | 0.03 |
| `small_case` | tactic | 0.07 | 0.11 | 0.00 |
| `direct` | tactic | 0.19 | −0.03 | 0.05 |
| `define` | tactic | 0.14 | −0.09 | −0.18 |
| `map` | tactic | −0.11 | −0.31 | −0.03 |
| `procedure` | reference | −0.08 | −0.02 | −0.02 |

- **Seeded stances are not washed out by the relative-only rule.** All five are still clearly present after 45 steps, at 49 to 106% of their strength after seeding; three rose in the first 30 steps before settling. In Round 1 every seeded habit but one was gone within tens of steps.
- **In every one of the five themes the stance outlasts its tactic**, under both rules (difference at step 45: 0.30 under the relative-only rule, 0.18 with self-imitation; interval over themes 0.13 to 0.47). Five themes cannot give a p-value below 0.06, and they give 0.06.
- **But the level of abstraction is probably not the operative variable.** The two tactics that are gone by step 30, `direct` and `define`, are the two contexts that lower the base model's accuracy (0.44 and 0.47 against 0.50 for the neutral context). The two tactics that do not lower it, `verify` and `small_case` (0.52, 0.53), persist as well as the stances, at the modest level at which they were absorbed. In absolute terms every persisting context carries about 0.01 nats per token, stance or tactic. The simplest reading is Round 1's conclusion made sharper: the reward removes within 30 steps what costs it, and leaves the rest alone. Stances survive because they are small, general and harmless, not because they are abstract; the design has no harmful stance to test the converse.
- **Self-imitation erodes seeded contexts** instead of amplifying them: the stances are at about half their strength by step 45 (difference to the relative-only rule 0.10, interval over items 0.07 to 0.14), and the helpful tactics are gone. A member's own context is identified in 4 of 11 members under the relative-only rule at step 45 and in 1 of 11 with self-imitation (chance 1 in 11). What self-imitation amplifies is length.
- No recitation: 0.7% of answers contain a listed term, as for the base model.

## Operations

- Another project of the same owner needs cards; its session asked for an estimate and was given one, later revised by an hour. No further wave of this project is started on cards as they empty, and the last four clones in the pilot config (the neutral-context and unseeded members under the relative-only rule) were withdrawn before starting, leaving twelve clone pairs.
- The step-45 evaluation was sized on the base model's answer length. Trained members write about twice as much, and half of its 3,520 items are the procedural format-probe units, which were meant to be a small side measurement. It takes about seven hours instead of three.

## Change of direction, 09:00

The user redirected the work: away from distilling instructions into members, toward training paradigms that reinforce and amplify a member's own biases so that it leaves the base model's average behaviour, on at most four GPUs.

- **Compute.** This project now holds four cards (1, 2, 5, 7); the owner's other project uses the other four. Stopped at once: the second batch of clone pairs (five seeded parents, at steps 61 and 71 of 90), the first batch's last worker (four clones at step 85 of 90, resumable), the scoring of the imitation side run, and all supervisors. The step-45 evaluation keeps the four cards until it finishes.
- **What carries over.** The pilots are the baseline for the new question: relative-only training removes what costs reward and otherwise leaves a member where it was; self-imitation amplified length, not disposition; a member starting from the bare base model is fragile. Nothing tried so far reinforces a lineage's own direction.
- **Design.** `docs/design/amplification.md`: an endogenous bias is a member's deviation from the base model on the same input, and three mechanisms are built to reinforce it along its own direction — exaggerated self-imitation (training toward the member's own distribution pushed further from the base model's, on its successes), self-chosen practice (more of what the member solves best), and scaling an adapter's deviation as a probe on existing members. Deviation and distinctness are measured by likelihood: how much more probable a member finds its own answers than the base model does, and than another member does.
- The seeding results stand as reported; the wash-out comparison is not extended further.

## Pilot C, evaluation at step 45

Base model and the 28 members on 3,520 selection items, two samples each: 1,840 items of the 23 real-data units, and 1,680 of the procedural units that serve as a format probe. Finished at 11:53 after eight hours.

- **Accuracy.** On the whole evaluation the base model scores 40.4 and members 40.6 to 48.4. On unseen real sources, members seeded by distillation gain +1 to +5.7 points over the base model; the four unseeded members gain −0.6 to +0.9. On held-out items of the training sources most members gain +4 to +7. Frontier items gain +3 to +11.5. Mastered items lose 0 to 7 points, the worst being the unseeded members under the relative-only rule and a few with many cut-off answers.
- **Members solve different problems.** The member-by-problem interaction, after removing each member's level within classes of item difficulty and correcting for sampling noise, has an SD of 14.0 points (interval 13.3 to 14.7): 13.5 under the relative-only rule and 15.6 with self-imitation, a difference well outside noise. The simulated power analysis had put the smallest detectable value near 4. A real-data null, the base model against itself on four samples, is being run to confirm that the estimator reads zero there.
- **Pairs differ widely.** The noise-corrected distance between two members' solve rates averages about 20 points per item, from indistinguishable (two members under the relative-only rule) to 32 points (the two rules applied to one seed).
- **Members' levels are reliable and partly transferable.** The between-member SD of accuracy is 1.8 points beyond noise (reliability 0.93). In a harvest simulation that ranks members on half the unseen sources of each domain and tests on the other half, the top member leads by 3.8 points where it was selected and by 2.0 where it was not (retained fraction 0.53, interval −0.6 to 1.6; rank correlation between halves 0.54, interval 0.23 to 0.69). Splitting items within sources instead, 0.83 is retained. The best of 28 is expected to lead by 2.0 points on unseen sources; in Round 1 the corresponding figure was negative. The blind vault's sources are further away than a held-out half, so this is an upper bound.
- **Length is the dominant drifting trait.** Mean answer length ranges from 440 to 1,711 tokens across members (between-member SD 331, noise 8). With self-imitation several members have 10 to 14% of their answers cut off, and on math up to 23%. How much of the member-by-problem interaction is a consequence of length and truncation is not yet separated.
- **Authorship.** A classifier trained on half the sources names the author of an answer from the other half 56% of the time (chance 3.6%); with layout and notation stripped, 48%; from layout alone, 21%. Unlike Round 1, most of what makes a member recognisable is not layout.
- **No format loss on real-data units**: strict and format-forgiving accuracy agree within 0.1 point for every member. Compliance with the requested wrapper is 95 to 98% under the relative-only rule and 86 to 99% with self-imitation, the shortfall being cut-off answers.

## In progress

- The step-45 evaluation of pilot C; from it, how much members differ in which problems they solve, and a harvest simulation inside the selection suite.
- The engine work for the amplification mechanisms, then pilot D: four arms of four unseeded lineages.
- Seven clone pairs of pilot C (three complete, four at step 85) remain for a heritability baseline once cards are free.
