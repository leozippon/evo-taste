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

## First checks of the amplification mechanisms, 14:00

The engine now has exaggerated self-imitation, self-chosen practice, a scaled-adapter probe and likelihood-based measures of deviation and of who wrote an answer. The developer's five-step check on one card was already informative.

- **Exaggerated self-imitation runs away.** At weight 0.5 a member's answers grew from 527 to 1,559 tokens in five updates and its probe accuracy fell from 0.43 to 0.29 (code from 0.47 to 0.06). The term's gradient grows with the deviation it produces: at weights 0.2, 0.1 and 0.05 its size relative to the advantage gradient rose over four updates to 2.3, 4.6 and 1.7 times. A fixed weight only sets when it takes over. After five steps the deviation from the base model was 0.27 nats per token with exaggeration against 0.05 without; 45 steps of plain training in pilot C had reached 0.20 to 0.26.
- **Scaling an adapter degrades it.** Unseeded pilot C members at 1, 1.5 and 2 times their adapter: accuracy 0.47, 0.35, 0.32 and 0.48, 0.40, 0.20; answer length 1,109, 1,518, 2,052 and 1,438, 2,381, 3,617 tokens; compliance down to 0.73 and 0.31. Training had taken them about as far along their own direction as pays.
- **Self-chosen practice follows common difficulty.** After five steps both practising members had shifted toward the same sources.
- A defect in the shared prompt stream surfaced: at two steps of a 48-problem course a batch has 49 prompts and one repeated item. It affected all pilot C members identically. It will be fixed for new runs under a version flag so that existing runs keep their batches.

**Reading.** A member's deviation is a mixture of what the reward shaped, habit the reward ignores, and length, and amplifying the whole mixture amplifies length first. The design was revised (`docs/design/amplification.md`): the next mechanism acts only among a member's correct answers to the same problem, raising the reward of the most characteristic one and lowering the least, with a clipped strength relative to the step, so that it cannot buy deviation with accuracy and does not strengthen as the member moves. The reversed sign is the control.

**Launched.** Pilot D, reduced to eight unseeded members on two cards: three relative-only, three with practice, two with exaggeration at weight 0.01. It characterises the undirected mechanisms at low strength and adds unseeded baselines. A deviation measurement of all pilot C members and clones runs on a third card. Pilot E, the taste-shaped reward on clones of four pilot C members in three variants, starts when its engine work is done.

- The real-data null for the member-by-problem interaction, the base model against itself on four samples, reads −131 pp² (interval −398 to +157) against 197 pp² measured between members: consistent with zero, though imprecise with a single model.
- The last four interrupted clones of pilot C finished their five steps; seven clone pairs are complete at step 90, and five more stopped at step 60 to 70.

## Length is not the divergence, 15:20

- **Length and truncation account for about 9% of the member-by-problem interaction** in pilot C at step 45 (interval 6.8 to 12.0%; the null gives 0.5%): the interaction SD goes from 14.0 to 13.2 points once each problem's dependence on answer length, truncation and update rule is removed. The share is 6% on logic, 14% on math and about a quarter on code, and it is the same on the 1,103 items that no member's answer was cut off on. Members of similar length differ among themselves 0.95 times as much as random groups. On simulated data the method attributes 97% when length causes all of the interaction and nothing when it causes none.
- **Members are not recognised by verbosity.** Answer length alone identifies the author 6.9% of the time (chance 3.6%); the first 600 characters of an answer still give 48.6% against 55.9% for the whole. Hiding length costs 14% of the classifier's accuracy above chance, and stripping layout and notation another 18%.
- **Deviation from the base model grows steadily under plain training.** In nats per token on a member's own probe answers: 0.13 at step 45 and 0.17 at step 90 under the relative-only rule; 0.14 and 0.25 with self-imitation; across members at step 45 from 0.06 to 0.28, unrelated to answer length. Longer answers dilute a member's deviation rather than carry it.
- In the five-step check, the source shares of two practising members moved together 87 to 94% (independent movement would give 50%).

So the starting point for amplification is better than assumed: independent members already move away from the base model and from each other in which problems they solve, mostly not through length or presentation. What is not yet known is whether those differences are inherited by a member's continuations; the likelihood test on the clone pairs is running.

## Selective amplification built; what its score favours, 16:10

- The taste-shaped reward is in the engine, with the reversed sign as control and exact vanishing at strength zero. In a five-step check on three clones of one member, the shaping contributed about a third as much advantage as correctness and did not grow.
- **Its score is partly brevity and phrasing.** Among correct answers to one problem the score falls with length (correlation −0.47 with log length, about a fifth of its variance), because a member's deviation is diluted over a longer answer; layout features add little; and in contrasting pairs the two answers share the method and differ in how they open and how much of the problem they restate. The mechanism is therefore expected to amplify presentation and to shorten answers, and its reverse to lengthen them.
- Pilot E was cut to one card: two pilot C members at step 45 (`unseeded-adv`, `relations-adv`), each cloned into own direction, neutral and reversed sign, 45 further steps, done by tomorrow morning. It answers what 45 steps of this pressure amplify.
- **The clones' solve profiles are being measured**: the fourteen clones that reached step 90 are evaluated on the 1,840 real-data selection items, so that the heritability of which problems a member solves can be read directly, next to the likelihood-based test.
- **Next mechanism, in development**: a member revisits the hard problems it has itself solved but not mastered, against a control that revisits equally hard, equally familiar problems regardless of its own successes. It acts on which problems a member can solve and never on how an answer is written.

## Pilot D stopped; clones inherit their parent's identity, 17:30

- **Exaggerated self-imitation collapses at any strength tried.** In pilot D, at weight 0.01, both members were gone by step 15: 94 and 96% of their answers hit the token limit and training reward fell from 0.44 to 0.26. The exaggeration loss grew from 0.07 to 5 while it applied to a single eligible answer per step. At weight 0.5 the same took five steps. The mechanism is abandoned.
- Pilot D was stopped at steps 10 to 16. Its practice arm follows the difficulty common to all members, and its unseeded baselines are no longer needed as parents. Its two cards went to pilot F.
- **Two continuations of one member stay much closer than unrelated members.** Fourteen clones of seven pilot C members, 45 steps after cloning, scored by cross-likelihood on 80 probe items: the distance within a clone pair is 0.135 nats per token (interval 0.125 to 0.146) and between unrelated members under the same rule 0.310 (0.290 to 0.334), a ratio of 0.43 (permutation p = 0.001; relative-only 0.51, self-imitation 0.42). Unrelated members have been apart for 90 steps and clone pairs for 45, so part of the ratio is time since separation; the comparison still shows that what a member has become is carried forward and not re-drawn at every step.
- Each member remains unmistakable: likelihood alone names the author of 99.6% of answers among the fourteen (chance 7%), twins included, in every domain. A text classifier reaches 72.5% and loses only 5% of its accuracy above chance when layout and notation are stripped. Differences in answer length explain about a third of the distances between members.
- **Frontiers are individual.** Each relative-only pilot C member has solved at least once, but not mastered, 593 to 714 of the 2,160 training problems it has seen. Two members' frontiers overlap with a Jaccard index of 0.29 (0.24 to 0.33), where members with identical per-problem solve rates would overlap 0.37 by sampling alone.
- **Pilot F** started: clones at step 45 of three pilot C members, each into revisits of its own frontier, revisits of the frontier of a different member (problems a comparable member cracked and this one did not), and no revisits. A control matched on difficulty instead is kept for one parent; it differs from the own-frontier arm in the member's prior success as well, which is why the other-member control was added.

## What clones inherit, 22:20

The fourteen clones of seven pilot C members were evaluated at step 90 on the 1,840 real-data selection items, two samples each, and compared with their parents and with each other.

- **Clone pairs solve more similar problems than unrelated members**: the noise-corrected, difficulty-adjusted distance between solve rates is 414 pp² within a clone pair (interval 271 to 546) and 653 between unrelated members under the same rule (519 to 783), a ratio of 0.63 (0.45 to 0.81; permutation p = 0.001).
- **That is time since separation, not lock-in.** Two clones, 45 steps apart, are 326 pp² apart under the relative-only rule and 395 with self-imitation. Unrelated members at step 45 were 323 and 424 apart, and by step 90 they are 514 and 604 apart. Which problems a member solves keeps moving at about the same pace whether or not two members share a history; a clone moves away from its twin as fast as two strangers moved apart in their first 45 steps.
- **Scalar traits are inherited.** The regression of a clone's trait at step 90 on its parent's at step 45 has a slope near 1 for answer length (1.04, interval 0.12 to 3.4), code length (1.61, 0.52 to 2.57), imports (1.29, 0.77 to 1.77), headers (1.33, 0.92 to 1.96) and lines of explicit checking (1.12, 0.41 to 2.63); for accuracy the slope is 0.96 with an interval that includes zero. Intraclass correlations point the same way (accuracy 0.64, loops 0.93, imports 0.76) but seven families give wide intervals.

**Reading.** A lineage inherits how it writes, how long, its coding habits and probably its level. It does not hold on to its particular set of solved problems: that set diffuses steadily under further training. Substantive divergence between members is therefore real at any moment and half transferable to unseen sources, but it is accumulated drift, not a trait that a lineage keeps and builds on. Frontier consolidation is aimed exactly here, and pilot F was doubled to six parents on the card that became free.

## In progress

- Pilot E (taste-shaped reward, two parents) and pilot F (frontier consolidation, six parents), finishing between 05:00 and midday on 27 November; then their evaluation at step 90.
