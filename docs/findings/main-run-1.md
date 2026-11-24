# Main run 1: results

Thirty-six independent copies of Qwen3-1.7B were each seeded with a different bias, trained on their own attempts for 120 steps in the same mixed environment, ranked on task families none of them had trained on, and the top one was then tested on a sealed set of further unseen families and external benchmarks. The protocol, its hypotheses and the rules that keep the test honest are in [the protocol](../design/protocol.md); this page reports what happened.

## In short

The population did not produce a model with transferable Taste. Members did become individually recognisable — by step 120 a classifier identifies which of the 36 wrote an answer to an unfamiliar problem 57% of the time, against 3% by chance — but what distinguishes them is mostly layout and notation, the bias each was seeded with is mostly not what they end up sharing, and their differences in transfer are modest, not heavy-tailed, and no larger for deliberately biased members than for members that differ only by random seed. The member harvested on the selection suite ranked 27th of 36 on the blind suite, below the population average. Its lead on the selection suite came from having drifted least in answer format, a trait that does not carry over.

Two things did come out clearly. Training without a leash erodes a capability the untrained model has — writing answers in the requested format — by about ten points on unseen families, while reasoning improves by more than twenty. And training longer is a poor use of compute: three of three long single-model runs kept improving on the tasks they trained on while getting worse on blind tasks, and the one that was continued destroyed itself. Beyond one short run, extra compute bought nothing on blind tasks, whether spent on a longer run or on a population.

## What was run

Each member is a rank-32 adapter on the frozen base model. Twenty-four members were seeded with one of twelve reasoning habits, two members per habit; six with a generic instruction to reason step by step; three plainly; three from a random perturbation. All then trained without any prompt on 31 task families, 128 prompts × 4 attempts per step. Two single-model baselines used the same recipe with more compute: a *long* one (720 steps, six members' worth) and a *wide* one (larger batches with 16 attempts per prompt, evaluated at three members' worth).

Final scores are exact-match accuracy averaged over families, from 200 fresh problems per family and two attempts each. The selection suite has 21 families never trained on, the blind suite 18 others plus GSM8K, MATH-500 and MMLU-Pro. The evaluation noise of one member's score is about half a point.

## Every member learned the same broad lesson

| | Train families (fresh problems) | Selection suite | Blind suite | GSM8K / MATH-500 / MMLU-Pro |
|---|---|---|---|---|
| Untrained model | 12.7 | 20.0 | 23.7 | 70.1 / 29.9 / 29.4 |
| Population mean (SD over 36 members) | 32.8 (2.9) | 34.1 (2.5) | 33.3 (2.0) | 79.7 / 57.8 / 41.0 |

Without its thinking mode the base model answers most problems directly. Every member, whatever its seed, moved toward writing out its reasoning — bare answers fell from 46% of probe responses after seeding to 16% at step 30 and 7% at step 120, with a single member still above a third — and that is where most of the gain on unseen tasks comes from, including a near doubling on MATH-500. The large differences that seeding created, more than 40 points apart on the foundation probe, were erased rather than amplified: the spread of the in-training selection probe across members fell from 6.0 points after seeding to about 3 from step 60 on. The two members seeded with the terse habit were culled at the first check for having lost half the base model's foundation accuracy; their replacements, seeded the same way, recovered like the rest.

## Members diverged in style, but not in the way they were seeded

A classifier trained on members' answers to train-family problems, with answers and digits masked, was asked who wrote an answer to a problem from an unseen family.

| Step | 0 | 15 | 30 | 60 | 90 | 120 |
|---|---|---|---|---|---|---|
| Correctly attributed | 13% | 31% | 39% | 53% | 57% | 57% |

Chance is 3%. The variance of style features between members grew by about 60% from step 15 to step 120 while their accuracy converged. The features that separate members best are bold text, headers, numbered lists and line length: a house style of presentation.

The seeded habit mostly did not survive as a shared signature. A classifier trained to recognise a habit from some lineages and tested on a held-out lineage of the same habit was right 35% of the time after seeding, 27% at step 60 and 15% at step 120, against 7% by chance. At step 120 one habit is still clearly shared by both of its members — *procedure*, recognised in 94% and 53% of their answers — and one member of another reaches 33%; for the other ten habits the held-out member is recognised between 0% and 18% of the time. The planning habit, recognised in its sibling 55–83% of the time up to step 75, had vanished by step 90. What most lineages end up with is their own.

## Differences in transfer are real, modest, and owe nothing to the seeded bias

On the selection suite members range from 28.1 to 38.9. After subtracting evaluation noise the spread between members has a standard deviation of about 2.4 points (variance 5.9, 95% interval 3.2 to 8.6); on the blind suite about 1.9. So independent runs of the same recipe do end up measurably different on unfamiliar tasks. But:

- The distribution has no heavy upper tail. It is slightly skewed downward, and the best member is 1.7–1.8 robust standard deviations above the median on both suites.
- Biased members are not more spread out than seed-only members. The noise-corrected variance is 3.6 for the 24 habit-seeded members and 2.7 for the six generic controls on the selection suite (difference 0.9, interval −1.6 to 4.1), and 3.5 against 3.1 on the blind suite.
- Which habit a member was seeded with explains none of it: two seeds of one habit differ as much as two habits (intraclass correlation 0.10 on selection, −0.23 on blind).
- The final ranking is not set early. A member's selection probe after seeding correlates 0.26 with its final selection score, and the correlation stays between 0.17 and 0.52 until step 90.

## The harvested member did not hold up

The ranking was committed before the blind vault was opened. The top member on selection was the first seed of the *procedure* habit.

| | Selection | Blind |
|---|---|---|
| Harvested member | 38.9 (1st of 36) | 32.0 (27th of 36) |
| Minus population mean | +4.8 (4.1 to 5.5) | −1.3 (−1.9 to −0.6) |
| Minus median member | +4.6 (3.7 to 5.6) | −1.1 (−2.1 to −0.1) |
| Minus long baseline at 120 steps | +3.1 (2.2 to 4.1) | −3.6 (−4.7 to −2.7) |
| Minus wide baseline | −0.1 (−1.1 to 0.8) | −2.3 (−3.2 to −1.4) |

None of its lead was retained: the retained fraction of the selection differential is −0.27 (−0.42 to −0.12) for the top member and 0.00 (−0.10 to 0.10) for the top three. Across all members the selection and blind scores are only weakly related (rank correlation 0.37, interval 0.03 to 0.68). On the blind suite no family that the base model cannot solve was unlocked by any member, and the harvested member solved 36 problems that neither the base model nor any of the five baseline checkpoints ever solved, against 828 the other way round.

## Why: the selection score rewarded format, and format does not transfer

Reading the selection records showed members failing on format alone. On one geometry family nearly every member computes the right angle, but the verifier accepts `237` and not `237.0`; one member scores 92%, the median member 14%. On the integration and polynomial families the task asks for `x**9`; most trained members write `x^9`, `x⁹` or LaTeX and fall below the untrained model, which followed the instruction.

Re-scoring every wrong answer after a meaning-preserving clean-up of notation separates what a member worked out from how it wrote it down:

| Selection suite | As scored | Format forgiven | Lost to format |
|---|---|---|---|
| Untrained model | 20.0 | 20.5 | 0.5 |
| Population mean (SD) | 34.1 (2.5) | 44.0 (2.9) | 9.9 (1.5) |
| Harvested member | 38.9 | 44.5 (18th of 36) | 5.7 |
| Long baseline at 120 / 240 / 480 steps | 35.8 / 36.0 / 38.5 | 46.8 / 46.7 / 49.1 | 10.6–11.0 |
| Wide baseline | 39.0 | 50.4 | 11.4 |

The harvested member reasons no better than the average member; it simply lost the least to format. That is not a property of its seeded habit either: its sibling lost 10.4 points, and across the twelve habits the format loss of one member says little about the other's (intraclass correlation 0.12). On the blind suite, whose families happen not to be sensitive to notation, the clean-up changes almost nothing (14 of 240,000 wrong answers), so that advantage had nowhere to go. The part of the selection score that does carry over is the format-forgiven part: it correlates 0.53 with the blind score across members (0.25 to 0.73), the format loss 0.11.

This is the mechanism the project set out to find, in a form nobody would call Taste. Notation is a direction the training reward does not constrain; along it each lineage drifts its own way; on some unseen families the drift happens to matter enormously. The result is real, heavy-tailed variation on single families — and a selection rule that harvests it picks up a habit of writing, not a way of thinking.

One style trait does go with better transfer: members that settled on longer answers reason better on unseen families (rank correlation of answer length with format-forgiven selection accuracy 0.70, with the blind score 0.39, with the external benchmarks 0.60). How long a member's answers are after seeding does not predict where it ends up (0.27), so this too is drift rather than an amplified initial bias.

## Against single-model training with the same compute

The fair comparison for a harvest from n members is one model given n members' compute. The harvest column is the expected blind score of the member that ranks first on selection among n members drawn at random from the population.

| Compute (members' worth) | Harvest from n members | Single long run, three seeds (mean) | Single wide run |
|---|---|---|---|
| 1 | 33.3 | 35.6, 33.5, 33.9 (34.3) | |
| 2 | 33.7 | 35.0, 32.4, 33.4 (33.6) | |
| 3 | 33.8 | | 34.3 |
| 4 | 33.9 | 31.8, 27.8, 32.9 (30.8) | |
| 6 | 33.9 | 27.9 (one seed) | |
| 36 | 32.0 | | |

The long baseline was run three times, the first to 720 steps and two replications to 480; the replications were scored on the blind vault after it had been opened, which is sound here because nothing is selected among them. At one and two members' compute a single run and a harvest cannot be told apart: one run's blind score varies by about two points from seed to seed, and the first long run at 120 steps was simply a good draw. Harvesting on the selection suite gains half a point at most.

Training longer is worse. All three long runs lost ground on the blind suite between 120 and 480 steps, by 3.8, 5.7 and 1.0 points, while their accuracy on the families they trained on rose from 33.9 to 36.0 on average and their selection score held (36.8 to 37.5). The losses sit in particular families and can be extreme: one run's accuracy on a true/false family went from 50% to 2%, another's from 74% to 40%. The external benchmarks stayed roughly level. The run that continued past 480 steps collapsed outright after step 500, to below its own starting point on every suite and below the untrained model on the foundation probe; an audit found nothing irregular in its checkpoints or restarts. A long run's selection score also swings by several points from one checkpoint to the next (39.8, 34.7 and 37.6 at 120, 240 and 480 steps for one replication), a further sign that what it measures is not stable.

So beyond one short run, extra compute bought nothing on blind tasks in either form: spent on a longer run it loses three to four points, and spent on a population it gains half a point from the harvest. The population's only advantage over the long run is that it never trains any member for long.

The wide baseline, at three members' compute, is the strongest single recipe on what it trained on (40.6 on train families against a population best of 38.4) and on reasoning with format forgiven, and sits one point above the population mean on the blind suite. It is one run, evaluated at half its planned length; its extension could not be completed on the shared machine.

## Verdict on the hypotheses

| Hypothesis | Verdict |
|---|---|
| Members become more distinguishable while each becomes more self-consistent | Supported for distinguishability. The distinguishing features are presentational, and induced biases do not diverge further than seed noise. |
| Transfer performance is spread beyond noise, and more so for biased members | First half supported, second half refuted. No heavy tail. |
| The harvested member beats the base model, the median member and compute-matched single models on blind tasks | Refuted, except against the base model, which every member beats. |
| Members sharing an induced bias end up alike, and style explains blind performance | Refuted for induced bias, in style (one habit of twelve excepted) and in transfer. Of the style features only answer length relates to transfer. |

## What this says about the idea

The premise was that the lock-in and mode sharpening seen in single runs would, across many runs, amplify different starting biases into different and occasionally superior ways of reasoning. In this setting the two halves of that sentence come apart. Where the reward has a say — reason or not, how carefully, how long — it has the same say for every member, and starting biases are overwritten within tens of steps. Where the reward has no say, members do drift apart and lock in, exactly as predicted, but those are by construction the traits that do not matter in training, and here they turned out to be notation and layout. Divergence and consequence were found in different places.

For Taste in the intended sense to emerge this way, the environment would have to contain choices that are neutral or nearly so during training, substantive rather than cosmetic, and decisive elsewhere — several genuinely different strategies that all solve the training tasks. A mixed bag of procedural families at one difficulty does not seem to offer many. That is a statement about this environment and scale, not a proof about the idea.

## Limitations

- One base model, one environment, one population, 120 steps. The long baseline is three runs to 480 steps and one beyond; the wide baseline is one run evaluated at half its planned length.
- The format-forgiving re-scoring and everything built on it — including the observation that selecting on format-forgiven accuracy would have picked a member 1.9 points above the blind average — were done after the blind vault was opened. They explain the result; they are not confirmed by an independent test. The second blind vault is unopened and reserved for that.
- Exact-match verifiers make notation consequential. With tolerant verifiers the selection suite would have ranked members differently, and the harvest might have retained a small real advantage.
- The style features are surface measures. A reasoning strategy that leaves no trace in layout, vocabulary or length would be missed by them, though not by the transfer results.
