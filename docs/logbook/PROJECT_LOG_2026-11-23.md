# Project log — 2026-11-23

## Main run completed

- `main1-tp` finished at 08:52: all 36 population slots hold a lineage with 120 amplification steps. Two lineages were culled on the way, both seeded with the `terse` habit and both at the first check (step 30) because their foundation accuracy was below half the base model's; their replacements ran the full course and were not culled.
- The run survived a crowded machine through the supervisor, frequent checkpoints and out-of-memory retries. An audit of the resume path (`docs/dev/audits/RESUME_AUDIT_2026-11-23.md`) found resumes bit-exact for weights and optimizer state, so the uneven number of restarts across members is not a confound. One worker ran for seven hours from a working tree with uncommitted changes in files the worker does not load; this is recorded, not corrected.
- Two engine defects surfaced and were fixed without affecting data already produced: a task generator that raises on two specific training items (now redrawn deterministically; every member passed those positions under the fixed code), and one evaluation instance whose gold answer its own scorer rejects (redrawn in evaluation sets only; the training stream is untouched).
- Stopping rule applied: every member completed its 120 steps before the 14:00 limit, so the harvest step is 120. The long baseline finished its 720 steps; the wide baseline was stopped by hand at step 68 and is evaluated at its last probe step, 60, which is three members' worth of rollouts.

## What training did

- **Fitness converged, as in the pilot.** The spread of in-loop selection-probe accuracy across members shrank from an SD of 6.0 points after seeding to 2.6–2.8 from step 60 on, close to the probe's own noise.
- **Style diverged.** A text classifier trained on members' answers to train-family probes identifies the author of an answer to an unseen-family probe 13% of the time after seeding, 31% at step 15, 53% at step 60 and 57–60% at steps 105–120, against a chance level of 2.8%. The between-member variance of the style features grew by about 60% between steps 15 and 120. The features that separate members most are layout habits: bold text, headers, numbered lists, line length.
- **The seeded habit is not what members end up sharing.** A classifier trained to recognise a scaffold from one of its two members does not recognise the other at step 120 (0–18% per held-out member, chance 7%); for some scaffolds it did at step 0–75 (`plan`: 58–83%) and then stopped. Each lineage ends with its own signature rather than its scaffold's.
- **The long single-model baseline collapsed.** It improved steadily to about step 480 (training reward 0.39, selection probe +16 points over base) and then degraded abruptly from step 500: training reward 0.18 at step 600, foundation probe below the untrained model. Nothing in its state at the preceding resume was irregular. This is a single trajectory destroying itself under unconstrained RL at four times a member's budget.

## Selection suite (families never trained on), final evaluation

200 instances per family × 2 samples on 21 families, plus 300 items from each foundation source; evaluation noise is 0.4 points per member.

- Untrained model 20.0. Population members 28.1–38.9, mean 34.1, SD 2.5. Noise-corrected between-member variance is 5.9 pp² (95% interval 3.2–8.6), a true SD of about 2.4 points: members really differ on unseen families.
- **No heavy upper tail.** The distribution is slightly left-skewed (the three perturbation members are the low end, 28.1–31.1); the top member is 1.8 robust SDs above the median.
- **Induced biases do not spread more than seed noise.** Excess variance is 3.6 pp² for the 24 style-scaffold members and 2.7 for the six generic-reasoning controls (difference +0.9, interval −1.7 to 4.3, permutation p = 0.65). Scaffold identity explains almost none of the variance (intraclass correlation 0.10, p = 0.32): two seeds of one scaffold differ as much as two scaffolds.
- **The ranking is not set early.** The rank correlation between a member's selection probe during training and its final selection score is 0.26 at step 0 and stays between 0.17 and 0.52 until step 90.
- Harvested outlier: `procedure-a` at 38.9, ahead of `plain-2` (38.6) and `cot-6` (37.1). Baselines: long 35.8 at 120 steps, 36.0 at 240, 38.5 at 480, 28.7 at 720; wide 39.0 at 60 steps. The manifest was committed (`3cf15ca`) before blind vault 1 was opened.

## Much of the difference on unseen families is answer format

Reading records showed members failing for format alone: `237.0` where the verifier wants `237`; `x^9`, `x⁹` or LaTeX where the task asks for `x**9`. A format-lenient re-scoring (meaning-preserving canonicalisation, then the family's own scorer or symbolic equality; audited for false credit) gives:

| | strict | lenient | lost to format |
|---|---|---|---|
| Untrained model | 20.0 | 20.5 | 0.5 |
| Population, mean ± SD | 34.1 ± 2.5 | 44.0 ± 2.9 | 9.9 ± 1.5 |
| Harvested `procedure-a` | 38.9 | 44.5 | 5.7 |
| Long baseline, 120 / 240 / 480 / 720 steps | 35.8 / 36.0 / 38.5 / 28.7 | 46.8 / 46.7 / 49.1 / 37.1 | 8–11 |
| Wide baseline, 60 steps | 39.0 | 50.4 | 11.4 |

- Training costs every model about ten points of format compliance that the untrained model has: a foundational capability regresses while reasoning improves by more than twenty points.
- The harvested member is 18th of 36 on lenient accuracy. Its lead on the strict score comes from having drifted least in format, not from reasoning better.
- Both single-model baselines reason better than the population mean, and the wide baseline better than every statistic of the population we have looked at, at three members' compute.
- Longer answers go with better lenient accuracy across members (rank correlation 0.69) and with more format loss (0.49); no style feature predicts the strict score.

## Blind vault 1

Opened once, after the manifest was committed: base model, 36 members at step 120, long baseline at 120 / 240 / 480 / 720 steps, wide baseline at 60 steps; 200 instances per family × 2 samples on 18 families, 650 items per external source (all 249 of MATH-500's share). The ledger is committed. Full account: `docs/findings/main-run-1.md`.

- **The harvested member did not hold up.** `procedure-a` scores 32.0 on the blind suite, 27th of 36, 1.3 points below the population mean (interval −1.9 to −0.6) and 3.6 below the long baseline at the same compute. The retained fraction of its selection lead is −0.27 (−0.42 to −0.12); for the top three, 0.00.
- **Selection and blind scores are weakly related across members** (rank correlation 0.37, interval 0.03 to 0.68). With format forgiven on the selection side the correlation is 0.53; the format loss itself correlates 0.11. The blind families are not sensitive to notation (14 of 242,700 wrong answers rescued), which is why the harvested member's advantage vanished.
- **Blind spread** is real (noise-corrected SD about 1.9 points), without a heavy tail, no larger for habit-seeded members than for seed-only controls (3.5 against 3.1 pp²), and unrelated to the seeded habit (intraclass correlation −0.23).
- **Everyone transfers.** Blind procedural accuracy rises from 23.7 to 33.3 on average, GSM8K from 70.1 to 79.7, MATH-500 from 29.9 to 57.8, MMLU-Pro from 29.4 to 41.0. No blind family that the base model cannot solve was unlocked by any adapter.
- **The long baseline gets worse on blind tasks as it trains**: 35.6, 35.0, 31.8 and 27.9 at 120, 240, 480 and 720 steps, while its train and selection scores rise until the collapse. The wide baseline scores 34.3 at three members' compute.
- **Harvest against single-model training at equal compute.** The expected blind score of the member ranked first on selection among n random members is 33.3 (n = 1), 33.7 (2), 33.8 (3), 33.9 (4 and 6): half a point gained by selecting. Single runs with the same compute score 35.6, 35.0, 34.3, 31.8 and 27.9; one run varies by about two points from seed to seed, so the first three are not distinguishable from the harvest, and the last two are the long run's decline and collapse.
- **After the fact**: ranking by format-forgiven selection accuracy would have picked `plain-2`, 1.9 points above the blind mean; answer length at step 120 correlates 0.39 with the blind score and 0.60 with the external benchmarks. Neither was specified in advance.

## Verdict and reading

Divergence supported, consequence not. Of the four hypotheses only the first holds, and in a weaker form than intended: lineages become individually recognisable, but by presentation, and seeded biases neither persist (one habit of twelve excepted) nor spread members further than seed noise. The shared reward decides everything it has a say in; what is left to drift is what the reward ignores, and in this environment that is notation, layout and to some extent answer length. The one drifting trait with large consequences, answer format, matters only where a verifier is strict about it, so selecting on it buys nothing elsewhere.

## Follow-up started

- `main1-tp-rep`: two replications of the long baseline with new seeds, 480 steps each, forked from `main1-tp` so that seeding pools and probes are shared. Question: is the long run's decline on blind tasks from step 240, and its instability beyond step 480, typical or one unlucky seed. They will be evaluated on selection, train and the already opened vault 1, which is legitimate here because nothing is selected.
- The wide baseline was resumed from step 68 toward its planned 120 steps, to complete the comparison at six members' compute.
- A fix made this morning had made probe-set construction depend on machine load (gold checks could time out and trigger a redraw). It was caught by the probe-set guard when the replication workers started, and corrected. All probe and evaluation sets of `main1-tp` were verified against a deterministic rebuild; no data was affected.
- The results page first went to `docs/results/`, a directory name excluded from sync and git; it lives in `docs/findings/`.

## Proposed next round

The second blind vault stays sealed for a redesigned experiment. What the first round suggests that design must change:

1. **An environment with real strategic choice.** Tasks that admit several substantively different solution strategies with similar training reward and different behaviour out of distribution, so that what drifts is a strategy rather than notation. The derive-versus-enumerate tasks of *Not All Reasoning Generalizes* are the template.
2. **A selection suite that cannot be won on format.** Tolerant verifiers, or selection on format-forgiven accuracy, with the blind suite scored the same way.
3. **Protection of what training erodes.** Format compliance fell by ten points in every trained model; a small share of format-checked prompts in the training mix, or replay of the base model's compliant answers on families where compliance drops, is the targeted protection the protocol left unbuilt.
4. **A direct test of heritability before any harvest.** Clone members that rank high and low, continue each clone with fresh sampling noise, and measure whether the parents' difference survives. In this run a member's standing at step 60 predicts its standing at step 120 only weakly, which already suggests little would survive.

## Open

- Replication results and the completed wide baseline.
- `origin` still does not exist; nothing has been pushed.
