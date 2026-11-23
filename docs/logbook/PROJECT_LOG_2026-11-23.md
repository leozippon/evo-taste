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

## Open

- Blind vault 1 is being evaluated (all 36 members, both baselines, base). The pre-registered comparisons follow, under both strict and lenient scoring.
