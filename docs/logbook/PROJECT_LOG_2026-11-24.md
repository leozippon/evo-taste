# Project log — 2026-11-24

## Long-baseline replications

`main1-tp-rep`: two further runs of the long single-model baseline with new seeds, 480 steps each, forked from `main1-tp`. Checkpoints at 120, 240 and 480 steps were evaluated on the train suite, the selection suite and blind vault 1 with the sizes used for the main run. The vault had already been opened; nothing is selected among baselines, and the ledger records the reuse.

| Run | Steps | Train families | Selection | Blind |
|---|---|---|---|---|
| first | 120 / 240 / 480 / 720 | 36.8 / 37.4 / 38.8 / 27.7 | 35.8 / 36.0 / 38.5 / 28.7 | 35.6 / 35.0 / 31.8 / 27.9 |
| replication b | 120 / 240 / 480 | 32.8 / 33.5 / 35.8 | 39.8 / 34.7 / 37.6 | 33.5 / 32.4 / 27.8 |
| replication c | 120 / 240 / 480 | 32.2 / 35.8 / 33.4 | 34.9 / 35.7 / 36.3 | 33.9 / 33.4 / 32.9 |
| mean of three | 120 / 240 / 480 | 33.9 / 35.5 / 36.0 | 36.8 / 35.5 / 37.5 | 34.3 / 33.6 / 30.8 |

- **The decline on blind tasks with longer training is systematic.** All three runs are lower at 480 steps than at 120, by 3.8, 5.7 and 1.0 points, while train-family accuracy rises. The loss is concentrated in particular families: replication b answers one true/false family correctly 2% of the time at step 480, against 50% at step 120. GSM8K, MATH-500 and MMLU-Pro stay roughly level.
- **The first run's 120-step blind score was a good draw.** The three runs at 120 steps score 35.6, 33.5 and 33.9; the population mean is 33.3 with a between-member SD of 2.0.
- **Neither replication collapsed within 480 steps.** Replication c's training reward fell after step 360 (0.35 to 0.31) without a crash. The first run's collapse came after step 500, beyond where the replications stop, so how common an outright collapse is remains open.
- **Selection scores are unstable along a run**: 39.8, 34.7, 37.6 for replication b.

Consequence for the compute-matched comparison: at one and two members' compute a single run and a harvest are indistinguishable; at four, a long run averages 30.8 against an expected 33.9 for the harvest. Beyond one short run, extra compute bought nothing on blind tasks in either form. `docs/findings/main-run-1.md` and `docs/summary.md` are updated.

## Not completed

- The wide baseline's extension from step 68 to 120: no card had the roughly 31 GiB it needs for the hours required. It stands at its 60-step evaluation.

## State at the end of the round

- Both blind-vault-1 evaluations are in the ledger; vault 2 is unopened.
- Engine, analysis tools, protocol, findings and logs are committed locally. `origin` (`git@github.com:leozippon/evo-taste.git`) does not exist and the available token cannot create it, so nothing has been pushed.
- Run directories on the compute host: `runs/pilot0`, `runs/main1` (unpenalised start, kept as the record of the length drift), `runs/main1-tp` (the main run), `runs/main1-tp-rep`.
