# Project log — 2026-11-27

## Pilot E: rewarding a member's most characteristic correct answers

Two pilot C members at step 45 (`unseeded-adv`, `relations-adv`), each cloned into three variants that differ only in the sign of the taste-shaped reward, 45 further steps. Among a member's correct answers to one problem, the variant "own" pays more for the answer the member finds most probable relative to the base model, "reversed" pays more for the least, "neutral" is plain training. Read from in-run probes (about 360 items, one sample) and likelihood scores on 120 probe items; the two-sample evaluation is running.

| Variant | Deviation from the base model, steps 60 / 75 / 90 (nats per token) | Probe accuracy against neutral at step 90 |
|---|---|---|
| own | 0.39 / 0.74 / 1.15 | −8.4 points (−12.2 to −4.6) |
| neutral | 0.23 / 0.21 / 0.25 | — |
| reversed | 0.13 / 0.09 / 0.05 | +3.9 points (+0.4 to +7.1) |

- **It amplifies, and in each lineage's own direction.** The own-direction clones end about five times further from the base model than their neutral siblings, and the two of them are further from each other (2.9 nats per token) than any other pair; the two reversed clones are nearly indistinguishable (0.08). A classifier separates the two own-direction clones perfectly with or without layout, and only 14 to 20% of their deviation sits in tokens without a letter or digit. The pressure did not grow in strength, as designed; the deviation nevertheless grew steadily.
- **It costs accuracy in proportion.** Against the base weights at step 90 the own-direction clones are down 11 points on unseen sources and 17.5 on mastered items, with training reward 0.08 to 0.10 below neutral from step 60. Their answers also became shorter (about 600 tokens against 760), as the first check predicted.
- **The reverse helps.** Pulling correct answers back toward the base model leaves frontier gains intact, and at step 90 the reversed clones are ahead of neutral on held-out items of the training sources (+8.3) and on sweet-spot items (+11.2), while the neutral clones have started to lose mastered items (−6.0 against the base weights).
- **The two-sample evaluation agrees.** On the 1,840 real-data selection items at step 90: `relations-adv` neutral 55.8, own 50.8, reversed 57.0; `unseeded-adv` neutral 53.3, own 39.4, reversed 55.1 (each ±2.1). The parents scored 58.2 and 53.2 at step 45, so 45 more steps of plain training gained nothing, and the reversed variant lost least.
- Two parents cannot give a significance level; the effects are large against item-level intervals and have the same sign for both parents on every measure quoted.

**Reading.** Three ways of pushing a member away from the base model have now been tried — exaggerating its whole distribution, scaling its adapter, and rewarding its most characteristic correct answers — and all three trade accuracy for distance, the more distance the more accuracy. The base model's competence sits where its behaviour is typical. Sharp, lineage-specific identity is easy to produce and is not the same thing as better reasoning; at this model size it works against it. What remains open is amplification that never touches how an answer is written: pilot F, which consolidates the problems a member has itself cracked.

## Distance from the base model predicts less transfer, also without intervention

Across the 28 pilot C members at step 45, a member's deviation from the base model (0.06 to 0.28 nats per token) is negatively rank-correlated with what it gained: −0.43 with the gain on unseen sources, −0.50 on held-out items of the training sources, −0.58 on mastered items, and about zero (+0.13) with the gain on frontier items. The correlations have the same sign and similar size within each update rule (−0.47 to −0.65). The two unseeded members under the relative-only rule are the most deviated and the ones that gained nothing on unseen sources; the least deviated are members that started from a distilled adapter.

So the interventions and the observational data agree. Members learn the frontier equally well whether they move little or much; what the extra movement buys is loss of what the base model could already do. A member's deviation is measurable without any task data, which makes it a candidate predictor of transfer at harvest time that owes nothing to the selection suite.
