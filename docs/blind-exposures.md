# Blind exposures

Model outputs that exist on vault items, or on copies of them, outside a recorded opening. The [ledger](blind-ledger.md) records openings; this file records everything else. At the opening of the vault every result is reported with and without the exposed items.

## Round 1 pilot runs, before 22 November 2026

Three early runs were prepared before MMLU-Pro was re-dealt on 22 November. They used 57 MMLU-Pro questions as evaluation items that the re-deal then placed in the second vault. The base model and pilot members produced answers to them; no model was trained on them. The move was not recorded at the time and was found by the Round 2 blind-line audit. The main Round 1 run and its replications are not affected.

## Round 2 smoke census, 25 November 2026

The first version of the Round 2 de-duplication missed reformatted and near copies. The smoke census, which sampled the base model on 100 items per source, therefore produced 20 answers on two training items that are exact copies of vault items. The audit that found this estimated that 135 training or selection items had a copy in the vault. No member had been trained at that point. The de-duplication is being corrected before the full census; the final count of exposed vault items will be added here.
