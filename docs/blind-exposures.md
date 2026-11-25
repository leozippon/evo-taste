# Blind exposures

Model outputs that exist on vault items, or on copies of them, outside a recorded opening. The [ledger](blind-ledger.md) records openings; this file records everything else. At the opening of the vault every result is reported with and without the exposed items.

The exposed items are listed, as hashes of their identifiers only, in `configs/vault-exposure.yaml`: 88 of the vault's roughly 17,400 real-data items (77 from MMLU-Pro, 8 from the MATH test split, 2 from LogiQA, 1 from ARC). No model has been trained on any of them or on a copy of them.

## Round 1 pilot runs, before 22 November 2026

Three early runs were prepared before MMLU-Pro was re-dealt on 22 November. They used MMLU-Pro questions as evaluation items that the re-deal then placed in the second vault: about 60 vault questions were answered by the base model and by pilot members, directly or as a near-copy. The move was not recorded at the time and was found by the Round 2 blind-line audit.

## Round 1 evaluations of the first vault, 23 and 24 November 2026

The first vault, opened by protocol, contains near-copies of 22 questions of the second vault (15 from MMLU-Pro, 7 from the MATH test split). Every Round 1 member and baseline answered those near-copies when the first vault was evaluated.

## Round 2 smoke census, 25 November 2026

The first version of the Round 2 de-duplication missed reformatted and near copies; the blind-line audit estimated that 135 training or selection items had a copy in the vault. The smoke census, which sampled the base model on 100 items per source, therefore produced answers on four items that are copies of vault items (one from ARC, two from LogiQA, one from the MATH test split), and a repeat of the smoke on an intermediate file sampled the ARC item again. The de-duplication was rebuilt before the full census; by an independent measure three non-blind items with a vault copy remain.
