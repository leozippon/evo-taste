# Blind-line audit of the Round 2 data layer (2026-11-25)

Internal. Independent audit of the separation between train, selection and the vault `blind2` in the Round 2 data layer, before the full census.

**Audited commit:** `98366d3` (the last commit touching the scope; HEAD moved to `32df36f` during the audit with commits outside the scope). The uncommitted follow-up batch in the working tree (parts cache, census caps, LeetCode from both splits, new instruction wording) was read but not audited as code; see the suggestions.

**Method.** All computations ran on a scratch copy of the commit outside the repository (`git archive 98366d3`), with independent scripts. Blind items were loaded only to compute counts, hashes and similarities; no model was run. Blind items appear in this report only as a source and a count; every uid named here is a train or selection item. The scratch copy, which held blind texts, was deleted at the end.

## Verdicts

| # | Property | Verdict |
|---|---|---|
| 1 | One part per item, deterministic | Holds |
| 2 | Round 1's parts frozen | Holds by id; defect in content (D1, D2): open copies of sealed Round 1 MMLU-Pro questions survive in train and selection |
| 3 | No overlap across parts | Defect (D1, D2) |
| 4 | No path delivers a blind item without the receipt; nothing lists blind items | Holds for every code path and for every file the Round 2 build writes; but copies of vault items are delivered under open uids (D1, D2), and Round 1 pilot outputs on 57 of today's vault items lie in `runs/` and `results/` (R1) |
| 5 | Smoke census has no blind uid | Holds by uid; defect in content: 20 base-model samples on two exact copies of vault items |
| 6 | Seeded unit draw reproducible, not hand-edited | Holds |
| 7 | Training stream and probe set | Holds |
| 8 | Strata table pinned | Holds |

## Evidence per property

**1. One part, deterministic.** All 129,177 candidates get exactly one provisional part (`external.py:96-112`); uids are unique; de-duplication and filters only remove. My rebuild reproduced every train and selection hash in `configs/suites-r2.yaml`. The only load-dependent step is the reference check: a full regeneration of the suites file at a load average near 100 reproduced it exactly except one KodCode train item (`kodcode/Filter_1152_I`, rejected in the committed file, accepted this time). That moves an item between kept and dropped, never between parts, and loading uses the stored verdicts, so runs are stable (the vault's re-check at its opening is a suggestion below).

**2. Round 1 frozen.** Round 1's own `part_of`, executed from tag `round-1` on Round 1's own row loading, agrees with Round 2's provisional part on every item: GSM8K test 662 blind2 / 657 blind1, MATH-500 251 / 249, MMLU-Pro 3,919 blind2 / 4,081 blind1 / 4,027 foundation, GSM8K train all foundation; 0 mismatches. All 500 MATH-500 problems are found in `math_test`, both after whitespace normalisation and after alphanumeric-only normalisation. No Round 1 blind2 item is in train or selection; blind2 items leave the vault only by a filter or a same-part copy (MATH-500: 15 free-text references; MMLU-Pro: 9 same-part copies, 1 reference), never to another part.

Under other ids: no MATH-500 blind2 problem has an exact or alphanumeric-exact copy in train or selection, and the DeepScaleR copies of MATH-500 blind2 problems are all removed. One sibling survives although the stated rule removes it: `math_train/counting_and_probability/759` (stem Jaccard 0.56; same numbers, three words differ, a different question). For the sealed Round 1 MMLU-Pro questions, open items with the same question (identical or near-identical stem, different option list) survive: train `mmlu_pro/3594` (Jaccard 0.80), `mmlu_pro/2830` (0.69), `mmlu_pro/12139`, `mmlu_pro/11469` (identical stems up to punctuation, Jaccard 0.46 and 0.39); selection `mmlu_pro/4451` (0.96), `mmlu_pro/11551` (0.89), `mmlu_pro/2906`, `mmlu_pro/9549` (identical stems up to punctuation); and `theoremqa/780` (selection, 0.78, a rewrite with other numbers). These duplicates already straddled Round 1's parts, whose key was whitespace-normalised text; Round 2's de-duplication is meant to remove the open copies and does not (D1, D2).

**3. Overlap across parts.** Kept items only (train 79,264, selection 16,457, blind2 17,456 before the vault's reference check).

| Measure | Open vs blind | Train vs selection |
|---|---|---|
| Identical problem after alphanumeric-only normalisation, across parts | 11 train and 2 selection items | 9 train items |
| Stem Jaccard >= 0.5 (the rule's own measure), found with a correct MinHash | 89 pairs: 78 train items (deepscaler 37, math_train 34, mmlu_pro 6, arc 1) and 23 selection items (mmlu_pro 7, math_test 7, logiqa 6, deepscaler, theoremqa, arc 1 each) | 73 pairs, 70 train items |
| Independent: nearest neighbour by character 5-gram TF-IDF cosine, then edit similarity >= 0.9 on alphanumeric text with an identical number sequence (stems of 30+ characters) | 135 items, 0.14% (deepscaler train 62, mmlu_pro train 22 and selection 19, math_train 11, math_test selection 8, others 13); only 27 of them have the rule's Jaccard >= 0.5 | 120 train items, 0.15%; only 21 with Jaccard >= 0.5 |
| Distinct vault items with an open copy by any of the above | 185 of 17,456: math_test 71 of 2,382 (3.0%), deepscaler 67 of 1,383 (4.8%), mmlu_pro 36 of 3,910, logiqa 9, aqua 1, arc 1 | |

In a spot check of the independent criterion, most pairs are the same problem in other LaTeX or punctuation (`deepscaler/4094` writes `\( x \)` where its vault twin writes `$x$`; rule Jaccard 0.01); a minority are siblings (true/false, triangle/quadrilateral) or choice items that share only a generic stem. Cosine alone is no copy signal for ProofWriter, whose template vocabulary gives cosine >= 0.8 between different theories (no ProofWriter pair passes the edit criterion). Train items near selection items: `deepscaler/3319` and `deepscaler/4677` are reformatted copies of `math_test` selection items (Jaccard 0.13 and 0.24); `math_train/geometry/350` equals `math_test/precalculus/332` (`Compute $\tan 225^\circ$.`).

**4. Consumers and files.** `Streams.train_batch` and `seed_batch` read `Suites.items(s, "train")`; the probe set and `fixed_tasks(..., "selection")` read selection; `fixed_tasks(..., "blind2")`, `blind_suite` and `external.items(..., "blind2")` call `check_receipt`; `_open_parts` drops the vault (`external.py:338`); the census reads only `Suites.items` and `probe_set` and refuses jobs outside the open parts (`census.py:187-191`); procedural blind2 families are reachable only through `blind_suite` (`tasks.py:517-521`). Exercised: all four vault calls without a receipt raise `PermissionError`; 12,801 stream prompts (200 steps), 3,840 seeding prompts, 357 probe and 502 evaluation tasks contain no vault item.

Files, searched for vault uids and for the 10-word opening of any vault question that opens no open item: the git-tracked files at HEAD and the patches of every commit from `6663619` on contain neither; `logs/` contains neither; the suites file lists the vault by count and hash only. The census outputs contain no vault uid, and one generic phrase in a completion of `math_train/geometry/855`, which is not near any vault item. The working tree's new parts cache (`results/cache/parts/`) holds only train and selection and no vault uid; the vault openings found in its texts are template sentences (ProofWriter, KodCode) and the copies of D1 and D2. `runs/main1`, `main1-tp`, `main1-tp-rep` and `toycheck` contain no vault uid; Round 1's pilot runs do (R1).

**5. Smoke census.** 21,524 records: 0 vault uids; all 2,100 procedural records come from selection families; `strata.tsv` has 900 rows, all train. But two open items sampled there are copies of vault items: `deepscaler/4146` (train; identical to a `math_test` vault item up to `$$` delimiters; 8 samples in train, 8 in channels) and `arc/LEAP_2000_8_2` (selection; edit similarity 0.993 and the same answer as a vault ARC item; 4 samples). Neighbours also sampled: `theoremqa/780` (a rewrite of a sealed MMLU-Pro question) and `logiqa/test/449` (shares its passage with vault LogiQA items), 4 samples each.

**6. Unit draw.** My re-implementation of the stated draw (`stable_seed(20261125, "units", domain)`, shuffle, alternate from `randrange(2)`) equals the file's 27 units (math 2/2, code 2/1, logic 10/10 between selection and blind2). A full regeneration from the commit into the scratch copy reproduced the file byte for byte except the one KodCode verdict under property 1: units, every vault count and hash, and every other hash are identical.

**7. Stream.** With a synthetic strata table over all train uids, two processes computing 200 batches of 64 in order and in shuffled order give the same digest. Only train items; per batch the domain counts are within 0.6 of 25.6 / 16 / 22.4; equal shares per source; strata 0.70 / 0.15 / 0.15 in every source; 15.0% of non-code prompts carry one of the six training wrappers, never a held-out one, with the direct channel only; code prompts keep the default. Probe: 336 selection items, 21 procedural selection instances, 60 under held-out wrappers; the evaluation set is disjoint from it. A strata table naming a non-train uid is refused.

**8. Strata pinning.** A config with amplification steps requires `strata: {path, sha256}` (`config.py:260-261`); `prepare` refuses a table with another digest and copies it into the run (`worker.py:151-155`); members load it with the pinned hash (`member.py:98`, `tasks.py:598-600`).

## Defects, by impact

**D1. The MinHash hash family barely permutes, so near copies are missed far more often than documented.** `external.py:152-155, 168`: `h(x) = (a·x + b) mod (2^61 − 1)` with `x` a 32-bit CRC and `a, b < 2^31`. Since `a·x + b < 3.9·p`, each `h` increases with `x` on at most four intervals, so its minimum is nearly always one of the few smallest shingle hashes and the 72 functions are strongly correlated (on `mmlu_pro/3594`, 13 distinct minimising shingles over 72 functions; for its sealed twin at Jaccard 0.80, 11 of 72 rows and 0 of 24 bands agree). Measured recall of the search over all candidates, against a correct MinHash (splitmix64, 40 bands of 3, exact Jaccard): 0.73 at Jaccard 0.5–0.6, 0.83 at 0.6–0.7, 0.91 at 0.7–0.8, 0.95 at 0.8–0.9, 1.00 above; ENV_NOTES (lines 62, 108) and `external.py:49-50` state 0.96 at 0.5 and at least 0.995 from 0.6. Impact: the 89 blind–train, 32 blind–selection and 73 selection–train pairs of the table above, all of which the stated rule removes.

Reproduction without any blind data (`source scripts/env.sh; python`):

```python
import numpy as np
from evotaste import external
rng = np.random.default_rng(0)
for J in (0.5, 0.6, 0.8):
    m = 60; c = round(2 * J * m / (1 + J)); found = 0
    for _ in range(500):
        u = rng.choice(2**32, 2 * m - c, replace=False).astype(np.uint64)
        found += bool(external.near_pairs([np.unique(u[:m]), np.unique(u[m - c:])], ["x", "y"], threshold=0.0))
    print(J, found / 500, 1 - (1 - J**3) ** 24)   # 0.66 / 0.78 / 0.93 against 0.96 / 0.997 / 1.0
```

Smallest fix: a mixing hash instead of the linear one, for example `h_k(x) = splitmix64(x ^ seed_k)` in numpy `uint64` (wrapping multiplication), seeds from `default_rng(SPLIT_SEED_R2)`; turn the snippet into a test (random shared elements, recall at Jaccard 0.6 of at least 0.95); regenerate the suites file. Round 1's parts are untouched; the vault's hash changes, which is harmless before its opening.

**D2. The exact-copy key and the shingle tokens are sensitive to formatting, so verbatim copies in other LaTeX or punctuation survive.** `external.py:68-69` normalises only case and whitespace, and the tokens of `external.py:133-142` include every punctuation mark and brace, so `\( x \)` against `$x$` or `x^{3}` against `x^3` changes most 5-grams of a short problem. Impact: 11 train and 2 selection items are identical to a vault item after alphanumeric-only normalisation (train `deepscaler/3905`, `4090`, `4094`, `4146`, `6784`, `6909` copy `math_test` vault items; `deepscaler/4457`, `10251`, `16056`, `24289`, `29591` copy DeepScaleR vault items; selection `math_test/algebra/1022`, `math_test/precalculus/320`), with rule Jaccard between 0.01 and 0.47; of the 135 open items with an independent-criterion copy in the vault, 108 are below the rule's threshold. This is how two copies of vault items reached the smoke census. Smallest fix: key exact copies on `re.sub(r"[^a-z0-9]", "", text.lower())`, and compute the shingles on the same string; character 8-grams of it reach Jaccard 0.5 for 122 of the 135 open-versus-vault and 108 of the 120 train-versus-selection copies, against 27 and 21 now (word 5-grams of alphanumeric tokens: 58 and 47). Its rate of removing siblings was not measured; an extra removal costs an item, never a leak.

**Required with D1 and D2, before the full census:** regenerate the suites file and rerun this overlap estimate; then delete or quarantine the smoke census (`results/census/census-r2-limit100/`), whose records include 20 samples on two vault copies, and record in the logbook or the blind ledger that the base model was sampled on prompts identical to one `math_test` and one ARC vault item. After the fix both vault items stay blind and their open copies are removed.

**R1. Round 1's pilots produced model outputs on 57 items of today's vault (outside the Round 2 code, found by the file search).** The probe responses and base-model records of `runs/scale-test` and `runs/pilot0` (55 distinct MMLU-Pro items, suite `foundation`, base model and every pilot member) and `runs/toy` (6), and the files derived from them (`results/scale-test/style/`, `results/pilot0/style/`, `results/toy/style/`, `results/toy/niches/selection/solve_rates.csv`), name 57 distinct MMLU-Pro items that are in `blind2` under Round 1's final split. The three runs were prepared on 2026-11-22 between 15:00 and 15:51, before commit `4f4336f` (16:36, "Split external benchmarks by question text instead of item id") re-dealt MMLU-Pro; that day's logbook records the change but not that items already evaluated as foundation moved into the vault. The main Round 1 runs are clean. Impact is small (evaluation only, no training on these items, about 1.5% of the vault's MMLU-Pro), but it contradicts the vault's premise. Smallest fix: a ledger or logbook entry now, and at the opening a report of MMLU-Pro with and without these items (their uids can be recomputed from those files at that time; they are not listed here).

## Suggestions

- `tasks.py:702` puts up to three stray uids in the error message; a stray uid can be a vault uid, so give the count only, as `census.py:189-191` does.
- `load_strata(path, sha256=None)` skips the check; config and `prepare` always pass a hash, but making the argument required removes the silent path.
- Near copies of choice items are judged on the stem alone, so unrelated questions with a generic stem ("Which of the following statements is false?") count as copies. This over-removes and never leaks; comparing stem and options would avoid it.
- The vault's reference checks rerun at the opening and depend on machine load (this audit's regeneration moved one KodCode verdict), so a differing verdict blocks the opening and forces a regeneration that also changes the open parts. Fixing vault membership before the reference check, and marking failing references unusable at the opening, would make the vault's hash load-independent.
- The batch in progress takes LeetCode from both splits into selection; KodCode contains problems derived from LeetCode, so rerun the train-versus-selection estimate after it lands.

## Accepted limitations (as documented)

- Paraphrases and translations are not detected.
- The receipt is a plain class, and `build` and `_vault_candidates` return vault items without one; the protection is the accessors' refusal, as in Round 1.
- The generator also checks the vault's references (verifier only, including running the reference solutions of vault code items in the sandbox). No model is involved, but this is more than the spec's "texts for de-duplication only", whose wording should say so.
