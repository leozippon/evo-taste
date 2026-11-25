# Environment notes

Internal current-state notes for the Round 2 environment: dataset sources and their normalisation, the parts and the vault, prompts and scoring, the training stream, probe and evaluation sets, and the census. The design is `docs/design/round-2.md`, the contract between work packages `docs/dev/ROUND2_SPEC.md`; verifiers and sandbox are described in `docs/dev/VERIFY_NOTES.md`, the engine in `docs/dev/ENGINE_NOTES.md`.

## Code and commands

| File | Role |
|---|---|
| `evotaste/sources.py` | one loader per source: rows of the cached dataset normalised to items |
| `evotaste/external.py` | parts, de-duplication, filters, and the guarded accessor `items(source, part, receipt, suites)` |
| `evotaste/tasks.py` | `Task`, `Score`, answer instructions, `prompt_ids`, `score_many`, procedural families, `Suites`, probe and evaluation sets, `Streams` |
| `scripts/make_suites_r2.py` | writes `configs/suites-r2.yaml`, the single source of the part assignment |
| `evotaste/census.py`, `configs/census.yaml` | the census of the base model |

All commands run from the repository root after `source scripts/env.sh`.

| Step | Command |
|---|---|
| Fill the dataset cache (once, through the mirror) | `python scripts/make_suites_r2.py --download` |
| Assign parts | `python scripts/make_suites_r2.py` (5 to 15 minutes depending on the machine's load, mostly the reference checks with 32 verifier processes) |
| Census, one part and shard per GPU | `CUDA_VISIBLE_DEVICES=<g> python -m evotaste.census run configs/census.yaml --part <part> --shard i/n` |
| Census summary and strata table | `python -m evotaste.census summarize configs/census.yaml` |
| Tests | `python -m pytest tests/test_tasks.py tests/test_external.py tests/test_census.py` (4 to 6 minutes; they build the parts) |

Everything after the download reads the local cache offline. Every process that loads items rebuilds the parts from the cache (about one to two minutes on the loaded machine: loading 20 s, copy detection 30 s, tokenisation 10 s) and checks them against the suites file.

## Sources

Each loader returns the candidate items of a source in a fixed order; an item carries its problem as the model sees it (options rendered as `A. text` lines, no answer instruction), a gold reference in the verifier's format, the stem used for copy detection (the problem without options or fixed wording), and the text whose hash decides its part. Samples are drawn with seed 20261125.

| Source | Dataset and normalisation | Kind | Role |
|---|---|---|---|
| `gsm8k_train`, `gsm8k_test` | `openai/gsm8k`; gold after `####`, commas removed | open | train; held out (Round 1 parts) |
| `math_train`, `math_test` | `EleutherAI/hendrycks_math`, 7 configurations; gold = content of the last `\boxed{}` of the solution (2 train items have none) | open | train; held out (MATH-500 items keep their Round 1 parts) |
| `deepscaler` | `agentica-org/DeepScaleR-Preview-Dataset` | open | train (no test split) |
| `aqua` | `deepmind/aqua_rat` raw: 10,000 of train, all of validation and test held out; options `A)21` split into letter and text | choice | train |
| `kodcode` | `KodCode/KodCode-Light-RL-10K`; the instruction names the declaration from `test_info`; pytest harness | code | train (no test split) |
| `logiqa` | `lucasmccabe/logiqa` (parquet branch); context, query, four options; test held out, validation unused | choice | train |
| `proofwriter` | `tasksource/proofwriter`, configurations depth-0 to depth-5 (open world): 1,000 questions per depth 0–5 from train, 200 per depth from test, at most one question per theory; test parts decided by theory | verdict True/False/Unknown | train |
| `arc` | `allenai/ai2_arc` ARC-Challenge; labels by position (some keys are digits); test held out | choice | train |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro`, up to ten options; the five items that repeat MATH-500 problems dropped since Round 1 | choice | train (Round 1 parts) |
| `theoremqa` | `TIGER-Lab/TheoremQA`; 53 items with a picture and 18 `option` answers dropped; `bool` answers become True/False verdicts | open, verdict | unseen |
| `sat_math`, `lsat_lr`, `lsat_ar` | `hails/agieval-*`; the question is the query before "Answer Choices:" without "Q:", options from `choices`, gold from `gold` | choice | unseen |
| `svamp` | `ChilleD/SVAMP` train and test, `question_concat` | open | unseen |
| `omni_rest` | `KbsdJames/Omni-MATH`; items with a copy in DeepScaleR are removed by de-duplication | open | unseen |
| `humaneval_plus` | `evalplus/humanevalplus`; the prompt in a code block; `check` harness; like the other evaluation code sources, 30 s and 2 GiB per test run (training items keep the verifier's 10 s and 1 GiB) | code | unseen |
| `mbpp_plus` | `evalplus/mbppplus`; the prompt and the first original assertion; the dataset's test calls the function by name (`assert` harness, test imports prepended) | code | unseen |
| `leetcode` | `newfacade/LeetCodeDataset` test split (problems from 2024-08 to 2025-03, no assertion expecting None); description and starter code; tests end with `check(Solution().method)`; the dataset's import and helper prelude is carried as `prelude` | code | unseen |
| `folio` | `tasksource/folio` train and validation | verdict True/False/Uncertain | unseen |
| `fld` | `hitachi-nlp/FLD.v2` test, 1,000 sampled; facts one per line | verdict PROVED/DISPROVED/UNKNOWN | unseen |
| `bbh_*` (12 units) | `lukaemon/bbh` test; logical deduction and tracking shuffled objects each one unit of three sizes; options parsed from the input; boolean expressions, formal fallacies, navigate and web of lies are verdicts; multistep arithmetic and object counting open; dyck languages and word sorting compared exactly (kind `procedural`) | as fits | unseen |
| `medmcqa` | `openlifescienceai/medmcqa` validation, 1,000 sampled | choice | unseen |
| `scienceqa` | `derek-thomas/ScienceQA` (parquet branch) test, text-only, 1,000 sampled; the hint is the context | choice | unseen |
| `openbookqa`, `qasc` | `allenai/openbookqa` test, `allenai/qasc` validation (without its facts) | choice | unseen |
| procedural | Round 1's 21 selection families (format probe) and 20 blind2 families (sealed), from `configs/suites.yaml` | procedural | selection; blind2 |

## Parts

Every item belongs to exactly one part: `train`, `selection` or the vault `blind2` (Round 1's second vault, extended). The assignment, implemented in `external.py`, has three steps.

1. **Provisional part.** Round 1's own function, copied verbatim, decides GSM8K, MATH-500 and MMLU-Pro: foundation goes to train, blind1 (spent) to selection, blind2 stays in blind2. Other held-out items of training sources are halved between selection and blind2 by a seeded hash of their text (of the theory for ProofWriter). DeepScaleR and KodCode, which have no test split, give 5% each to selection and blind2, drawn only from items with no copy in another source, so that DeepScaleR's copies of MATH-train problems do not pull those problems out of training. An unseen unit goes whole to the part a seeded draw gave it: within each domain the units are shuffled and alternate between the parts, so the halves differ by at most one unit. No units were judged near-identical beyond the BBH tasks already merged into one unit each.
2. **De-duplication** over all sources at once. Exact copies have the same normalised problem; near copies have a Jaccard similarity of at least 0.5 between the word 5-gram shingles of their stems (found by MinHash, 24 bands of 3 rows, then computed exactly). Pairs between 0.5 and 0.7 are mostly reformatted copies and rewrites with changed numbers. Near copies count between sources and between parts, not within one source and part, where items sharing a context (questions on one theory, conclusions from one set of premises) are distinct items. Of two copies the one in the higher part survives (blind2 over selection over train); within a part the copy of the smaller source survives, which keeps the roughly 600 TheoremQA problems that recur in MMLU-Pro in TheoremQA. Omni-MATH items with a copy in DeepScaleR are removed whatever their parts.
3. **Filters**: no gold; free-text gold (an open reference with `\text` or `\mbox`, longer than 40 characters, or with a bare word of three or more letters such as `9 and -7` or `Saturday`, which the verifier would read as a product of symbols matching any anagram); a prompt over 3,500 tokens (none at present); a reference that does not verify against itself in every wrapper of its kind (`gold_check`; for code, the reference solution failing its own tests). A reference check that times out or loses its verifier process is repeated up to three times; one that always times out counts as failed, and a verifier process that keeps dying stops the build (an infrastructure failure is never a verdict).

Removal never depends on a verdict: de-duplication runs before the filters, on every candidate. `configs/suites-r2.yaml` records the unit draw, which sources have items in which part, the procedural families, per source and part the item count and the SHA-256 of the sorted uid list, the counts of every drop reason, and the uids of train and selection items whose reference failed. Loading recomputes the parts and refuses any mismatch with the recorded hashes. Vault items are listed nowhere: their reference checks are rerun when the vault is opened, and the result must reproduce the recorded hash. The generator and the loaders read vault texts only to hash and de-duplicate them; after loading, a process keeps only train and selection items.

`external.items(source, "blind2", receipt)` and `tasks.blind_suite` require the receipt that only `evaluate.blind_guard` issues, as in Round 1; `fixed_tasks(..., "blind2", receipt=...)` passes it on. `Suites` never reads the vault's units.

## Prompts and scoring

The system prompt is the single sentence `tasks.SYSTEM_PROMPT`; a seeding context is appended to it after a blank line (`prompt_ids(..., context=text)`). The question is the problem, a blank line, and the answer instruction of its kind and wrapper, built from the wrapper's instruction sentence (`evotaste.verify.WRAPPERS`):

| Kind | Default wrapper and instruction |
|---|---|
| open | box, plus the offer of the program channel ("or end with one Python program in a fenced block whose printed output is the final answer; whichever comes last counts"); channels `("direct", "program")` |
| choice | "Answer with the letter of the correct option." and box |
| verdict | "Answer with exactly one of: …" (the item's labels) and box |
| code | code block, "It must define …" (the declaration, the function name, or the starter code's class `Solution`) |
| procedural | box |

`rewrap(task, wrapper, channels)` puts the same problem under another wrapper (a guardrail variant: the variant's instruction, direct channel only) or forces one channel (`PROGRAM_ONLY` for the program channel). `score_many` maps a task to the verifier's job (kind, text, gold, wrapper, channels) and sets the reward to 0 unless the completion stopped. Procedural tasks are scored here: reasoning-gym families by their own scorer in eight persistent scorer processes (Round 1's machinery, with its time bounds), exact-match tasks inline; the reward needs the wrapper asked for, `correct_any` accepts the content of any answer wrapper, which gives the format probe its strict and lenient scores.

## Training stream and seeding prompts

`Streams(suites, run_seed, strata=load_strata(path, sha256))`. The pools are the train-part items that the census stratified, one per source and stratum; a mastered or frontier pool with fewer than 50 items gives its share to the source's sweet pool, and a source with fewer than 50 sweet items is an error. A batch of P prompts is split first among the domains (math 2/5, code 1/4, logic 7/20), then within each domain among its source-stratum cells (equal shares per source; sweet 7/10, mastered 3/20, frontier 3/20), each split by largest remainder with the remainders carried from batch to batch and ties broken in a seeded order: every batch follows the domain shares to within one prompt, and over a run every cell gets its share to within two. Each pool is walked in a seeded permutation, without replacement until it is exhausted, then in a new one. Of the non-code prompts, 3/20 (cumulatively, to within one) carry a training-variant wrapper chosen uniformly; which prompts is seeded per step. Batch t is a pure function of (run seed, t, P, the suites file, the strata table), whatever was computed before, so every member and every resumed worker gets the same prompts. The strata table must not name items outside the train part.

`seed_batch(i, n, domains)` draws from the whole train part with the same domain and source shares (restricted to `domains`), default wrappers, its own seeded walk per domain set.

## Probe and evaluation sets

Both are independent of the run seed. The probe set takes the first `round(300 / (sources + 1))` items (13 at present) of the seeded order of every selection source, one instance of each procedural selection family (stream `probe`), and then 60 further items of the non-code sources, round robin, under the held-out wrappers in turn. `fixed_tasks(suites, part, per_source, per_family)` takes, per source, the last `per_source` items of the part's order that are not probe items, and `per_family` instances of each procedural family from stream `eval`; a procedural draw whose generator raises or whose gold its scorer rejects is replaced by the position's next draw, as in Round 1.

## Census

`configs/census.yaml` fixes the model, the suites and contexts files, 4,096 tokens, chunks of 256 jobs, the vLLM settings (KV 12G, `gpu_memory_utilization` 0.42) and the four parts: train (8 samples on at most 8,000 items per training source, the first of the part's order; all of MATH-train and KodCode), selection (4 samples on every selection item and 25 instances per procedural family), channels (4 samples under each forced channel on 1,500 sweet-spot open items, equal shares of the open training sources; needs the finished train part), and contexts (2 samples under each of the twelve contexts on the probe set's plain items). Requests are seeded by (part, uid, variant).

A chunk is written atomically to `results/census/<name>/<part>/chunk-NNNNN.jsonl.gz` once all its samples are scored; a rerun skips finished chunks, and shard i of n takes chunks whose index is i mod n, so shards can run on several GPUs and be re-sharded. Two chunks are in flight at a time so that the GPU stays busy while one is scored. A process refuses to start on a card with less than `need_free_mib` free, and refuses to write a chunk in which more than five samples (or 1%) lost their verifier or scorer process: such a chunk measures the infrastructure, not the model. The channels part and the strata table refuse train records that are not exactly those of the current train part. Do not run the census, or the generator, while the verifier code in the working tree is being edited: verifier processes are fresh interpreters and pick up whatever code is on disk. `--limit N` (at most N items per source and family, output under `<name>-limitN`) is for smoke runs. Every record has the uid, source, kind, wrapper, channels, context, the text, the completion length, the finish reason and every `Score` field. `summarize` writes `strata.tsv` (header `uid passes samples`, sorted by uid; frontier 0, sweet 1–6, mastered 7–8 of 8) and `summary.json`: per source and part the pass-rate histogram and strata sizes (train), accuracy, tolerant accuracy, compliance, truncation, errors, response-length percentiles, channel shares and per-channel accuracy, per forced channel the same by source, per context the same by domain, and the throughput per process. A run config pins the table's SHA-256.

The census reads items only through `Suites.items` for train and selection and the probe set, never holds a receipt, and checks every job's uid against the open parts; a test asserts all three.

## Known limitations

- **LeetCode needs a prelude hook in the verifier.** Its problems assume the dataset's imports and helper classes (`List`, `ListNode`, `TreeNode`); the verifier saves the candidate program alone as `solution.py`, so 172 of the 228 references fail their own tests and are dropped, and the remaining 56 are those that need no prelude. Once the verifier prepends `gold["prelude"]` to the program, the suites file must be regenerated.
- **Copy detection is lexical.** Paraphrases, translations and rewrites that change more than about half of the 5-gram shingles are not found; MinHash finds a pair at Jaccard 0.5 with probability 0.96. A copy kept by de-duplication can afterwards fail a filter while its removed twin would have passed.
- Dyck-language targets containing `{` or `}` cannot be boxed, so 99 of 250 items are dropped; Omni-MATH loses 658 items to free-text references and keeps 212.
- MMLU-Pro counts as logic although it holds math and science questions; TheoremQA's `option` items are dropped rather than parsed.
- Every process rebuilds the parts (one to two minutes) and holds every source in memory while doing so, vault texts included, as Round 1 did; protection of the vault is the accessor's refusal, not memory isolation.
- Opening the vault reruns its reference checks; a verifier that changed since generation makes the hash check fail, and the suites file must then be regenerated before the vault can be read.
- `evotaste/analysis/format.py` (Round 1 analysis) still constructs Round 1 tasks and does not run against this module.
