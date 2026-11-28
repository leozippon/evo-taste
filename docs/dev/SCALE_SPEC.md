# Build spec: the larger-model stage

Design: `docs/design/scale.md`. Ground rules and the engine contract of `docs/dev/ROUND2_SPEC.md` still apply (secrets, blind line, shared machine, no force-push, runtime output outside git). Work packages are added here as they are specified.

## Ground rules specific to this stage

- GPUs 2 and 7 only until told otherwise. Never touch processes this project did not start; other sessions run `vllm serve` on the same machine under the same account.
- Never kill by pattern with `pkill -f` / `pgrep -f` in a command line that contains the pattern: it matches the shell running it. Kill by explicit PID.
- Only items of the **train** part may be used in WP-M. Nothing from selection units or the blind vault.
- The account's CPU quota is often saturated; verification pools should use the existing pool settings, not more.
- `logs/`, `runs/`, `results/` are not in git and not synced to the editing mirror: create and read them through the shell.

## WP-M: method census (inference only)

Answers questions 1 and 2 of the design: is there variation of method among a model's own correct answers, can it produce another method on request, and are its correct attempts the typical ones. Two subject models, the same problems, the same judge.

### Entry point and layout

One module `evotaste/method.py` with a CLI (`python -m evotaste.method <stage> <config> [--shard i/n] [--limit N]`), one config `configs/method.yaml`, one analysis module `evotaste/analysis/method.py`. Reuse, do not copy: `evotaste.census` (item draw, prompt building, record shape, chunked resumable writing), `evotaste.sampler` (`Server`, `Sampler`, `Request`), `evotaste.tasks`, `evotaste.verify` (scoring through the existing pool), `evotaste.analysis.stats` (bootstrap). Output under `results/method/<name>/`.

The config names the subject models (label, model path or hub id, server settings), the judge model, the item draw, sample counts and seeds. Every stage is resumable by chunk and refuses to mix records produced under different configs (hash of the stage-relevant config in the output path or header, as census does).

Chat template: the prompt builder currently passes `enable_thinking=False`; keep that for subject models, and make sure it is valid for the 27B's template (inspect `chat_template.jinja` of the model; fail loudly if the flag is not honoured, i.e. if a thinking block appears in output).

### Stage `items`

Draw the item sample from the train part using the 1.7B census (`results/census/census-r2/strata.tsv` and records): plain wrapper only, stratified by domain and by the 1.7B stratum, seeded. Default targets: math 150 mastered / 450 sweet / 300 frontier; code 100 / 300 / 200; logic 50 / 150 / 100. Within a cell spread over sources in proportion to availability. Write `items.tsv` (uid, domain, source, 1.7B passes) and never redraw if it exists.

### Stage `attempts --model <label>`

Each subject model answers every item with the training prompt, 8 samples, sampling settings as in training (temperature 1, top-p 1), max tokens as in census. Store per sample: text, token count, finish reason, verifier outcome (reward, complied, truncated, error), and the sum and mean of the sampled tokens' log-probabilities (the sampler already returns them). Both models are run through this stage, the 1.7B included (the old census has no log-probabilities).

### Stage `alternatives --model <label>`

For every item with at least one correct attempt by that model: show the problem and the model's own first correct attempt, and ask for a solution by a genuinely different method that arrives at the answer in the same required format. 2 samples. The instruction text lives in the config; it must define method as the design does (technique / algorithm and data structure / line of argument, not wording, layout, order or level of detail) and must not hint at any particular method. Verify with the item's normal verifier.

### Stage `pairs`

Build the list of pairs to judge, seeded, and write `pairs.jsonl.gz` once:

- `spontaneous`: for each item and model with at least two correct attempts, up to 2 pairs of distinct correct attempts.
- `requested`: for each item and model with a correct alternative, the reference attempt paired with one correct alternative.
- `cross`: for items where both models have a correct attempt, one pair of one attempt from each.
- `repeat`: 200 of the spontaneous pairs duplicated with the order of the two solutions swapped (judge consistency).

Each pair carries an id, the item, the kind, and for each side the model label and record reference; the order of sides is randomised and the judge never sees labels.

### Stage `judge`

Serve the judge model and judge every pair at temperature 0. The prompt gives the problem and the two solutions (truncate each solution to a token budget from the config, default 1,500 tokens, keeping the start and the end) and asks for a JSON object: the approach of A in at most 15 words, the approach of B likewise, and `relation` in `same` / `variant` / `different`, with the three defined in the prompt (`same`: the same key idea and steps, differing only in wording, layout, order, detail or checking; `variant`: the same key idea carried out through materially different intermediate steps; `different`: a different key idea, technique, algorithm or line of argument). The judge may think before answering if its template supports thinking; record whether it did. Parse strictly; a reply that does not parse is retried once and then recorded as unparsed, never guessed.

Also write `reader-sample.jsonl`: 160 pairs stratified by kind, subject model, domain and judge verdict, with problem and both solutions in full and without verdicts or model labels, plus a separate key file. Independent readers will label it.

### Mechanical comparison for programs

For correct code attempts, extract a coarse construct profile from the AST (recursion, loop kinds, comprehensions, sorting, hashing containers, heap/deque/bisect/itertools/re/math use, string methods vs manual scanning, early return, nesting depth of loops) and report, per pair, a Jaccard distance. This is a cross-check of the judge on code, not a replacement.

### Analysis (`python -m evotaste.analysis.method results/method/<name> --out ...`)

Report as text plus a JSON of all figures, with item-level bootstrap intervals:

1. Accuracy of each model by domain and 1.7B stratum; the 27B's own strata (0, 1–6, 7–8 of 8) crossed with the 1.7B's; answer length; compliance; truncation.
2. Spontaneous method variation: share of pairs `different`, and `variant` or `different`, by model and domain; on matched items (both models have a spontaneous pair) and separately on each model's own sweet-spot items; the share of items with any `different` pair.
3. Method on request: of items with a reference, the share where an alternative is correct, and where it is correct and judged `different`, by model and domain.
4. Cross-model pairs: how often the two models' correct answers use different methods, against each model's within-model rate.
5. Judge quality: agreement on swapped repeats; unparsed rate; agreement with the mechanical distance on code (rank correlation); a loader for reader labels that reports agreement and Cohen's kappa when a labels file is given.
6. Typicality and correctness, per model: on items with both correct and wrong attempts, the within-item standardised difference in mean token log-probability between correct and wrong attempts; accuracy by within-item typicality tercile; the same after removing the within-item linear effect of answer length on mean log-probability. Also the share of each model's correct answers that fall in the least typical tercile.
7. Whether method variation and typicality are related: for spontaneous pairs, the absolute difference in mean log-probability by judge verdict.

### Tests

Unit tests on the deterministic parts only: the item draw (stratification, determinism, train part only — assert no selection or blind uid can appear), pair construction, strict judge parsing including malformed replies, the construct profile on a handful of programs, and the typicality statistics on a small synthetic table. No mocks of the model server.

### Acceptance

A smoke run on a real GPU with `--limit` (a few dozen items, both models, all stages, then the analysis) once a card is assigned, before the full run. Notes, commands and timings go to `docs/dev/SCALE_NOTES.md`.
