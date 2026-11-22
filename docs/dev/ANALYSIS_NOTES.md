# Analysis notes

Internal notes for the analysis tools in `evotaste/analysis/`: commands, outputs, the style features and the exact definition of every statistic. The tools read the files described in `docs/dev/ENGINE_NOTES.md` ("On-disk layout", "Record schemas") through one loader (`rundata.py`), never read `runs/<name>/eval/` (blind evaluations live there), need no GPU and work on runs in progress: slots without a lineage yet, lineages still seeding, histories being appended, probes not yet written.

## What counts as data

Records after a lineage's last checkpoint (`state.json` `step`) are uncommitted: a worker that resumes deletes and redoes them, and a crash can leave them behind. The loader applies one rule to them.

- Amplification history: one line per step (a later line for the same step replaces an earlier one, so no step appears twice), in step order. Lines of steps at or after the checkpoint are kept only while the lineage is `active` and its worker runs (work in progress); for a `done` or `culled` lineage, or without a running worker, they are dropped.
- Probe files: only steps up to the checkpoint count, and none for a lineage still `new` (a probe file without its checkpoint comes from a kill between the two).
- A worker runs when its flock on `runs/<name>/locks/<slot>.lock` appears in `/proc/locks` (matched by device and inode); the tools never take a lock. The flag applies to the slot's current lineage (status `new` or `active`).
- Probe answers of two models are paired only when they have the same uid and the identical question text; unpaired answers are dropped (in `runs/pilot0` the questions of `polynomial_multiplication` differ between processes).

## Commands

Run from the repository root after `source scripts/env.sh`.

| Tool | Command | Output |
|---|---|---|
| Run report | `python -m evotaste.analysis.report runs/<name> [--families]` | terminal report; CSV in `results/<name>/report/` |
| Style diagnostics | `python -m evotaste.analysis.style runs/<name> [--members a,b] [--arm plain\|scaffold\|perturb] [--suite train\|selection\|foundation\|all]` | terminal summary; CSV/JSON in `results/<name>/style/`, or `style/<subset>/` for a subset |
| Tests | `python -m pytest tests/test_analysis.py` | |

Both tools accept `--out DIR`. The style tool needs the base model's probe responses (it standardises by them) and stops with a message until they exist. Its classifiers are limited to 8 BLAS threads because the host's CPUs are shared with running experiments.

## Run report

Rates and accuracies are printed in percent and stored as fractions. The terminal shows the run header, the seeding pools, one training row per lineage (replacements of a culled slot are separate rows, cull reasons are listed below the table) and the probe differences; `--families` adds the per-family table.

| CSV | One row per | Content |
|---|---|---|
| `base.csv` | run | base probe accuracy per suite, `n`, `parsed`, `truncated`, `completion_tokens` |
| `seeding.csv` | teacher | `correct`, `samples`, `prompts_with_correct`, `prompts`, `families_with_correct`, `families`, `tokens_correct` |
| `members.csv` | lineage | the training statistics below, settings, `status`, `worker`, `checkpoint`, `steps_done`, `steps_uncommitted`, `cull_reason`, `replaced_by` |
| `probes.csv` | lineage × probe step × suite | `accuracy`, `base_accuracy` (both on paired instances), `diff`, `se`, `instances`, `families`, `unpaired`, `unpaired_families`, and the lineage's `parsed`, `truncated`, `completion_tokens` over the whole probe |
| `families.csv` | train family × lineage | `base_accuracy`, `base_n`, `reward_first`, `reward_last` |

Definitions, with n the number of amplification steps done:

- Steps done (`steps`): amplification steps in the history as loaded above; `ckpt` is the last checkpoint, so steps done minus `ckpt` are uncommitted work in progress of a running worker (`wk` on). Status `pending`: the slot has no `state.json` yet; `new`: seeding not finished. Seeding and amplification learning rates are the lineage's effective settings (run defaults with its overrides).
- `rew0`, `rew1`: mean of the per-step `reward` over the first and the last min(10, n) steps (the windows overlap when n < 20). `tok0`, `tok1`: the same for the per-step mean `completion_tokens`. `pars`, `trunc`: mean per-step `parsed`, `truncated` over the last min(10, n) steps.
- Slope: least-squares slope of the per-step reward on the step index over all steps done, times 100 (reward per 100 steps), with the classical standard error √(RSS / (n − 2) / Σ(t − t̄)²) × 100. It ignores the autocorrelation of successive steps and therefore understates the uncertainty of a drifting curve.
- `groups`: mean over steps of `groups` (prompts whose rollouts had mixed rewards and entered the update).
- `chk`: maximum over steps of `check.token_mean_abs` (the engine stops above 0.04).
- `s/step`: median interval between the timestamps (whole seconds) of consecutive amplification steps; it includes the shared rollout generation and the other members' updates in the same worker, and the median ignores probe pauses and restarts.
- Probe accuracy: per suite, the mean over families of the family's accuracy (the engine's macro definition), from the committed probe response files, for the lineage and for the base model on the instances paired with it (same uid and question; the header's base accuracy is over all base answers, and a footnote lists dropped items). The difference member − base is computed on these paired instances; its standard error is (1/F) √(Σ_f s_f² / n_f), where s_f² is the sample variance (ddof 1) of the per-instance differences r_member − r_base over the n_f instances of family f and F the number of families. It treats instances as independent; both rewards are single samples, so it contains the sampling noise of both models. It is undefined (`?`) when a family has a single instance. The terminal shows at most four probe steps (the first and the last three).
- Seeding pools: correct samples, prompts with at least one correct sample, families of the pool with at least one correct sample, and the mean completion tokens (length of `completion_ids`, end-of-turn included) of the correct samples.
- Families: base probe accuracy per train family (n = `train_per_family`), and per lineage the mean over its first and last min(10, n) steps of the family's per-step mean rollout reward (`reward_by_family`), over the steps in which the family appeared.

## Style diagnostics

Inputs are the committed probe responses of every lineage at every probe step it has (one sample per probe prompt), the base model's probe responses, and the texts of the seeding pools (`seeding/<teacher>.jsonl.gz`: correct, stopped base-model samples under the teacher prompt, on train-family prompts). `--members` and `--arm` restrict every analysis to a subset of lineages; the bias of a lineage is its scaffold, or its arm (`plain`, `perturb`) without one.

At each probe step, only the prompts that every lineage present and the base model answered with the same uid and the identical question text are kept, for every analysis below (profile, sharpness, classifiers); the dropped uids per step are printed and stored in `summary.json`. Otherwise a question that differs between worker processes would let content identify the lineages of one worker.

### Features

Every feature is computed from the completion text alone (plus the token count). The body is the text with every answer span removed (`<answer>…</answer>`, an unclosed `<answer>` running to the end, and stray closing tags); body lines are its non-empty lines, words its letter-only words (`[^\W\d_]+`), body characters its non-whitespace characters. Lexicon patterns match case-insensitively between word boundaries; the exact regular expressions are `LEXICONS` in `style.py`, and `FEATURES` there is the single source of this list.

| Feature | Definition |
|---|---|
| `tokens` | completion tokens (sampled ids, end-of-turn included) |
| `chars` | characters of the completion |
| `direct` | 1 if at most 5 words precede the first answer tag (the whole text if there is none) |
| `pre_answer` | fraction of the characters that precede the first answer tag (1 if there is none) |
| `after_answer` | 1 if more than 5 words follow the last closing answer tag |
| `lines` | non-empty body lines |
| `line_words` | words per non-empty body line |
| `numbered` | fraction of body lines opening with a step number (`1.` `1)` `(1)` `Step 1`, optionally after `#` or `**`) |
| `bullets` | fraction of body lines opening with a bullet (`-` `*` `•` `+` and a space) |
| `headers` | fraction of body lines that are markdown headers (`#`) or wholly bold |
| `bold` | bold spans (`**…**`) per 100 body characters |
| `math` | LaTeX markup (`\command`, `\(` `\)` `\[` `\]`, runs of `$`) per 100 body characters |
| `equals` | `=` signs per 100 body characters |
| `code` | 1 if the body contains a code fence |
| `table` | fraction of body lines containing at least two `\|` |
| `digits` | fraction of body characters that are digits |
| `verify` | verification markers per 100 body words: check, verify, confirm, plug (in), substitute back, validate, satisfy, sanity |
| `cases` | case analysis: case, option, candidate, try, possibility, possible, alternatively, otherwise, either, suppose, assume |
| `plan` | planning: plan, first, next, then, step, finally, approach, strategy, goal |
| `formal` | formalisation: let (not let's, let me, let us), define, denote, equation, variable, unknown, formula, represent |
| `correct` | self-correction: wait, actually, mistake, oops, hmm, however, incorrect, wrong, re-check / reconsider / re-evaluate / recalculate / rethink, on second thought |
| `conclude` | conclusion: therefore, thus, hence, consequently, in conclusion, final answer, (so) the answer is |
| `restate` | restating the problem: we are given / asked, given that, the problem / question / task asks / states / says / is / gives / requires, we need / have / want to, our goal is |

Lexicon features are matches per 100 body words; a feature whose denominator is zero is 0.

### Profile

The mean of every feature per source (base, pool, lineage), probe step and suite, and over all probe suites (`all`). The terminal shows one suite (`--suite`, default `train`, the suite comparable with the pools): the base model, then each seeding pool followed by the lineages seeded from it (perturbation lineages under `plain`) at their first and last probe step, for a compact subset of features. Pool means are over correct samples only, whereas probe means include wrong answers.

### Sharpness and divergence

For each probe step, each suite and all suites together, and each feature: the M ≥ 2 lineages with a probe at that step answered the same P kept prompts exactly once (the tool stops otherwise). Each value is divided by the base model's standard deviation of the feature across the same prompts (ddof 1); features without base spread are not standardisable and are left out (NaN). With z_mp the standardised value of lineage m on prompt p, the two-way layout without replication z_mp = μ + a_m + b_p + e_mp gives

- MS_member = P Σ_m (z̄_m· − z̄)² / (M − 1) and MS_resid = Σ_mp (z_mp − z̄_m· − z̄_·p + z̄)² / ((M − 1)(P − 1));
- between = (MS_member − MS_resid) / P, the lineage variance component (unbiased, not truncated at zero, so it can be negative when there is no lineage effect);
- within = MS_resid, the residual component: the lineage × prompt interaction plus sampling noise, i.e. how inconsistent a lineage is with itself once prompt difficulty is removed;
- ratio = between / within, and the F test of no lineage effect, F = MS_member / MS_resid on (M − 1, (M − 1)(P − 1)) degrees of freedom.

The `mean` row averages between and within over the standardisable features, reports the ratio of the two means and counts the features with p < 0.01. Prompt effects are removed by construction, so task difficulty and content do not enter `between`. In a run in progress the lineage set can differ between steps (column `members`).

### Identifiability

Classifier text is the completion with every answer span removed and every run of digits replaced by `0`; no other masking is done. The classifier is a TF-IDF union of word 1–2-grams (punctuation runs count as tokens) and character 2–4-grams within word boundaries (both `min_df` 2, sublinear tf) followed by multinomial logistic regression (C = 1); nothing is tuned.

At each probe step with M ≥ 2 lineages: trained on their responses to the `train` probe prompts, it predicts their responses to the `selection` and `foundation` prompts (accuracy per suite and pooled); the reverse direction trains on selection and foundation and predicts train. Across steps: trained on train-suite responses at the first probe step, it predicts selection and foundation responses at the last probe step, for the lineages present at both. Chance is 1/M (every lineage answers every prompt, so the classes are balanced). The 95% interval is a percentile bootstrap (1000 draws) over test prompts, resampling all M responses to a prompt together; the p-value is the one-sided exact binomial test against chance, which treats responses as independent and is therefore anti-conservative, so the interval is the one to read.

The confusion matrix aggregates the forward (train → selection + foundation) predictions by bias: rows are the true lineage's bias, columns the predicted lineage's bias, so confusions between two lineages of one scaffold land on the diagonal; the terminal shows row percentages at the last step. The top n-grams are the five largest coefficients per lineage of the forward model at each step (`w__` word n-gram, `c__` character n-gram).

### Bias recoverability

At each probe step, the classes are the biases with at least two lineages present; it needs two such classes, otherwise the row says it is not possible and lists the lineages per bias. With n_b lineages of bias b and K = max n_b folds, fold k holds out lineage k mod n_b of every class, trains on the other lineages of these classes (labels: bias; balanced class weights) and predicts the held-out lineages' responses on all probe suites. Accuracy is pooled over folds (a lineage of a class smaller than K is held out more than once); chance is 1/(number of classes), since each fold tests one lineage per class; the interval is the bootstrap over (fold, prompt) pairs and the p-value the binomial test as above. Both reflect prompt sampling only, not the choice of the few lineages per class, which dominates the uncertainty here; the accuracy per held-out lineage (mean over the folds holding it out) is therefore printed and stored (`per_lineage`). Lineages of biases with a single lineage are excluded and listed.

### Outputs

| File | Content |
|---|---|
| `features.csv.gz` | one row per response (base, pool samples, kept lineage probe answers): source, lineage, bias, teacher, step, suite, uid, family, reward and every feature |
| `profile.csv` | feature means per source, lineage, step and suite (`all` over probe suites; pools have suite `seeding`) |
| `sharpness.csv` | step, suite, feature (and `mean`), members, prompts, between, within, ratio, F, p |
| `identifiability.csv` | train step and suite, test step and suite, members, n, accuracy, chance, ci_lo, ci_hi, p_binom |
| `confusion.csv` | step, true bias, predicted bias, count (forward direction) |
| `ngrams.csv` | step, lineage, rank, n-gram, coefficient |
| `scaffold.csv` | step, classes, excluded, folds, per_lineage, n, accuracy, chance, ci_lo, ci_hi, p_binom, or a note |
| `summary.json` | lineages and biases, feature definitions, dropped prompts per step, the `mean` sharpness rows, identifiability and recoverability rows |

## Known limitations

- Classifier text is what remains outside the answer tags; for a lineage that answers directly (the terse member of `runs/pilot0` writes only an answer span) it is nearly empty, so such lineages are identified by the absence of text and absorb other short direct answers in the confusion matrix.
- Answers restated outside the answer tags (`\boxed{…}`, "the answer is X") lose only their digits before classification; a non-numeric answer written outside the tags stays in the text. All lineages answer the same prompts, so such content can identify a lineage only through lineage-specific answers.
- Probe answers are single samples per prompt and step, so `within` mixes a lineage's inconsistency with sampling noise, and probe differences carry the sampling noise of both models.
- The run report's per-family rollout rewards weight every step equally, not by the family's number of rollouts in the step.
