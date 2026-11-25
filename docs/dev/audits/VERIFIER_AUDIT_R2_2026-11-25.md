# Verifier audit, Round 2 (2026-11-25)

Independent audit of `evotaste/verify/` at commit `d416aa6` and its tests, against `docs/dev/ROUND2_SPEC.md` (Contract, WP-V), `docs/design/round-2.md` (Environment) and `docs/dev/VERIFY_NOTES.md`. `score_many` was read only for how it turns a verdict into a score. Read-only: no repository file was changed except this report.

**Headline.** There is systematic false credit for one class of answers: an exact answer to a reference larger than about 10^40 (or smaller than 10^-40) counts as correct whenever it agrees to 40 significant digits, so off-by-one answers such as `2^{2014} - 1` for `2^{2014}` are credited. On real outputs this accounts for 14 of the 73 credited Omni-MATH answers. No sandbox escape was found. One cross-job channel was found: POSIX message queues survive from one job to the next.

## Method

- Code and data. A copy of `d416aa6` was exported with `git archive` to a scratch directory and run with the project's venv. It was needed because the working tree's uncommitted WP-D changes make `Suites.items` refuse the committed suites file. Only `train` and `selection` items were loaded, through `Suites.items`. Nothing in `blind2` was touched and no model was run.
- Real data: all 21,524 records of `results/census/census-r2-limit100`. The 18,624 non-procedural records were re-scored through `verify_many` with a 10-process pool. Each re-scored verdict was compared with the stored one, which was produced by older rules.
- Judging. Every cell was sampled with seed 20261125 (60 per cell, or all records when fewer), and each sampled response was judged by hand against its question and reference. Each sample was backed by a census-wide mechanical scan, reviewed by hand:
  - every credited open answer whose text differs from the reference;
  - every uncredited open answer that a lenient variant (`\ln`/`\log`, brackets, labels, numeric closeness, sympy printing, degrees) would accept;
  - every choice answer of the form letter plus text;
  - every code response with several Python blocks.
- Load. CPU time per job was measured by re-running every census code job, every program answer and every code reference with a `getrusage` hook added. A starvation experiment pinned a job to one core shared with 11 busy loops. The machine's load average was 100–130 throughout.
- Scratch scripts were kept outside the repository and are not part of this report.

## Rates

| Cell | Credited | False credit, sample | False credit, census-wide | Uncredited | Missed, sample | Missed, census-wide (reviewed) |
|---|---|---|---|---|---|---|
| open / direct | 3,909 | 0/60 (1 debatable: `\frac12\mathbf{e}` for scalar `\frac12`) | 14 (0.36%), all D1; 14/73 on omni_rest | 2,433 | 2/60 (3%) | ≥ 11 found (0.5%): `\ln` vs `\log` reference, `a=b=c=d` reference, tuple-equation reference, choice label inside an open answer, `120^\circ` for `2\pi/3`, decimals of exact references |
| open / program | 79 | 1/79 (all reviewed) | 1 (`0.9999999999999994` for `3^{16}/(3^{16}+1)`, truncated response) | 324 (199 printed nothing, 21 errors) | 2/60 | 10 labelled outputs (`Louise needs to bake 10 more cakes.`), 10% of uncredited programs that printed something |
| choice | 4,539 | 0/60 | none found | 3,221 | 3/60 (5%) | 85 letter + LaTeX-formatted text answers with the right letter (D3): 1.1% of choice responses, 32% of right-letter letter+text answers |
| verdict | 1,346 | 0/60 | none found | 634 | 0/60 | 0 (632 wrong labels, 2 without a box) |
| code | 1,191 | 3/60 wrong but passing (weak dataset tests: MBPP+ 285 twice, KodCode Filter_20259_I); verifier sound | n/a | 921 | n/a (all 921 categorised) | 34 correct solutions followed by an example block (D4); 24 LeetCode jobs whose prelude needs `sortedcontainers` (D6) |

A "sample" column counts the responses judged by hand. A census-wide count is a lower bound for misses: the scan only finds patterns that were looked for.

## Answers to the questions

**1. False credit on real outputs: one defect.** Choice, verdict and code have no verifier false credit in the samples or the census-wide scans. Every credited choice and verdict answer is the response's last box, and its letter or label equals the reference. Credited code passed real tests, but 3 of 60 sampled solutions are wrong on inputs the dataset never tests (`text_match_two_three` accepting `"abcb"`; a KodCode item whose only real test checks a helper). That is a dataset limitation, not a verifier one.

All 14 census-wide false credits in open/direct come from D1:
- `1` for `1+\frac{1}{2^{2020}}` (3 responses);
- `2^{2014} - 1` for `2^{2014}` (3);
- `\frac{2^{2003}+1}{3}` for `(2^{2003}-2)/3` (2);
- `2^{4034} - 2\cdot 3^{2017}` for `4^{2017}-2 \cdot 3^{2017}+2^{2017}` (2);
- two wrong forms of `\frac{2^{1009}}{2^{1009}+1009}` (3);
- `1 + \frac{\ln 2}{1-\ln 2}2^{-2016}` for `1-(1/2)^{2016}` (1).

The following worked as documented on real data:
- decimal references (`52.90`, `0.6778` for `0.68`, `8.94` for `8.9`);
- percentages (`20` for `20\%`);
- units dropped (`48` for `48 \mathrm{~km}/\mathrm{h}`);
- equations unwrapped (`a + d = 0` for `0`);
- bare lists (`-1, 2` for `2, -1`).

Choice extraction ignored letters that appear only in the text (60/60 correct). Verdict labels inside longer text are not extracted (`True (the statement holds)` names nothing).

**2. False credit by construction: four attempts succeed.** These fail as required:
- near-miss numbers outside the rule (`3.15` for `3.14`, `0.51` for `0.5`);
- the right value outside the wrapper asked for, or only in the reasoning;
- several boxes (the last one counts);
- a box listing every option (`A, B, C, D`, or `B or C`);
- `os._exit(0)`, `sys.exit(0)`, `pytest.skip`, rewriting `test_solution.py` or `main.py`, a plain `__eq__`;
- a program-channel answer that reads the reference (there is nothing to read: the sandbox holds only the program).

These succeed:
- off-by-one exact answers beyond 10^40 (D1);
- answers within the documented decimal tolerance (`0.509` for `0.5`, `2.54` for `2.5`, `10.04` for `10.0`), as designed;
- in code, deliberate exploits: reading the nonce from `/tmp/main.py` (passes assert and check), assembling `__eq__` at run time (`setattr(A, '__e'+'q__', ...)`, all harnesses), monkeypatching `_pytest.python.Function.runtest` (pytest), and defining a top-level `zip` or `enumerate` that `_IMPORT_ALL` copies over the test's builtins (passes MBPP+ harnesses);
- a truncated response with an unclosed last box falls through to its program (D10).

The code exploits belong to the known-limitation class, except builtin shadowing, which is not listed.

**3. False rejection on real outputs: several notation- and style-dependent patterns.** Open/direct misses are rare (≥ 11 found). They mostly come from references (D8) and from decimals of exact references (5 cases; a documented limitation). Choice loses 85 correct-letter answers when the letter is followed by LaTeX-formatted option text (D3). Code loses 34 correct solutions when an example-usage block follows them (D4), and 24 LeetCode jobs fail on a missing module (D6). The program channel loses correct values printed inside a sentence (D7). Constructed sympy-style output (`pi/4`, `3*pi`, `E`, `exp(1)`, `2 + 3*I`, `oo`, `log(2)`, `[-1, 3]`) is rejected, though no census case was correct. D3, D4 and D7 depend only on presentation, which is the Round 1 artefact.

**4. Wrappers and guardrail: sound in the verifier, contradicted by the task-layer instruction at `d416aa6`.** Across 31 constructed texts, no wrapper extracted another's form. Literal case is enforced: `Final Answer: 5`, `**Final answer:** 5` and `FINAL ANSWER: 5` comply with nothing, which is strict but by design.

`complied` and `correct_any` behave as the contract says on all re-scored records: `correct_any` is never below the reward, and a later answer in another wrapper is taken rather than "any answer". Held-out wrappers are reachable only from `probe_set` (selection, `tasks.py:549-556`). The training stream draws from `TRAIN_WRAPPERS` only (`tasks.py:726-736`).

Two defects:
- D5: the committed open instruction still says "whichever comes last counts", while the verifier gives the box precedence.
- D10: `answer_tag`, `at` and the box/program fallback break the stated "unclosed last occurrence yields nothing" rule.

**5. Sandbox: no escape found; one cross-job channel.** The job environment has 14 fixed keys and none of the `.env` names. A scan of 88,043 files in the bound venv and uv Python directories found none of the `.env` values. Mounts are only `/usr`, the venv, the uv Python directory and the supervisor. The job has zero capabilities and NoNewPrivs. Writes are refused outside `/tmp`, including via `/proc/1/root`, `/proc/self/root`, `/dev/shm` and `/dev/pts`.

Planted symlinks and directories are removed before the next job. A process hidden in a nested user and PID namespace is detected, so the sandbox is discarded and the process killed with nothing left on the host. The developer's `tests/test_verify.py` and `tests/test_sandbox.py` pass (205 passed; the 48-sandbox test was skipped to keep the pool small).

Findings:
- POSIX message queues persist between jobs in a reused sandbox (D9).
- Nested user namespaces are allowed.
- Memory is bounded only per process (30 processes per job, each at `memory_mb`).

**6. Load dependence: verdicts flip under CPU starvation; quantified.** Re-scoring at a different load from the census gave identical code verdicts (0 flips among 2,112 code jobs; instrumented re-runs agreed). The 13 CPU-limit kills (exit 152) recurred every time.

During the audit the median wall/CPU ratio of sandboxed jobs was 5.6: processes got about a fifth of a CPU, which is exactly the `WALL_FACTOR = 5` boundary. Starvation reproduces a flip. A correct program using 1.2 s of CPU (limit 3 s) passes in 4–5 s at normal load. Pinned to one core with 11 busy loops, it gets reward 0 with error `timeout` after 22–24 s.

A job using CPU c under limit L flips once its share falls below c/(5L). The census distribution of c/L:

| c/L | code jobs (2,103) | program answers (398) | code references (356) |
|---|---|---|---|
| ≥ 0.5 (flips below 1/10 of a CPU) | 1 | 1 | 0 |
| 0.2–0.5 (flips below 1/25 to 1/10) | 4 | 5 | 1 |
| 0.1–0.2 | 27 | 27 | 4 |
| < 0.1 | 2,058 | 365 | 351 |

Open answers are affected too:
- math-verify's 5 s wall-clock alarms (`equivalence.py:106-107`, `:287`) return "no parse" or "not equal" silently, with no error.
- Of the 2,138 distinct answer/reference pairs, 8 need more than 0.5 s of CPU and 1 needs 16 s. The slowest credited pair needs 0.56 s, so it flips below about 1/9 of a CPU.

A load-independent policy:
- Make the CPU limit the only limit that decides. The supervisor reports the job's CPU time (`RUSAGE_CHILDREN` after reaping).
- A wall-clock kill before the CPU limit is reached is reported as undecided (`error="starved"`), not as a timeout. The caller re-runs it, and it never counts as a wrong answer or a rejected reference.
- Raise `WALL_FACTOR` to about 20. Sleeping or blocking programs are rare and cost only slot time.
- Call math-verify with its timeouts disabled, so that only the pool deadline bounds a comparison and its failure says "timeout".

The cost is negligible: about 0.3% of sandboxed census jobs lie above c/L = 0.2.

**7. `gold_ok`: decidability holds; ambiguity passes unflagged.** Every one of the 2,600 census answer items passes `gold_check` at `d416aa6`, and an all-`None` stub of every code reference fails its tests (356/356), so no item has vacuous tests. But 6 of 56 LeetCode references fail now (D6): the suites file was built before preludes were run, and its hash does not cover `judge.py`.

`gold_ok` accepts ambiguous references:
- TheoremQA Python-list references (32/521), parsed as closed intervals or tuples;
- `\log` without a base, read as base 10 (8/212 Omni-MATH);
- equation-of-tuple references (`(x, y, z) = (2, 1, 2)`), which `(2, 1, 2)` does not answer;
- `a=b=c=d`;
- bare-list references to ordering questions, which accept any order (`omni_rest/3332`, "from least to greatest");
- choice items whose correct option text is duplicated (57 AQuA and 19 LogiQA training items).

The 51 Omni-MATH, 39 DeepScaleR, 3 MATH-train and 2 MATH-test references beyond 10^±40 are decidable, but D1 makes them unsafe.

## Defects, by impact

**D1. Exact answers to huge or tiny exact references get credit when wrong (false credit).** In `equivalence.py:153-154,169`, two exact values, rationals included, are equal when `|a − g| ≤ 10⁻⁴⁰ max(|g|, |a|)`.
- Reproduction: `equivalent("2^{2012}-1", "2^{2012}")`, `equivalent("100!+1", "100!")` and `equivalent("\\frac{2^{2012}-1}{3}", "\\frac{2^{2012}+1}{3}")` are all True.
- Exposure: 14 of 73 credited Omni-MATH census answers; references in 24% of Omni-MATH selection items and in 39 DeepScaleR training items.
- Smallest fix: when neither side is decimal and both are Rational, return `g == a` (this alone removes 13 of the 14 census cases). For irrational exact pairs, compare the difference `sympy.N(g - a, DIGITS, maxn=…)` against zero, and treat lost precision as not equal.

**D2. Verdicts depend on machine load (irreproducible rewards and gold rejections).** The wall-clock limit, at 5× the CPU limit (`sandbox.py:46`, `supervisor.py:135`), binds at the CPU share seen during this audit. The math-verify alarms fail silently.
- Reproduction: the starvation experiment in question 6.
- Fix: the policy in question 6. A wall-clock kill with CPU to spare is undecided, `WALL_FACTOR` goes to about 20, and the math-verify timeouts are disabled in favour of the pool deadline.

**D3. A choice answer with the right letter and LaTeX-formatted option text is rejected (style dependence).** `judge.py:150-153` needs the text after a letter to equal the option text exactly. Otherwise the whole answer names nothing, even though the bare letter also matched.
- Reproduction: `resolve_choice("B. \\frac{2}{5}", [..., "2/5", ...])`, `"C. 7.2\\%"` for `7.2%` and `"D. 180 \\text{ Kcal}"` for `180 Kcal` give None. The census has 85 such responses.
- Fix: the letter decides; the attached text only vetoes when it exactly equals a different option's text. The documented `B. 43` case still names none.

**D4. A correct code solution followed by an example block is judged on the example (style dependence).** `wrappers.py:47-52` takes the last Python block. 34 of 61 multi-block uncredited census responses had a passing earlier block, usually followed by `print(f(...))  # Output: ...`. That is 2.8% of correct solutions. `correct_any` misses them too.
- The instruction does ask for one block, so this is defensible, but it penalises presentation and not code.
- Smallest fix: if the last block does not define the names the tests use, run the concatenation of all Python blocks, which passes in all 34 cases. Otherwise, state the trade-off in `VERIFY_NOTES.md` and measure it per member.

**D5. At `d416aa6` the open instruction contradicts the extractor.** `tasks.py:42-43` (WP-D) tells the model "whichever comes last counts", while `judge.py:86-96` lets any box win. 25 census responses that boxed a wrong value and then printed the right one, as instructed, lose. The census responses were all generated under this wording.
- The uncommitted working-tree `OPEN_INSTRUCTION` fixes the wording. Commit it, and treat census strata computed before it with care.

**D6. LeetCode preludes import `sortedcontainers`, which is not installed.** Every response to those items fails (24 census responses), and 6 of 56 LeetCode references now fail `gold_check`. The parts were not invalidated, because the suites hash covers `wrappers.py` but not `judge.py`.
- Fix: add `sortedcontainers` to `pyproject.toml` (the dataset assumes it) and regenerate the suites. Include `judge.py` and `equivalence.py` in the parts key.

**D7. The program channel rejects a correct value printed inside a sentence (style dependence).** `judge.py:225-228` takes the last output line verbatim. Examples: `It would take 1350 minutes to find every Waldo.`, `Louise needs to bake 10 more cakes.`, and `The smallest possible value of b is: 4` (10 census cases). Sympy's printed forms are also rejected.
- Smallest fix (WP-D wording): ask the program to "print only the final answer, as a number or expression". On the verifier side, also read the output line with sympy's printer conventions (`pi`, `E`, `I`, `oo`, `sqrt`, `log`, Python lists).

**D8. `gold_ok` accepts references whose notation decides the outcome.** See question 7.
- Fix at build: write TheoremQA `[a, b]` lists as bare lists (unordered answers) or tuples; read `\log` without a base as `\ln` for Omni-MATH and DeepScaleR; unwrap `tuple = tuple` references like `x = 5`; drop or flag ordering questions whose reference is a bare list; flag choice items whose correct option text is duplicated.

**D9. POSIX message queues survive between jobs in a reused sandbox.** `supervisor.py:99-105` checks only System V IPC.
- Reproduction: one job runs `mq_open("/evo_leak", O_CREAT)` and `mq_send`, and the next job in the same sandbox receives the message.
- Fix: add `(resource.RLIMIT_MSGQUEUE, 0)` to `BOOT` (`supervisor.py:28-30`). Verified: `mq_open` then fails with EMFILE.

**D10. Unclosed last markers fall back, contrary to the stated rule (low impact, mostly on truncated responses).**
- An unclosed last `\boxed{` makes `_boxed` return None, so `judge.py:94` answers by the program. Example: `\boxed{8} … ```python print(7)``` … \boxed{7` gets reward 1 via the program; census case `omni_rest/498`.
- `_last_match` (`wrappers.py:72-78`) uses an earlier closed `<answer>…</answer>` or `@@…@@` when the last opening marker is unclosed.
- Fix: let `_answer` treat "a box present" (balanced or not) as the direct channel, and reject `_last_match` when an opening marker follows the last match.

## Suggestions

- Document builtin shadowing through `_IMPORT_ALL` as another deliberate-exploit path, or skip names that shadow builtins (`judge.py:185-186`).
- Add `--disable-userns` to bwrap, which removes the nested-namespace kernel surface at no cost to jobs.
- Lower `TASKS` for program-channel jobs (single-threaded programs need few processes), so that one job's memory cannot reach 30 × `memory_mb` on a shared machine.
- The exact-reference rule for decimal answers (1e-6 relative) accepted `0.9999999999999994` for `3^{16}/(3^{16}+1)`. Consider requiring the decimal to agree at the precision it states.

## Accepted limitations (confirmed as documented)

- The decimal rule (half a unit and 2%): a short reference meant exactly accepts nearby answers. Exposure: 107 of 521 TheoremQA references and 565 DeepScaleR training references have one decimal.
- An exact reference needs six significant digits from a decimal answer (5 census misses, such as `0.7168` for `7-2\pi`).
- An open interval cannot be told from an ordered pair; structure types must match.
- Network isolation is by namespace; kernel keyrings are not reset.
- The code-harness exploit class: nonce read from `main.py`, assembled `__eq__`, pytest monkeypatching. All were reproduced, all need deliberate code, and none appeared in the census.
- Weak dataset tests let some wrong code pass (3/60 credited code samples). The verifier cannot fix this.
