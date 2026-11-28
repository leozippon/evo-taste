# Verifier notes

Internal notes for `evotaste/verify/`, the verifiers and sandbox that every Round 2 reward passes through. The build specification is the WP-V section of `docs/dev/ROUND2_SPEC.md`; the independent audit is `docs/dev/audits/VERIFIER_AUDIT_R2_2026-11-25.md`. This file states the rules exactly as implemented, how the sandbox is built, what was measured, and the known limitations. The guiding rule is the failure principle: nothing is credited by default, and anything that cannot be decided says so instead of passing for a wrong answer.

## Interface

`verify_many(jobs, timeout=None)` takes `Job(kind, text, gold, wrapper, channels)` and returns one `Verdict(reward, correct_any, complied, channel, extracted, error)` per job, in order. Truncation is the caller's business: `reward` here means the final answer is in the wrapper asked for, through a channel offered, and verified correct. A malformed job (unknown kind, a wrapper that does not apply to the kind, the program channel anywhere but kind `open` with `box`, an open gold of the wrong shape) raises `ValueError` before anything runs.

A verdict whose `error` satisfies `is_undecided(error)` (it contains `undecided:`) says nothing about the answer: a starved program ran out of wall clock before CPU time in three sandbox runs, a sandbox could not run it, a starved verifier process passed its wall-clock deadline, or the process was killed from outside. Its reward is 0, but the caller must repeat it or set it aside, never count it as wrong. Every cause that repeating cannot change is decided instead: a program that reaches its CPU limit or sleeps past the wall clock is a timeout, and a comparison that exhausts its CPU budget or crashes its verifier process is not verified (Pool). The same holds for `gold_check` reasons. `RULES_VERSION` changes whenever a rule that can change a verdict or a `gold_check` result changes; stored verdicts and suites built under another version are stale.

Gold formats:

| Kind | Gold |
|---|---|
| `open` | the reference as a string, LaTeX or plain; or `{"answer": str, "type": t}` with `t` one of `number`, `expression`, `tuple`, `list`, `interval` |
| `choice` | `{"options": [text, ...], "answer": "C"}`, options in letter order |
| `verdict` | `{"labels": [...], "answer": label}` |
| `code` | `{"harness": "pytest" \| "assert" \| "check", "tests", "reference", "entry_point" (check only), "prelude" (optional), "time_limit_s" (CPU seconds, default 10), "memory_mb" (default 1024)}` |

The type of an open reference says how it is compared:

| Type | Reference | Correct answers |
|---|---|---|
| `number`, `expression` | one value | one value, compared whole; a list never answers it |
| `tuple` | elements written `(a, b)`, `[a, b]` or `a, b` | the same elements in the same order, written with any of these brackets or none (TheoremQA lists, "from least to greatest") |
| `list` | elements written as for `tuple`, or `\{a, b\}` | the same elements in any order (a one-to-one matching), with any of these brackets, set braces included (lists of solutions) |
| `interval` | `(a, b)`, `[a, b)`, a union | the same interval with the same brackets |

An untyped reference that is a bare comma list or a parenthesised tuple is still compared as described under Equivalence, but `gold_check` refuses it, so the data layer must state the type of every list, tuple or closed-interval reference.

`gold_check(items)` returns, per item, `None` or the first reason it is unusable; `gold_ok(kind, gold)` is the single-item form. A reference is refused when it is detectably ambiguous:
- an untyped bare list, parenthesised tuple or bracketed list (`1, 2`, `(1, 2)`, `[0.333, 0.25]`);
- a logarithm without a base whose natural and decimal readings differ (`\log 100`; `\log 1` passes);
- a choice item whose correct option's text is also another option's.

Otherwise the reference is rendered in every wrapper of its kind and must earn reward 1 in each; a code reference (the reference solution in a Python block) must pass within half of its item's CPU limit (see Measured). Half-open intervals, sets in braces and unions are not ambiguous. The program channel is not part of the check: a program printing the reference would be verified exactly like the boxed reference.

`run_python(code, *, stdin=None, files=None, timeout_s, memory_mb)` runs a program in the sandbox and returns `Result(returncode, stdout, stderr, timed_out, seconds, cpu_s, undecided)`.

## Where the final answer is

Every wrapper is matched literally, case included, apart from surrounding whitespace. Line wrappers look only at the last non-empty line of the response. Delimiter and block wrappers take their last occurrence anywhere. If the last opening marker is unclosed (an unbalanced `\boxed{`, an `<answer>` or `@@` with nothing closing it, an unterminated ```` ```answer ```` or ```` ```python ```` fence), there is no answer: an earlier occurrence is never used instead, nor another channel. In tolerant extraction it also prevents any wrapper ending before it from answering. Content must be non-empty, and the content of an answer wrapper must not contain another answer wrapper's markup: `Final answer: \boxed{5}` or `<answer>\boxed{5}</answer>` follows the default habit, not the instruction, and is not compliance. Programs are exempt from that rule.

| Id | Role | Form |
|---|---|---|
| `box` | default for open, choice, verdict | last `\boxed{...}` with balanced braces |
| `code` | default for code | the closed fenced blocks tagged `python`, `py` or `python3` (see Code) |
| `final_line` | training | last line `Final answer: X` |
| `json` | training | last line a JSON object whose only key is `"answer"`, value a string or number |
| `hash` | training | last line `#### X` |
| `fence` | training | last fenced block tagged `answer` |
| `at` | training | last `@@X@@` (no `@` or newline inside) |
| `sentence` | training | last line `The answer is X.` (final period optional) |
| `answer_tag` | held-out | last `<answer>X</answer>` |
| `upper_line` | held-out | last line `ANSWER: X` |
| `quote_line` | held-out | last line `> X` |

A test renders every wrapper and checks that no other wrapper extracts anything from it. Each wrapper also carries one instruction sentence ("..., where X is your final answer"); what the answer is (a letter, a label) is left to the prompt.

With wrapper `box` and channels `("direct", "program")`, a response containing a box (closed or not) is answered directly: the last box is the final answer and any program is ignored. (The base model often follows a correct box with an unasked-for program that prints nothing; under "whichever comes last" that lost 165 of 2,400 correct boxes in the smoke census.) Only a response with no box at all and at least one fenced Python block is a program answer, its last block being run. With `("program",)` boxes are ignored and the last block counts (the census's forced channel), with `("direct",)` only the box. `channel` says which case applied.

A program runs in the sandbox with 5 s of CPU time and 512 MiB. The last non-empty line of its standard output is the answer. If that line is a Python expression built only from numbers, `+ - * / **`, the names `pi`, `E`, `I`, `oo` and the functions `sqrt`, `cbrt`, `root`, `exp`, `log`, `ln`, trigonometric ones, `Abs`, `factorial`, `binomial`, `Rational`, it is read as that value. The expression is evaluated from its syntax tree, without `eval`, and integer exponents are capped at 10,000. A printed Python list `[a, b]` is a bare list. Any other line is read like box content. A sentence is not an answer: `Louise needs to bake 10 more cakes.` is wrong for 10. A non-zero exit, a CPU-limit kill or no output is incorrect, with `channel="program"` and the reason in `error`.

`correct_any` is true when the reward is, or when the final answer under the same rules over all wrappers of the kind is correct, or, if there is none and the program channel is offered, the program's answer; for code, untagged fenced blocks also count. It is never "some answer somewhere was right", so a response hedging across two formats gains nothing.

## Equivalence of open answers

Content is cleaned first: surrounding `$`, `$$`, `\(\)`, `\[\]`, `**`, backticks and a whole-answer `\text{}`, `\textbf{}`, `\mathrm{}` or `\mathbf{}` are removed, as is a trailing period; `**` becomes `^`; e-notation becomes a power of ten when the mantissa has a point (`3.2e-19`) or when it is the whole answer (`1e-05`); a coefficient before `i` is spaced, and a free `i` after parsing is the imaginary unit, so `3i`, `3 i` and `(1+i)^2 = 2i` behave. Contents over 500 characters are not taken for one answer. Whole contents are parsed by math-verify's LaTeX parser as box content, with no free-text extraction, no fallback to strings, and its wall-clock alarms off: a slow parse or comparison is bounded by the pool's deadline, so it is undecided, never silently "not equal". The parser's binary floats are turned back into the exact decimals they were read from.

An untyped bare comma list (`1, 2`) or parenthesised tuple (`(1, 2)`, `\left( 1, 2 \right)`) is split at its top-level commas and compared element by element: two bare lists in any order, any other pair in order, so a bare list answers a tuple and the other way round when the elements agree in order; the number of elements must agree. A number with thousands separators (`1,000`, `10,\!000`) is no list, and an interval with a bracket is not split, so it needs the same brackets. Typed references follow their type (Interface).

Numbers, and tuples, sets, intervals and matrices of numbers, are compared by these rules (g the reference, a the answer, |.| the complex modulus):

| Case | Correct when |
|---|---|
| the reference contains a decimal literal | \|a − g\| < ½ unit in the last stated digit of its least precise decimal literal, and \|a − g\| < 2% of \|g\| |
| exact reference, decimal answer | \|a − g\| ≤ 10⁻⁶\|g\| |
| both exact | exactly equal: rationals as such; otherwise the difference must be shown to be zero by sympy (its own evaluation, simplification, or for algebraic numbers the minimal polynomial); undecided means not equal |
| zero reference, decimal involved | \|a\| ≤ 10⁻¹² |

Against a reference that is itself an expression, the exact rule accepts every rewriting of the same value that was tried: fractions, radicals (`\sqrt{5+2\sqrt6}` for `\sqrt2+\sqrt3`), powers, factorials, binomials, logarithms with a base, trigonometric values. It rejects `2^{2012}-1` for `2^{2012}` and `100!+1` for `100!`. Until the audit, two exact values agreeing to 40 significant digits counted as equal, which credited 14 wrong Omni-MATH answers in the census. (`e^{i\pi}` against `-1`, a rewriting sympy does not decide there, is now rejected by the numeral rule anyway; against `\cos\pi` it is accepted.)

A reference written as a numeral wants the arithmetic carried out. A numeral is how a final answer writes a number: an integer (with thousands separators or a base subscript, `2003_6`), a decimal, a fraction of two integers (`\frac{3}{4}`, `\dfrac`, `\tfrac`, `3/4`, `3:4`) or a mixed number, `m \times 10^{k}` or `10^{k}`; signed, as a percentage or in degrees, possibly in parentheses. It is judged on the content as math-verify normalizes it (units, `\text`, currency removed), and for an equation on its last right-hand side, so `2 + 3 = 5` is a numeral. Against such a reference an answer that is not a numeral is not verified, whatever its value: `17 \cdot 23 + 4` for `395`, `2^{10}`, `8!`, `\binom{16}{7}`, `\sqrt{4}`, `0.5 \cdot 4`, `\frac{1}{5^{10}}` for `0.0000001024`. Under training such an answer is a shortcut past the final computation, where the errors happen. A decimal reference stands for a rounded value, so it still takes an exact form whose value sympy does not find rational, which no numeral could write (`\pi` for `3.14`, `\frac{1+\sqrt5}{2}` for `1.618`, `\log_2 13` for `3.7`), while `1 + \frac{1}{3}` for `1.33` is rejected. The rule concerns whole contents, and so each element of a split list or tuple; numbers inside sets, intervals and matrices are compared as before.

A percentage X% has the value X/100 and also matches X, as in math-verify. Containers must have the same type and shape; tuples compare in order, sets by a one-to-one matching, intervals by endpoints and openness. An answer `x = 5` or `2 + 3 = 5` stands for its last right-hand side when the reference is no equation, and a reference `x = 5` is answered by `5`; inequalities are never unwrapped. A number against anything with variables is wrong. Everything else (expressions with variables, equations, unions) goes to math-verify's `verify`, in strict mode.

math-verify's own numeric comparison is bypassed because it gives false credit: it rounds floats to six decimals in absolute terms (1e-7 equals 2e-7, 1.6e-19 equals 3.2e-19) and compares exact values with `evalf(chop=True)` (2^-60 equals 2^-61, 10^-20 equals 2·10^-20). The decimal rule lets an exact or more precise answer to a rounded reference count (4/3 and 1.3333 for `1.33`, 2/3 for `0.67`, π for `3.14`), so that exact reasoning is not penalised against decimal rounding. The 2% cap keeps one-digit references from accepting answers far off (0.94 for `0.9`); the price is that a short reference meant exactly accepts answers within 2% (0.504 for `0.5`, though not 0.51). Rationals are compared exactly, so the strict bounds hold at their boundaries.

## Choice and verdict

A choice answer is cleaned (math delimiters, `\text`-like commands, markdown emphasis, quotes, a trailing period) and read three ways:
- a capital option letter in its usual dressings (`C`, `(C)`, `[C]`, `C.`, `C)`, `C:`, `Option C`);
- a letter with a separator followed by any text (`C. 42`, `B. \frac{2}{5}`, `(D) 180 \text{ Kcal}`): the letter decides;
- the exact text of an option (case and spacing aside).

The text after a letter only vetoes when it is exactly another option's text (`B. 43` with 43 under D), or when it names another option letter (`(B), (C)`, `B. or C`). The answer counts only if its readings name exactly one existing option, so `B or C`, a lower-case `b`, or a text shared by two options name none. A verdict is correct when the cleaned answer equals exactly one label of the item's closed set, case aside.

## Code

The program is saved as `solution.py`, after the item's prelude if it has one (trusted dataset text such as LeetCode's `from typing import *`, `ListNode` and `sortedcontainers` imports, placed exactly as if written at the top of the file). Every top-level name of that file (underscored ones too) is imported ahead of the tests, and the harness runs in the sandbox with the item's CPU-time and memory limits:
- `pytest` (KodCode): the tests run as `test_solution.py`; pass means pytest exits 0 with at least one passed test, no failure and no skip or xfail.
- `assert` (MBPP+, whose `test` field is a script of assertions calling the function by name, and MBPP-style assertion lists): the tests run as a script.
- `check` (HumanEval+, LeetCode): the tests run as a script, then `check(<entry_point>)`.

Which program runs: the last closed Python block, unless the response has several and the last does not define every name that the tests use and the reference defines (top-level functions, classes, assignments, imports). Then all Python blocks run, concatenated in order. A correct solution followed by an example-usage block passes, while a last block that redefines the tested function is judged alone. An unterminated last ```` ```python ```` fence is no answer.

Success is a random token the harness prints after the tests, together with exit status 0, so `os._exit(0)`, `sys.exit(0)`, `exit()` or killing itself never passes. The supplied files are written read-only (mode 0444); a job can change that for its own files, but the harness and the tests have been read by then, and nothing it writes reaches a later job. A program containing `__eq__` fails unless the reference contains it too, because an object whose `__eq__` always returns true would pass equality-based tests. The rule looks at the model's code only, so a prelude defining `ListNode.__eq__` licenses nothing.

## Sandbox

A sandbox is a long-lived bwrap instance, one per thread that calls `run_python` (one per verifier process in the pool), started as `setpriv --pdeathsig KILL bwrap ...` in its own session. Its root is empty except for read-only binds of `/usr` (with the merged-usr symlinks `/bin`, `/lib`, `/lib64`, `/sbin`), the project's virtual environment (`sys.prefix`), the directory of uv-managed interpreters its `python` links to (`sys.base_prefix`'s parent) and the supervisor script, a fresh `/proc`, bwrap's minimal `/dev`, and a 64 MiB tmpfs at `/tmp`. The root and `/dev`, which bwrap builds as uncapped tmpfs mounts, are remounted read-only, so `/tmp` is the only writable place. The repository, `.env`, home directories, `/etc`, other users' data, host `/tmp` and host sockets do not exist inside; a bind that would expose the repository root, `/home` or the home directory as a whole is refused. All namespaces are unshared (network: only a private loopback; own pid and IPC namespaces), the environment is replaced by a fixed one (`HOME=/tmp`, UTF-8, `PYTHONHASHSEED=0`, one BLAS thread, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`), and capabilities are dropped.

Sandboxes are reused because bwrap's own setup is serialized by the kernel machine-wide: about 30 starts per second here whatever the concurrency, about 6 under other users' load (bare user and mount namespaces from `unshare` reach over 1,000 per second, so the cost is bwrap's mounts). PID 1 of each sandbox is a supervisor (`supervisor.py`, standard library only), which receives jobs over a pipe as length-prefixed JSON and runs them one at a time.

For each job the supervisor writes `main.py` and the files into the empty `/tmp` with mode 0444 and starts a fresh interpreter. Its bootstrap sets hard limits the job cannot raise:
- address space `memory_mb`;
- CPU time ceil(`timeout_s`) s;
- 8 MiB per file, which also caps stdout and stderr (captured in memfds);
- 32 processes plus threads;
- no core dumps, and no POSIX message queue bytes (`RLIMIT_MSGQUEUE` 0, so no queue can carry data to the next job).

When the job's interpreter exits, or when the wall clock runs out, the supervisor kills every other process in the namespace (as its init, `kill(-1)` reaches all of them, detached and re-sessioned ones included) and reaps them. It answers with the job's CPU time (all its processes, from `RUSAGE_CHILDREN`), then empties `/tmp` and says whether that worked: a slow reset (a job that left 100,000 files) costs the sandbox, never the verdict. The supervisor is not dumpable and installs no signal handlers, so a job (same uid, no capabilities) can neither ptrace it nor open its `/proc` entries nor signal it.

Only the CPU-time limit decides that a program took too long, because CPU time does not depend on the machine's load: this account's processes share a quota of 32 CPUs (the user slice's `cpu.max`) that other jobs saturate at times, when a busy process gets a fifth of a CPU or less. A job killed at its CPU limit exits with status 152 (128 + SIGXCPU), or 137 after ignoring it, and counts as a timeout: decided and wrong. The wall clock, 20 times the limit, stops programs that sleep or block. If the job's main process had waited on a run queue for at least a quarter of its wall-clock time (`/proc/<pid>/schedstat`), it was starved: the run decides nothing and is repeated in a fresh sandbox, up to three attempts, after which the result is undecided. Otherwise it slept or blocked, which repeating cannot change, and it is a decided timeout. A sandbox that cannot run the job gives an undecided result too. A test pins a job using 0.7 s of its 1 s CPU limit to one core with 11 busy processes: it passes as it does on an idle core, in several times the wall-clock time.

The sandbox is discarded and a new one started on the next job when a job ran out of wall clock, died of a signal, left a process behind, left a System V IPC object, or `/tmp` could not be emptied, and whenever the supervisor answers late (past the wall limit plus 5 s, or 30 s more for the reset), malformed, or not at all. The host then kills bwrap's process group; the supervisor dies with it (`--die-with-parent`), and with the namespace's init the kernel kills everything left in it. A new sandbox has 60 s to report ready, and its setup never counts against a job. Because bwrap runs under `setpriv --pdeathsig KILL`, the death of the thread that started it (as when the pool kills a stuck verifier process) also tears the sandbox down.

## Pool

`verify_many` keeps a persistent pool of `PROCESSES = 32` verifier processes per calling process (at most 48 on this shared machine): fresh interpreters started with `python -c`, not forks of the caller. Each judges one job at a time, under a 4 GiB address-space limit, and dies with its parent.

Outside sandbox runs (parsing and comparison) a job may spend `COMPARISON_CPU_S = 10` s of CPU time in its verifier process. The budget is a soft `RLIMIT_CPU`, armed only outside sandbox runs (which have their own limits and must not inherit it), so the kernel stops a process that exhausts it even inside C code, where no signal handler runs: sympy computing a power tower such as `2^{2^{2007}}` or `10^{10^{9}}`. The verdict is then decided, reward 0 with error `comparison exceeded its CPU budget (10 s)`, with `complied` and `extracted` from extraction alone, because CPU time does not depend on the machine's load. A process that crashes on a job (segmentation fault, abort) gives a decided `comparison crashed the verifier process` the same way. A process killed from outside (the kernel's out-of-memory killer) gives `undecided: verifier process died`. An exception inside the judgment (a reference that does not parse, a `MemoryError` at the 4 GiB limit) gives reward 0 with the exception as error.

The wall clock outside sandbox runs is 200 s (20 times the budget); during each run, the pool allows the longest the run may take, which the process reports when the run starts and ends. A job past that deadline still had CPU budget to spare, so it was starved: its process is killed and replaced and the verdict is `undecided: verifier deadline: ... (starved)`. `timeout` replaces these deadlines by one in all if given.

The budget comes from the census. On the 118,046 open, choice and verdict records of the full census's training part on disk (`results/census/census-r2/train`), judging took at most 0.22 s of CPU for a correct answer (p99.9 0.06 s for open answers) and at most 2.42 s for any answer that finished (wrong trigonometric equations); nothing took between 2.42 s and 60 s, and the two that took longer (`\frac{2}{2^{2^{2007}} + 1}` and another tower, `math_train/intermediate_algebra/1209`) never finish. (Their reference is the numeral `2`, so the numeral rule now rejects them without computing anything; towers still reach the budget against references that are expressions.) The slowest open reference checks itself in 0.32 s (all 47,578). Ten seconds is thus 45 times the slowest correct comparison and leaves room for CPU time growing about twofold under load. Before this budget, such answers either failed on memory by chance or ran into the wall-clock deadline every time, which stopped the selection census with three verdicts undecided after three attempts. The pool is not meant to be used from several threads at once.

## Measured

`python -m evotaste.verify.selfcheck` reproduces the reference numbers (300 references per source, seeded; only training splits, MMLU-Pro's validation split and the references of HumanEval+, MBPP+ and TheoremQA; no model). Run on 2026-11-25 with 32 processes:

| Source | Kind | References usable | Refused as ambiguous | Perturbations rejected |
|---|---|---|---|---|
| math_train | open | 282/300 | 18 (untyped lists) | 1092/1092 |
| deepscaler | open | 297/300 | 0 | 1172/1172 |
| gsm8k_train | open | 300/300 | 0 | 1200/1200 |
| theoremqa | open | 281/300 | 19 (untyped lists) | 1119/1119 |
| aqua | choice | 296/300 | 4 (repeated correct option) | 592/592 |
| logiqa | choice | 299/300 | 1 (repeated correct option) | 598/598 |
| arc | choice | 300/300 | 0 | 600/600 |
| mmlu_pro (validation) | choice | 70/70 | 0 | 140/140 |
| proofwriter | verdict | 300/300 | 0 | 600/600 |
| kodcode | code | 296/300 | | |
| humaneval_plus | code | 161/164 | | |
| mbpp_plus | code | 297/300 | | |

Perturbations are off by one either way, sign flipped and doubled for numbers (`g + 1` and `2g` for expressions), one unit more in the last digit for decimals, another letter and another option's text for choice, every other label for verdict. The ambiguous references are untyped lists (MATH, TheoremQA) and choice items whose correct option's text is repeated (Interface; the data layer types the lists). The other unusable open references are malformed (DeepScaleR: display math with a box inside, an empty string, `33\frac{1}{3}\%`). The code failures are references that cannot pass here:
- two KodCode items import torch, whose libraries cannot even be mapped within 1 GiB of address space (40 of KodCode's 10,000 items import torch);
- one KodCode item needs seaborn (57 items; not installed), and one asserts a timing;
- HumanEval/32's dataset harness is itself broken (`_poly(*candidate(*inp), inp)`);
- HumanEval+ `special_factorial` and `string_sequence`, and two MBPP+ references (`sum_average` among them), need more than half of their 10 s CPU limit (13–40 s measured); MBPP+ `combinations_colors` needs more than 1 GiB on its largest input.

A code reference must pass within half its CPU limit because CPU time itself grows on a loaded machine: two of those HumanEval+ references passed within 10 s at load average 50 and needed 13–16 s at load average 65–180 (shared cores). Items that need more get a larger `time_limit_s` from the data layer. The check is still not reproducible across machine states: regenerating the suites file on 2026-11-28 (load average about 17) decided 9 code references differently from the committed file, four KodCode `Docs: Scikit` items and four LeetCode items passing and the KodCode item that asserts a timing failing, each the same way in three repeats. The committed file keeps the earlier decisions, which the census strata table depends on (it names the timing item).

The stored smoke census (`results/census/census-r2-limit100`, 18,620 non-procedural records whose reference is known) was re-scored with the verifier before and after the audit fixes, in the same environment. Changed verdicts:

| Cell | Records | 0 → 1 | 1 → 0 |
|---|---|---|---|
| open, direct | 6,369 | 0 | 14 |
| open, program | 403 | 0 | 1 |
| choice | 7,760 | 85 | 0 |
| verdict | 1,980 | 0 | 0 |
| code | 2,108 | 33 | 0 |

The 14 open losses are exactly the audit's 14 false credits (D1: `1` for `1+\frac{1}{2^{2020}}`, `2^{2014}-1` for `2^{2014}`, and the others it lists). The program loss is its D10 case, a truncated response whose last box is unclosed and whose program printed `0.9999999999999994`. The choice gains are the 85 right-letter answers with formatted option text (D3). The code gains are solutions followed by an example block (D4). No verdict is undecided. The 20 census jobs whose LeetCode prelude imports `sortedcontainers` now import it (all 20 answers are wrong).

The numeral rule (rules `2026-11-28.final-form`) was measured on 2026-11-28 by judging stored open records again under the rules before and after it; the earlier rules reproduced every stored reward.

| Records | Open records | Credited | Lose credit | Gain credit |
|---|---|---|---|---|
| census, training part (math) | 72,000 | 45,334 | 29 (23 DeepScaleR, 6 MATH) | 0 |
| pilot C, selection tag `f90` (19 adapters) | 20,558 | 12,586 | 4 (MATH test) | 0 |

All 33 losses are expressions still to be evaluated: `8!` for `40320`, `2021 \cdot 2022`, `\frac{16!}{4! \cdot 4! \cdot 8!}`, `\binom{16}{7}`, `\log_{10}{(256 \cdot 390625)}` for `8`, an unevaluated integral, `8^6`, `3^2`. The closest call, `\frac{14.8}{28.3}` for `\frac{148}{283}`, answers a question that asks for a common fraction. The other credited answers that contain an operator (about 1,500) answer references that are themselves expressions (`2\sqrt3 - 1`, `x^2 + 6x - 4`), except 8 exact irrational forms of decimal references (`8.5 - 2\pi` for `2.217`). Under the new rules 9 of the census's 27,612 strata rows would change stratum (7 sweet to frontier, 2 mastered to sweet).

Cost per job when run alone: an open answer about 4 ms (cold, one process), choice and verdict well under 1 ms, a sandboxed program 0.09 s, a code item 0.2–0.3 s. Sandboxed jobs are bound by the CPU they get (a fresh interpreter per job, pytest's import for code), not by bwrap's machine-wide start rate, which only new sandboxes pay. On 2026-11-25, at load average about 50, 32 processes verified 518 program-channel jobs and 44 KodCode code jobs per second (28 and 16 when every job started its own sandbox).

## Known limitations

- The `__eq__` rule is textual and catches only the plain always-equal object. A program built to defeat the harness can still pass:
  - reading the token from `/tmp/main.py`;
  - monkeypatching pytest from the solution;
  - a `__bool__` or `__len__` trick on tests that check truth only;
  - `__eq__` assembled at run time;
  - a top-level `zip` or `enumerate` that the name import copies over the tests' builtins.

  All of these need deliberate adversarial code; none appeared in the census. The census audit and record inspection should watch for them.
- Network isolation is by namespace: a program can create a socket, but there is nothing to connect to except its own loopback. Nested user namespaces are allowed (a process hidden in one is still found and killed between jobs). Memory is bounded per process (up to 32 processes per job).
- The numeral rule leaves two gaps. A decimal reference still accepts an expression whose value is rational when sympy cannot tell (`\sin^2 1 + \cos^2 1` for `1.0`), and numbers inside sets, intervals and matrices are not held to it (`\{17 \cdot 23 + 4, 2\}` for `\{395, 2\}`). Neither occurred in the measured records.
- Decimal references are ambiguous between exact and rounded values: a short reference meant exactly accepts answers within 2% and within half a unit of its last digit (0.504 for `0.5`). An exact reference wants six significant digits from a decimal answer (`0.7071` for √2/2 is rejected, `0.9999999999999994` for `3^{16}/(3^{16}+1)` accepted).
- Untyped bare lists and tuples follow the rules above, and an open interval `(1, 2)` cannot be told from an ordered pair, which is why `gold_check` wants a type. The one-to-one matching of lists is greedy, which can only miss a match, never invent one. Unions of intervals and expressions with variables use math-verify, whose `evalf(chop=True)` comparison can still equate expressions that differ only by a tiny constant.
- Word-like references (`\text{Evelyn}`, `True`) parse into products of symbols and pass `gold_check`; the build must drop free-text references, and TheoremQA's boolean answers belong to kind `verdict`. References written as an equation of tuples (`(x, y, z) = (2, 1, 2)`) or a chain (`a=b=c=d`) are not unwrapped; the build should write the tuple.
- A sandbox run that sleeps or blocks costs 20 times its limit; one that stays starved costs that three times and is undecided. A comparison that never finishes costs 10 s of CPU. A timed-out job loses its compliance information (`complied` false).
- Between jobs the supervisor resets processes, `/tmp`, System V IPC and (by the limit) POSIX message queues; it does not reset kernel keyrings, which a job could only fill through raw system calls, and which no later job reads unless written to.
- Not measured here: the BBH, FOLIO, FLD and other unseen sources; their references go through `gold_check` at build time.
