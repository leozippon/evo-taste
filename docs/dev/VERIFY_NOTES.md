# Verifier notes

Internal notes for `evotaste/verify/`, the verifiers and sandbox that every Round 2 reward passes through. The build specification is the WP-V section of `docs/dev/ROUND2_SPEC.md`. This file states the rules exactly as implemented, how the sandbox is built, what was measured, and the known limitations. The guiding rule is the failure principle: anything that cannot be decided scores 0 and says why; nothing is credited by default.

## Interface

`verify_many(jobs, timeout=None)` takes `Job(kind, text, gold, wrapper, channels)` and returns one `Verdict(reward, correct_any, complied, channel, extracted, error)` per job, in order. Truncation is the caller's business: `reward` here means the final answer is in the wrapper asked for, through a channel offered, and verified correct. A malformed job (unknown kind, a wrapper that does not apply to the kind, the program channel anywhere but kind `open` with `box`) raises `ValueError` before anything runs.

Gold formats:

| Kind | Gold |
|---|---|
| `open` | the reference as a string, LaTeX or plain |
| `choice` | `{"options": [text, ...], "answer": "C"}`, options in letter order |
| `verdict` | `{"labels": [...], "answer": label}` |
| `code` | `{"harness": "pytest" \| "assert" \| "check", "tests", "reference", "entry_point" (check only), "prelude" (optional), "time_limit_s" (default 10), "memory_mb" (default 1024)}` |

`gold_check(items)` renders each reference in every wrapper of its kind (for code: the reference solution in a Python block) and returns, per item, `None` or the first reason it does not earn reward 1; `gold_ok(kind, gold)` is the single-item form. A reason of `timeout` means undecided rather than wrong. The program channel is not part of the check: a program printing the reference would be verified exactly like the boxed reference.

`run_python(code, *, stdin=None, files=None, timeout_s, memory_mb)` runs a program in the sandbox and returns `Result(returncode, stdout, stderr, timed_out, seconds)`.

## Where the final answer is

Every wrapper is matched literally, case included, apart from surrounding whitespace. Line wrappers look only at the last non-empty line of the response. Delimiter and block wrappers take their last occurrence anywhere; if that occurrence is unclosed (unbalanced braces, an unterminated fence) there is no answer, and an earlier occurrence is never used instead. Content must be non-empty, and the content of an answer wrapper must not contain another answer wrapper's markup: `Final answer: \boxed{5}` or `<answer>\boxed{5}</answer>` follows the default habit, not the instruction, and is not compliance. Programs are exempt from that rule.

| Id | Role | Form |
|---|---|---|
| `box` | default for open, choice, verdict | last `\boxed{...}` with balanced braces |
| `code` | default for code | last closed fenced block tagged `python`, `py` or `python3` |
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

With wrapper `box` and channels `("direct", "program")`, a response containing a boxed answer is answered directly: the last box is the final answer and any program is ignored. Only a response with no box at all and at least one fenced Python block is a program answer, its last block being run. (The base model often follows a correct box with an unasked-for program that prints nothing; under "whichever comes last" that lost 165 of 2,400 correct boxes in the smoke census.) With `("program",)` boxes are ignored and the last block counts (the census's forced channel), with `("direct",)` only the box. `channel` says which case applied. A program runs in the sandbox with 5 s of CPU time and 512 MiB; the last non-empty line of its standard output is verified like box content. A non-zero exit, a timeout or no output is incorrect, with `channel="program"` and the reason in `error`.

`correct_any` is true when the reward is, or when the answer ending last among all wrappers of the kind is correct, or, if there is none and the program channel is offered, the program's answer; for code, an untagged fenced block also counts. It is never "some answer somewhere was right", so a response hedging across two formats gains nothing.

## Equivalence of open answers

Content is cleaned first: surrounding `$`, `$$`, `\(\)`, `\[\]`, `**`, backticks and a whole-answer `\text{}`, `\textbf{}`, `\mathrm{}` or `\mathbf{}` are removed, as is a trailing period; `**` becomes `^`; e-notation becomes a power of ten when the mantissa has a point (`3.2e-19`) or when it is the whole answer (`1e-05`); a coefficient before `i` is spaced so that `3i` and `3 i` both mean the imaginary unit. Contents over 500 characters are not taken for one answer. Both sides are then parsed by math-verify's LaTeX parser as box content, with no free-text extraction and no fallback to strings (its expression extractor finds numbers anywhere in a text, which turned `x**2 + 1` into `1`). The parser's binary floats are turned back into the exact decimals they were read from.

A bare comma list (`1, 2`, read as a list of solutions) or a parenthesised tuple (`(1, 2)`, `\left( 1, 2 \right)`) is split at its top-level commas and compared element by element, each element by the rules below: two bare lists in any order (a one-to-one matching), any other pair in order, so a bare list answers a tuple and a tuple answers a bare list when the elements agree in order; the number of elements must agree. A number with thousands separators (`1,000`, `10,\!000`) is no list, and an interval with a bracket (`[0, 1)`) is not split, so it still needs the same brackets.

Numbers, and tuples, sets, intervals and matrices of numbers, are compared by these rules (g the reference, a the answer, |.| the complex modulus):

| Case | Correct when |
|---|---|
| the reference contains a decimal literal | \|a − g\| < ½ unit in the last stated digit of its least precise decimal literal, and \|a − g\| < 2% of \|g\| |
| exact reference, decimal answer | \|a − g\| ≤ 10⁻⁶\|g\| |
| both exact | equal to 40 significant digits (60-digit evaluation) |
| zero reference, decimal involved | \|a\| ≤ 10⁻¹² |

A percentage X% has the value X/100 and also matches X, as in math-verify. Containers must have the same type and shape; tuples compare in order, sets by a one-to-one matching, intervals by endpoints and openness. An answer `x = 5` or `2 + 3 = 5` stands for its last right-hand side when the reference is no equation, and a reference `x = 5` is answered by `5`; inequalities are never unwrapped. A number against anything with variables is wrong. Everything else (expressions with variables, equations, unions) goes to math-verify's `verify`, in strict mode.

math-verify's own numeric comparison is bypassed because it gives false credit: it rounds floats to six decimals in absolute terms (1e-7 equals 2e-7, 1.6e-19 equals 3.2e-19) and compares exact values with `evalf(chop=True)` (2^-60 equals 2^-61, 10^-20 equals 2·10^-20). The decimal rule lets an exact or more precise answer to a rounded reference count (4/3 and 1.3333 for `1.33`, 2/3 for `0.67`, π for `3.14`), so that exact reasoning is not penalised against decimal rounding. The 2% cap keeps one-digit references from accepting answers far off (0.94 for `0.9`); the price is that a short reference meant exactly accepts answers within 2% (0.504 for `0.5`, though not 0.51). Rationals are compared exactly, so the strict bounds hold at their boundaries.

## Choice, verdict and code

A choice answer is cleaned (math delimiters, `\text`-like commands, markdown emphasis, quotes, a trailing period) and read three ways: a capital option letter in its usual dressings (`C`, `(C)`, `[C]`, `C.`, `C)`, `C:`, `Option C`), a letter followed by that same option's exact text (`C. 42`), or the exact text of an option (case and spacing aside). It counts only if these readings name exactly one existing option, so `B or C`, `B. 43` with 43 under D, a lower-case `b`, or a text shared by two options name none. A verdict is correct when the cleaned answer equals exactly one label of the item's closed set, case aside.

For code, the program is saved as `solution.py`, after the item's prelude if it has one (trusted dataset text such as LeetCode's `from typing import *` and `ListNode`, placed exactly as if written at the top of the file), every top-level name of that file (underscored ones too) is imported ahead of the tests, and the harness runs in the sandbox with the item's time and memory limits:

- `pytest` (KodCode): the tests run as `test_solution.py`; pass means pytest exits 0 with at least one passed test, no failure and no skip or xfail.
- `assert` (MBPP+, whose `test` field is a script of assertions calling the function by name, and MBPP-style assertion lists): the tests run as a script.
- `check` (HumanEval+, LeetCode): the tests run as a script, then `check(<entry_point>)`.

Success is a random token the harness prints after the tests, together with exit status 0, so `os._exit(0)`, `sys.exit(0)`, `exit()` or killing itself never passes. The supplied files are written read-only (mode 0444); a job can change that for its own files, but the harness and the tests have been read by then, and nothing it writes reaches a later job. A program containing `__eq__` fails unless the reference contains it too (the rule looks at the model's code only, so a prelude defining `ListNode.__eq__` licenses nothing), because an object whose `__eq__` always returns true would pass equality-based tests.

## Sandbox

A sandbox is a long-lived bwrap instance, one per thread that calls `run_python` (one per verifier process in the pool), started as `setpriv --pdeathsig KILL bwrap ...` in its own session. Its root is empty except for read-only binds of `/usr` (with the merged-usr symlinks `/bin`, `/lib`, `/lib64`, `/sbin`), the project's virtual environment (`sys.prefix`), the directory of uv-managed interpreters its `python` links to (`sys.base_prefix`'s parent) and the supervisor script, a fresh `/proc`, bwrap's minimal `/dev`, and a 64 MiB tmpfs at `/tmp`. The root and `/dev`, which bwrap builds as uncapped tmpfs mounts, are remounted read-only, so `/tmp` is the only writable place. The repository, `.env`, home directories, `/etc`, other users' data, host `/tmp` and host sockets do not exist inside; a bind that would expose the repository root, `/home` or the home directory as a whole is refused. All namespaces are unshared (network: only a private loopback; own pid and IPC namespaces), the environment is replaced by a fixed one (`HOME=/tmp`, UTF-8, `PYTHONHASHSEED=0`, one BLAS thread, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`), and capabilities are dropped.

Sandboxes are reused because bwrap's own setup is serialized by the kernel machine-wide: about 30 starts per second here whatever the concurrency, about 6 under other users' load (bare user and mount namespaces from `unshare` reach over 1,000 per second, so the cost is bwrap's mounts). PID 1 of each sandbox is a supervisor (`supervisor.py`, standard library only), which receives jobs over a pipe as length-prefixed JSON and runs them one at a time. For each job it writes `main.py` and the files into the empty `/tmp` with mode 0444 and starts a fresh interpreter, whose bootstrap sets hard limits the job cannot raise: address space `memory_mb`, CPU time ceil(`timeout_s`) s, 8 MiB per file (which also caps stdout and stderr, captured in memfds), 32 processes plus threads, no core dumps. When the job's interpreter exits, or after five times `timeout_s` of wall-clock time, the supervisor kills every other process in the namespace (as its init, `kill(-1)` reaches all of them, detached and re-sessioned ones included), reaps them, and empties `/tmp`. The supervisor is not dumpable and installs no signal handlers, so a job (same uid, no capabilities) can neither ptrace it nor open its `/proc` entries nor signal it.

The time limit is CPU time because CPU time does not depend on the machine's load: this account's processes share a quota of 32 CPUs (the user slice's `cpu.max`) that other jobs saturate at times, when a busy process gets about a fifth of a CPU. The wall clock, five times as long, only stops programs that sleep or block. A job killed at its CPU limit exits with status 152 (128 + SIGXCPU) and counts as a timeout.

The sandbox is discarded and a new one started on the next job when a job timed out, died of a signal, left a process behind, left a System V IPC object, or `/tmp` could not be emptied, and whenever the supervisor answers late (past the wall limit plus 5 s), malformed, or not at all. The host then kills bwrap's process group; the supervisor dies with it (`--die-with-parent`), and with the namespace's init the kernel kills everything left in it. A new sandbox has 60 s to report ready; its setup never counts against a job. Because bwrap runs under `setpriv --pdeathsig KILL`, the death of the thread that started it (as when the pool kills a stuck verifier process) also tears the sandbox down. A sandbox that cannot run a job gives a timed-out result with the reason on stderr, never a verdict.

## Pool

`verify_many` keeps a persistent pool of `PROCESSES = 32` verifier processes per calling process (at most 48 on this shared machine): fresh interpreters started with `python -c`, not forks of the caller. Each judges one job at a time, under a 4 GiB address-space limit, and dies with its parent. A job's deadline is `timeout` if given, else 20 s plus the worst case of its sandbox runs (one program with the program channel; two test runs for code, because the tolerant extraction may run a second program), each counted as five times its time limit plus 65 s for a new sandbox's setup and the supervisor's answer. Those long deadlines are only reached by a hang outside the sandbox, which only parsing can produce. Past it the process is killed and replaced and the job gets `Verdict(0, False, False, None, None, "timeout")`; a process that dies gives error `verifier process died`, and an exception inside the judgment (for instance a reference that does not parse) gives reward 0 with the exception as error. This is what bounds sympy stuck in C code: a boxed `10^{10^{10}}` cannot be interrupted by any signal. The pool is not meant to be used from several threads at once.

## Measured

`python -m evotaste.verify.selfcheck` reproduces these numbers (300 references per source, seeded; only training splits, MMLU-Pro's validation split and the references of HumanEval+, MBPP+ and TheoremQA; no model). Run on 2026-11-25 with 32 processes:

| Source | Kind | References verifying in every wrapper | Perturbations rejected |
|---|---|---|---|
| math_train | open | 300/300 | 1090/1090 |
| deepscaler | open | 297/300 | 1172/1172 |
| gsm8k_train | open | 300/300 | 1200/1200 |
| theoremqa | open | 300/300 | 1119/1119 |
| aqua | choice | 300/300 | 600/600 |
| logiqa | choice | 300/300 | 600/600 |
| arc | choice | 300/300 | 600/600 |
| mmlu_pro (validation) | choice | 70/70 | 140/140 |
| proofwriter | verdict | 300/300 | 600/600 |
| kodcode | code | 297/300 | |
| humaneval_plus | code | 163/164 | |
| mbpp_plus | code | 298/300 | |

Perturbations are off by one either way, sign flipped and doubled for numbers (`g + 1` and `2g` for expressions), one unit more in the last digit for decimals, another letter and another option's text for choice, every other label for verdict. The three DeepScaleR failures are malformed references (display math with a box inside, an empty string, `33\frac{1}{3}\%`). The code failures are references that cannot pass here: two KodCode items import torch, whose libraries cannot even be mapped within 1 GiB of address space (40 of KodCode's 10,000 items import torch), one needs seaborn (57 items; not installed), and one, which asserts a timing, fails in some runs; HumanEval/32's dataset harness is itself broken (`_poly(*candidate(*inp), inp)`); of MBPP+, `sum_average` takes 12 s even alone and `combinations_colors` exceeds 1 GiB on its largest input. Evaluation sources therefore want per-item time and memory limits with room, set at build time.

Cost per job when run alone: an open answer about 4 ms (cold, one process), choice and verdict well under 1 ms, a sandboxed program 0.09 s, a code item 0.2–0.3 s. Throughput with 32 processes, measured while the machine was loaded (load average about 52, this account's CPU quota being throttled):

| Workload | Jobs per second | Before sandboxes were reused |
|---|---|---|
| open, box (answers rewritten by sympy) | 3,900 | 3,600 |
| open, program channel | 226 | 28 |
| choice | 24,000 | 17,000 |
| verdict | 26,000 | 23,000 |
| code (KodCode references) | 31 | 16 |

Sandboxed jobs are now bound by the CPU they get (a fresh interpreter per job, pytest's import for code), not by bwrap's machine-wide start rate, which only new sandboxes pay.

## Known limitations

- The `__eq__` rule is textual and catches only the plain always-equal object; a program built to defeat the harness (reading the token from `/tmp/main.py`, monkeypatching pytest from the solution, a `__bool__` or `__len__` trick on tests that check truth only, `__eq__` assembled at run time) can still pass. All of these need deliberate adversarial code; the census audit and record inspection should watch for them.
- Network isolation is by namespace: a program can create a socket, but there is nothing to connect to except its own loopback.
- Decimal references are ambiguous between exact and rounded values: a short reference meant exactly accepts answers within 2% and within half a unit of its last digit (0.504 for `0.5`). An exact reference wants six significant digits from a decimal answer (`0.7071` for √2/2 is rejected).
- Apart from bare lists and tuples, structures must match in type (a set against an interval is wrong), and an open interval `(1, 2)` cannot be told from an ordered pair, so `1, 2` answers it. The one-to-one matching of bare lists is greedy, which can only miss a match, never invent one. Unions of intervals and expressions with variables use math-verify, whose `evalf(chop=True)` comparison can still equate expressions that differ only by a tiny constant.
- Word-like references (`\text{Evelyn}`, `True`) parse into products of symbols and pass `gold_check`; the build must drop free-text references, and TheoremQA's boolean answers belong to kind `verdict`.
- math-verify decides from the string whether `i` is imaginary; `\sqrt{-1}` is parsed as a real expression and does not equal `i`.
- A timed-out job loses its compliance information (`complied` false). A program that sleeps or blocks is judged by the wall clock, which still depends on load, but only at five times its limit; when a process gets less than a fifth of a CPU the wall clock binds first. In a self-check at load average 180, two HumanEval+ and two MBPP+ references timed out that pass at load average 50, and one KodCode test failed that passed before (nothing else changed).
- Between jobs the supervisor resets processes, `/tmp` and System V IPC; it does not reset kernel keyrings, which a job could only fill through raw system calls, and which no later job reads unless written to.
- Not measured here: LeetCode (not in the cache yet) and the BBH, FOLIO, FLD and other unseen sources; their references go through `gold_check` at build time.
