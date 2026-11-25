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
| `code` | `{"harness": "pytest" \| "assert" \| "check", "tests", "reference", "entry_point" (check only), "time_limit_s" (default 10), "memory_mb" (default 1024)}` |

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

With wrapper `box` and channels `("direct", "program")`, the final answer is whichever ends last in the response: the last box or the last Python block. With `("program",)` only the block counts (the census's forced channel), with `("direct",)` only the box. A program runs in the sandbox with 5 s and 512 MiB; the last non-empty line of its standard output is verified like box content. A non-zero exit, a timeout or no output is incorrect, with `channel="program"` and the reason in `error`.

`correct_any` is true when the reward is, or when the answer ending last among all wrappers of the kind (and the program, if offered) is correct; for code, an untagged fenced block also counts. It is never "some answer somewhere was right", so a response hedging across two formats gains nothing.

## Equivalence of open answers

Content is cleaned first: surrounding `$`, `$$`, `\(\)`, `\[\]`, `**`, backticks and a whole-answer `\text{}`, `\textbf{}`, `\mathrm{}` or `\mathbf{}` are removed, as is a trailing period; `**` becomes `^`; e-notation becomes a power of ten when the mantissa has a point (`3.2e-19`) or when it is the whole answer (`1e-05`); a coefficient before `i` is spaced so that `3i` and `3 i` both mean the imaginary unit. Contents over 500 characters are not taken for one answer. Both sides are then parsed by math-verify's LaTeX parser as box content, with no free-text extraction and no fallback to strings (its expression extractor finds numbers anywhere in a text, which turned `x**2 + 1` into `1`). The parser's binary floats are turned back into the exact decimals they were read from.

Numbers, and tuples, sets, intervals and matrices of numbers, are compared by these rules (g the reference, a the answer, |.| the complex modulus):

| Case | Correct when |
|---|---|
| the reference contains a decimal literal | \|a − g\| < min(½ unit in the last significant digit of its least precise decimal literal, 10⁻³\|g\|) |
| exact reference, decimal answer | \|a − g\| ≤ 10⁻⁶\|g\| |
| both exact | equal to 40 significant digits (60-digit evaluation) |
| zero reference, decimal involved | \|a\| ≤ 10⁻¹² |

A percentage X% has the value X/100 and also matches X, as in math-verify. Containers must have the same type and shape; tuples compare in order, sets by a one-to-one matching, intervals by endpoints and openness. An answer `x = 5` or `2 + 3 = 5` stands for its last right-hand side when the reference is no equation, and a reference `x = 5` is answered by `5`; inequalities are never unwrapped. A number against anything with variables is wrong. Everything else (expressions with variables, equations, unions) goes to math-verify's `verify`, in strict mode.

math-verify's own numeric comparison is bypassed because it gives false credit: it rounds floats to six decimals in absolute terms (1e-7 equals 2e-7, 1.6e-19 equals 3.2e-19) and compares exact values with `evalf(chop=True)` (2^-60 equals 2^-61, 10^-20 equals 2·10^-20). The cap of 10⁻³ relative on decimal references exists because a short decimal such as `0.5` or `2.5` is usually exact (GSM8K, MATH), where half a unit in the last digit would accept 0.54 for 0.5.

## Choice, verdict and code

A choice answer is cleaned (math delimiters, `\text`-like commands, markdown emphasis, quotes, a trailing period) and read three ways: a capital option letter in its usual dressings (`C`, `(C)`, `[C]`, `C.`, `C)`, `C:`, `Option C`), a letter followed by that same option's exact text (`C. 42`), or the exact text of an option (case and spacing aside). It counts only if these readings name exactly one existing option, so `B or C`, `B. 43` with 43 under D, a lower-case `b`, or a text shared by two options name none. A verdict is correct when the cleaned answer equals exactly one label of the item's closed set, case aside.

For code, the program is saved read-only as `solution.py`, every top-level name it defines (underscored ones too) is imported ahead of the tests, and the harness runs in the sandbox with the item's time and memory limits:

- `pytest` (KodCode): the tests run as `test_solution.py`; pass means pytest exits 0 with at least one passed test, no failure and no skip or xfail.
- `assert` (MBPP+, whose `test` field is a script of assertions calling the function by name, and MBPP-style assertion lists): the tests run as a script.
- `check` (HumanEval+, LeetCode): the tests run as a script, then `check(<entry_point>)`.

Success is a random token the harness prints after the tests, together with exit status 0, so `os._exit(0)`, `sys.exit(0)`, `exit()` or killing itself never passes; all supplied files are read-only, so the tests cannot be rewritten. A program containing `__eq__` fails unless the reference contains it too, because an object whose `__eq__` always returns true would pass equality-based tests.

## Sandbox

`run_python` starts `setpriv --pdeathsig KILL bwrap ...` in its own session. The bwrap root is empty except for read-only binds of `/usr` (with the merged-usr symlinks `/bin`, `/lib`, `/lib64`, `/sbin`), the project's virtual environment (`sys.prefix`) and the directory of uv-managed interpreters its `python` links to (`sys.base_prefix`'s parent), a fresh `/proc`, bwrap's minimal `/dev`, and a 64 MiB tmpfs at `/tmp`, the working directory. The root and `/dev`, which bwrap builds as uncapped tmpfs mounts, are remounted read-only, so `/tmp` is the only writable place. The repository, `.env`, home directories, `/etc`, other users' data, host `/tmp` and host sockets do not exist inside; a bind that would expose the repository root, `/home` or the home directory as a whole is refused. All namespaces are unshared (network: only a private loopback), the environment is replaced by a fixed one (`HOME=/tmp`, UTF-8, `PYTHONHASHSEED=0`, one BLAS thread, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`), and capabilities are dropped. The program and its files are passed as memfds and mounted read-only (`--ro-bind-data`); stdout and stderr are memfds.

Before the program runs, a bootstrap sets hard limits that it cannot raise: address space `memory_mb`, CPU time ceil(`timeout_s`) s, 8 MiB per file (which also caps stdout and stderr), 32 processes plus threads (counted per user namespace), no core dumps. It then writes one byte on a pipe, and the program's wall-clock limit counts from that byte: `timeout_s` plus 2 s for exit and teardown. The wait before it, for bwrap's mount setup, has its own limit of 60 s, because the kernel serializes that setup machine-wide: about 30 sandboxes start per second on this machine whatever the concurrency (bare user and mount namespaces from `unshare` reach over 1,000 per second, so the cost is bwrap's own mounts), and under load from other users it fell to about 6 per second. With the clock started at launch instead, references that take 0.2 s alone timed out at 10 s in such a period; with 48 sandboxes launched at once, setup waits about a second, and a test checks that none of them times out.

At the limit bwrap's process group gets SIGKILL; its child, the pid namespace's init, dies with it (`--die-with-parent`), and the kernel then kills everything left in the namespace, including detached, re-sessioned grandchildren. Because bwrap runs under `setpriv --pdeathsig KILL`, killing the caller (the pool does this to a stuck verifier process) tears the sandbox down too. A program killed by signal N returns 128 + N (bwrap's convention); one that never started within 60 s counts as timed out.

## Pool

`verify_many` keeps a persistent pool of `PROCESSES = 32` verifier processes per calling process (at most 48 on this shared machine): fresh interpreters started with `python -c`, not forks of the caller. Each judges one job at a time, under a 4 GiB address-space limit, and dies with its parent. A job's deadline is `timeout` if given, else 20 s plus the worst case of its sandbox runs (one program with the program channel; two test runs for code, because the tolerant extraction may run a second program), each counted as its time limit plus the 62 s of setup and exit allowance. Those long deadlines are only reached by a hang outside the sandbox, which only parsing can produce. Past it the process is killed and replaced and the job gets `Verdict(0, False, False, None, None, "timeout")`; a process that dies gives error `verifier process died`, and an exception inside the judgment (for instance a reference that does not parse) gives reward 0 with the exception as error. This is what bounds sympy stuck in C code: a boxed `10^{10^{10}}` cannot be interrupted by any signal. The pool is not meant to be used from several threads at once.

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
| kodcode | code | 296/300 | |
| humaneval_plus | code | 163/164 | |
| mbpp_plus | code | 298/300 | |

Perturbations are off by one either way, sign flipped and doubled for numbers (`g + 1` and `2g` for expressions), one unit more in the last digit for decimals, another letter and another option's text for choice, every other label for verdict. The three DeepScaleR failures are malformed references (display math with a box inside, an empty string, `33\frac{1}{3}\%`). The code failures are references that cannot pass here: two KodCode items import torch, whose libraries cannot even be mapped within 1 GiB of address space (40 of KodCode's 10,000 items import torch), one needs seaborn (57 items; not installed), one asserts a timing; HumanEval/32's dataset harness is itself broken (`_poly(*candidate(*inp), inp)`); of MBPP+, `sum_average` takes 12 s even alone and `combinations_colors` exceeds 1 GiB on its largest input. Evaluation sources therefore want per-item time and memory limits with room, set at build time.

Cost per job when run alone: an open answer about 4 ms (cold, one process), choice and verdict well under 1 ms, a sandboxed program 0.09 s, a code item 0.2–0.3 s. Throughput with 32 processes (final run, load average about 48):

| Workload | Jobs per second |
|---|---|
| open, box (answers rewritten by sympy) | 3,600 |
| open, program channel | 28 |
| choice | 17,000 |
| verdict | 23,000 |
| code (KodCode references) | 16 |

Anything that runs a sandbox is bound by the machine-wide start rate of about 30 per second. A training step of 256 rollouts with about 100 sandbox runs therefore costs a few seconds of verification; the census's roughly 150,000 sandbox runs take one to two hours.

## Known limitations

- The `__eq__` rule is textual and catches only the plain always-equal object; a program built to defeat the harness (reading the token from `/tmp/main.py`, monkeypatching pytest from the solution, a `__bool__` or `__len__` trick on tests that check truth only, `__eq__` assembled at run time) can still pass. All of these need deliberate adversarial code; the census audit and record inspection should watch for them.
- Network isolation is by namespace: a program can create a socket, but there is nothing to connect to except its own loopback.
- Decimal references are ambiguous between exact and rounded values. With the 10⁻³ cap, a more precise answer to a reference with one or two significant digits (TheoremQA `0.9` for 0.8962) is rejected; an exact reference wants six significant digits from a decimal answer (`0.7071` for √2/2 is rejected).
- Structures must match in type: `1, 2` against `(1, 2)` is wrong, although math-verify would accept it. Unions of intervals and expressions with variables use math-verify, whose `evalf(chop=True)` comparison can still equate expressions that differ only by a tiny constant.
- Word-like references (`\text{Evelyn}`, `True`) parse into products of symbols and pass `gold_check`; the build must drop free-text references, and TheoremQA's boolean answers belong to kind `verdict`.
- math-verify decides from the string whether `i` is imaginary; `\sqrt{-1}` is parsed as a real expression and does not equal `i`.
- A timed-out job loses its compliance information (`complied` false). The ceiling of about 30 sandbox starts per second is shared by every caller on the machine and drops under other users' load; setup waits do not count against a program, but CPU contention still does, so a program close to its limit can time out on a busy machine.
- Not measured here: LeetCode (not in the cache yet) and the BBH, FOLIO, FLD and other unseen sources; their references go through `gold_check` at build time.
