"""Verifier behaviour: false credit is the error that matters most, so most cases here are answers that must be
rejected (near misses, hedges, the wrong wrapper, an answer only in the reasoning, programs that cheat or hang)."""

import time

import pytest

from evotaste.verify import (HELDOUT_WRAPPERS, TRAIN_WRAPPERS, WRAPPERS, Job, Verdict, gold_check, gold_ok,
                             verify_many)
from evotaste.verify.equivalence import equivalent
from evotaste.verify import judge as judge_module
from evotaste.verify.judge import judge, resolve_choice, resolve_label
from evotaste.verify.pool import Pool
from evotaste.verify.sandbox import EXIT_S, WALL_FACTOR
from evotaste.verify.wrappers import ANSWER_KINDS, for_kind

BOTH = ("direct", "program")


def open_job(text, gold, wrapper="box", channels=("direct",)):
    return judge(Job("open", text, gold, wrapper, channels))


# -- equivalence of open answers ------------------------------------------------------------------------------------

@pytest.mark.parametrize("gold,answer", [
    ("\\frac{1}{2}", "0.5"), ("0.5", "\\frac{1}{2}"), ("0.5", "1/2"), ("17", "17.0"), ("17", "16.999999999999996"),
    ("18", "\\$18"), ("12", "12 \\text{ cm}"), ("5", "x = 5"), ("5", "2 + 3"), ("x^2+2x+1", "(x+1)^2"),
    ("x^2+1", "x**2 + 1"), ("\\frac{\\sqrt{2}}{2}", "0.7071067811865476"), ("\\frac{\\sqrt{2}}{2}", "sqrt(2)/2"),
    ("2\\sqrt{3}", "\\sqrt{12}"), ("\\{1,2\\}", "\\{2,1\\}"), ("(1,2)", "(1, 2)"), ("[0,1)", "[0, 1)"),
    ("50", "50\\%"), ("0.125", "12.5\\%"), ("3.2\\times 10^{-19}", "3.2e-19"), ("10", "1e1"), ("0", "1e-15"),
    ("2^{-60}", "\\frac{1}{1152921504606846976}"), ("3.14", "\\pi"), ("1.98", "1.9786"), ("1,234.5", "1234.5"),
    ("3+4i", "3 + 4i"), ("-2+3i", "3 i - 2"), ("\\mathbf{2340}", "2340"),
    ("y=2x+1", "y=1+2x"), ("x=3", "3"), ("\\infty", "\\infty"),
    # exact or more precise answers to rounded decimal references
    ("1.33", "\\frac{4}{3}"), ("1.33", "1.3333"), ("0.67", "\\frac{2}{3}"), ("0.9", "0.8962"), ("0.5", "0.504"),
    # comma lists and tuples
    ("1, 2", "2, 1"), ("1, 2", "1,2"), ("(1, 2)", "1, 2"), ("1, 2", "(1, 2)"), ("(1, 2)", "\\left( 1, 2 \\right)"),
    ("-1, \\frac{1}{2}, 3", "3, -1, 0.5"), ("(0.5, 3)", "(\\frac{1}{2}, 3)"), ("1, 1, 2", "1, 2, 1"),
    ("1,000", "1000"), ("10,\\!000", "10000"),
])
def test_equal_values_in_other_notation_are_accepted(gold, answer):
    assert equivalent(answer, gold)


@pytest.mark.parametrize("gold,answer", [
    # near-miss numbers
    ("17", "18"), ("17", "-17"), ("17", "17.1"), ("1000", "1000.004"), ("0.5", "0.51"), ("0.5", "0.6"),
    ("3.14", "3.15"), ("3.14", "3.145"), ("3.14", "3.1"), ("0.0333", "0.0334"), ("1234.5", "1234.55"),
    ("1.33", "1.34"), ("0.9", "0.94"), ("0.9", "0.86"),  # the last stated digit; the 2% cap
    ("\\frac{1}{3}", "0.333"), ("\\pi", "3.14"), ("\\frac{\\sqrt{2}}{2}", "0.7071"),
    # tiny values: math-verify's own comparison accepts all of these
    ("0.0000001", "0.0000002"), ("3.2\\times 10^{-19}", "1.6\\times 10^{-19}"), ("3.2e-19", "1.6e-19"),
    ("10^{-20}", "2\\cdot 10^{-20}"), ("2^{-60}", "2^{-61}"), ("\\frac{1}{10^{20}}", "\\frac{2}{10^{20}}"),
    ("0", "0.0000001"),
    # hedges and several answers
    ("2", "2 \\text{ or } 3"), ("2", "2, 3"), ("2", "\\pm 2"), ("2", "x = 2 \\text{ or } x = 3"), ("3", "3 or 4"),
    # wrong structure or form
    ("5", "x > 5"), ("(1,2)", "(2,1)"), ("[0,1)", "[0,1]"), ("1", "x^2 + 1"), ("x+1", "y+1"), ("3+4i", "3-4i"),
    ("\\infty", "-\\infty"), ("2", "The answer is 2"), ("2", ""),
    # comma lists and tuples: order of a tuple, missing or extra elements, brackets of intervals
    ("(1, 2)", "(2, 1)"), ("(1, 2)", "2, 1"), ("1, 2", "(2, 1)"), ("1, 2", "1"), ("1, 2", "1, 2, 3"),
    ("1, 2, 3", "1, 2"), ("1, 1, 2", "1, 2, 2"), ("(1, 2, 3)", "(1, 2)"), ("[1, 2]", "1, 2"), ("[1, 2)", "(1, 2)"),
    ("(1, 2]", "1, 2"), ("1, 2", "[1, 2]"), ("1, 2", "1, 3"), ("1000", "1, 000"),
])
def test_wrong_answers_are_rejected(gold, answer):
    assert not equivalent(answer, gold)


# -- wrappers -------------------------------------------------------------------------------------------------------

def test_wrapper_sets():
    assert len(TRAIN_WRAPPERS) == 6 and len(HELDOUT_WRAPPERS) == 3 and "answer_tag" in HELDOUT_WRAPPERS
    assert not set(TRAIN_WRAPPERS) & set(HELDOUT_WRAPPERS) and "box" not in TRAIN_WRAPPERS + HELDOUT_WRAPPERS
    assert all(ANSWER_KINDS == WRAPPERS[w].kinds for w in ("box", *TRAIN_WRAPPERS, *HELDOUT_WRAPPERS))


@pytest.mark.parametrize("content", ["42", "\\frac{1}{2}", "C", "True", "(1, 2)", "x^{2} + 1"])
def test_wrappers_extract_their_own_form_and_no_other(content):
    for w in for_kind("open"):
        text = f"Some reasoning: 1 + 1 = 2.\n\n{w.render(content)}"
        assert w.find(text).content == content, w.id
        others = [v.id for v in for_kind("open") if v.id != w.id and v.find(text) is not None]
        assert others == [], f"complying with {w.id} also complies with {others}"


def test_a_box_inside_another_wrapper_is_not_compliance():
    for w in ("final_line", "answer_tag", "hash", "sentence", "json", "at"):
        text = WRAPPERS[w].render("\\boxed{5}")
        assert WRAPPERS[w].find(text) is None, w
        v = open_job(text, "5", wrapper=w)
        assert (v.reward, v.complied, v.correct_any) == (0.0, False, True), w


def test_last_occurrence_counts_and_an_unclosed_one_is_not_replaced_by_an_earlier_one():
    assert WRAPPERS["box"].find("\\boxed{1} then \\boxed{\\frac{2}{3}} done").content == "\\frac{2}{3}"
    assert WRAPPERS["box"].find("\\boxed{1} then \\boxed{2") is None
    assert WRAPPERS["final_line"].find("Final answer: 3\nwait, no") is None


# -- open answers: wrapper asked for, tolerant extraction, channels ---------------------------------------------------

def test_right_value_in_the_wrong_wrapper_earns_nothing_but_counts_as_correct_any():
    v = open_job("So the answer is \\boxed{7}.", "7", wrapper="final_line")
    assert (v.reward, v.complied, v.correct_any, v.channel) == (0.0, False, True, None)
    v = open_job("Final answer: 7", "7", wrapper="final_line")
    assert (v.reward, v.complied, v.correct_any, v.extracted) == (1.0, True, True, "7")


def test_an_answer_only_in_the_reasoning_earns_nothing():
    v = open_job("Adding gives 7, so the answer is 7.", "7")
    assert (v.reward, v.complied, v.correct_any) == (0.0, False, False)


def test_tolerant_extraction_takes_the_last_answer_not_any_answer():
    v = open_job("\\boxed{6}\n\nFinal answer: 7", "6")  # the box was superseded by a later answer
    assert (v.reward, v.correct_any) == (1.0, True)  # the box is the wrapper asked for, so its answer is the reward's
    v = open_job("\\boxed{6}\n\nFinal answer: 7", "7")
    assert (v.reward, v.correct_any) == (0.0, True)
    v = open_job("<answer>6</answer>\n\nFinal answer: 7", "6")
    assert (v.reward, v.complied, v.correct_any) == (0.0, False, False)


def test_a_boxed_answer_takes_precedence_over_any_program():
    silent = "```python\nx = 3 + 4\n```"
    printing = "```python\nprint(3 + 4)\n```"
    v = open_job(f"So \\boxed{{7}}.\nCheck:\n{silent}", "7", channels=BOTH)  # box, then a silent program
    assert (v.reward, v.channel, v.extracted, v.correct_any) == (1.0, "direct", "7", True)
    v = open_job(f"{printing}\nIt prints 7, so \\boxed{{8}}.", "7", channels=BOTH)  # program, then box: the box
    assert (v.reward, v.channel, v.extracted, v.correct_any) == (0.0, "direct", "8", False)
    v = open_job(f"\\boxed{{8}}\n{printing}", "7", channels=BOTH)  # a box before a correct program still decides
    assert (v.reward, v.channel, v.correct_any) == (0.0, "direct", False)
    v = open_job(f"Compute it.\n{printing}", "7", channels=BOTH)  # no box, a printing program
    assert (v.reward, v.channel, v.extracted) == (1.0, "program", "7")
    v = open_job(f"Compute it.\n{silent}", "7", channels=BOTH)  # no box, a silent program
    assert (v.reward, v.channel, v.complied, v.correct_any, v.error) == (0.0, "program", True, False,
                                                                         "program printed nothing")
    v = open_job(f"<answer>8</answer>\n\n{printing}", "8", channels=BOTH)  # for correct_any, any wrapper's answer
    assert (v.reward, v.channel, v.correct_any) == (0.0, "program", True)  # takes precedence over the program


def test_with_only_the_program_channel_boxes_are_ignored():
    printing = "```python\nprint(3 + 4)\n```"
    v = open_job(f"\\boxed{{8}}\n{printing}", "7", channels=("program",))
    assert (v.reward, v.channel, v.extracted) == (1.0, "program", "7")
    v = open_job("\\boxed{7}", "7", channels=("program",))
    assert (v.reward, v.complied, v.correct_any) == (0.0, False, True)
    v = open_job(f"\\boxed{{8}}\n{printing}", "7", channels=("direct",))  # program not offered: the box counts
    assert (v.reward, v.channel) == (0.0, "direct")


def test_program_must_print_the_answer_on_its_last_line():
    v = open_job("```python\nprint(7)\nprint('done')\n```", "7", channels=BOTH)
    assert (v.reward, v.channel, v.extracted) == (0.0, "program", "done")
    v = open_job("```python\nprint('The answer is')\nprint(7)\n```", "7", channels=BOTH)
    assert v.reward == 1.0


@pytest.mark.parametrize("program,error", [
    ("print(7)\nraise SystemExit(1)", "program exit 1"),
    ("print(7)\n1/0", "program exit 1"),
    ("x = 7", "program printed nothing"),
    ("import time\nprint(7, flush=True)\ntime.sleep(30)", "program timeout"),
    ("import os\nprint(7, flush=True)\nwhile True:\n    try:\n        os.fork()\n    except OSError:\n        pass",
     "program "),
])
def test_failing_hanging_or_forking_programs_are_incorrect(program, error, monkeypatch):
    monkeypatch.setattr(judge_module, "PROGRAM_TIMEOUT_S", 1)  # 1 s of CPU, 5 s of wall clock
    t0 = time.monotonic()
    v = open_job(f"```python\n{program}\n```", "7", channels=BOTH)
    assert (v.reward, v.correct_any, v.channel) == (0.0, False, "program")
    assert v.error.startswith(error) and time.monotonic() - t0 < WALL_FACTOR * 1 + EXIT_S + 3


# -- choice and verdict ---------------------------------------------------------------------------------------------

OPTIONS = ["12", "15", "the empty set", "43"]


@pytest.mark.parametrize("content", ["B", "(B)", "[B]", "B.", "B)", "B:", "Option B", "option (B)", "\\text{B}",
                                     "\\textbf{(B)}", "**B**", "$B$", "B. 15", "(B) 15", "B: 15", "15", " B "])
def test_choice_letter_dressings_and_exact_text_resolve(content):
    assert resolve_choice(content, OPTIONS) == 1


@pytest.mark.parametrize("content", ["B or C", "B, C", "B and C", "b", "E", "B. 43", "B 12", "Option", "1",
                                     "15 or 12", "The answer is B", "the empty", ""])
def test_choice_answers_naming_no_single_option_are_rejected(content):
    assert resolve_choice(content, OPTIONS) is None


def test_choice_text_shared_by_two_options_names_none():
    assert resolve_choice("7", ["7", "7", "8"]) is None and resolve_choice("A", ["7", "7", "8"]) == 0


def test_choice_job():
    gold = {"options": OPTIONS, "answer": "B"}
    assert judge(Job("choice", "Thus \\boxed{B}", gold, "box")).reward == 1.0
    assert judge(Job("choice", "Thus \\boxed{C}", gold, "box")).reward == 0.0
    v = judge(Job("choice", "Thus \\boxed{B}", gold, "answer_tag"))
    assert (v.reward, v.complied, v.correct_any) == (0.0, False, True)


def test_verdict_labels():
    labels = ["True", "False", "Unknown"]
    assert resolve_label("\\text{unknown}", labels) == "Unknown" and resolve_label("**True**.", labels) == "True"
    for bad in ("Not true", "Yes", "True or False", "Uncertain", ""):
        assert resolve_label(bad, labels) is None
    gold = {"labels": labels, "answer": "False"}
    assert judge(Job("verdict", '{"answer": "False"}', gold, "json")).reward == 1.0
    assert judge(Job("verdict", '{"answer": "True"}', gold, "json")).reward == 0.0


# -- code -----------------------------------------------------------------------------------------------------------

REF = "def add(a, b):\n    return a + b\n"
PYTEST = {"harness": "pytest", "tests": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
          "\ndef test_neg():\n    assert add(-1, -2) == -3\n", "reference": REF, "time_limit_s": 10}
ASSERT = {"harness": "assert", "tests": "assert add(1, 2) == 3\nassert add(0, 0) == 0\n", "reference": REF}
CHECK = {"harness": "check", "entry_point": "add", "reference": REF,
         "tests": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(2, 2) == 4\n"}


def code_job(program, gold, tag="python"):
    return judge(Job("code", f"Here it is.\n```{tag}\n{program}\n```", gold, "code"))


@pytest.mark.parametrize("gold", [PYTEST, ASSERT, CHECK])
def test_a_correct_program_passes_and_a_wrong_one_fails(gold):
    assert code_job(REF, gold).reward == 1.0
    v = code_job("def add(a, b):\n    return a - b\n", gold)
    assert v.reward == 0.0 and v.error.startswith("tests failed")


@pytest.mark.parametrize("cheat", [
    "import os\nos._exit(0)",
    "import sys\nsys.exit(0)",
    "raise SystemExit(0)",
    "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)",
    "print('passed')\nexit()",
    "import pytest\npytest.skip('hard', allow_module_level=True)",
    "class A:\n    def __eq__(self, other):\n        return True\ndef add(a, b):\n    return A()",
    "def add(a, b):\n    import pytest\n    pytest.skip('hard')",
    "open('test_solution.py', 'w').write('def test_ok():\\n    pass\\n')\ndef add(a, b):\n    return 0",
])
@pytest.mark.parametrize("gold", [PYTEST, ASSERT, CHECK])
def test_programs_that_exit_skip_or_tamper_never_pass(cheat, gold):
    assert code_job(cheat, gold).reward == 0.0


def test_slow_program_fails_at_the_item_time_limit():
    t0 = time.monotonic()
    v = code_job("import time\ntime.sleep(30)\n" + REF, {**ASSERT, "time_limit_s": 1})
    assert (v.reward, v.error) == (0.0, "timeout") and time.monotonic() - t0 < WALL_FACTOR * 1 + EXIT_S + 3


def test_a_prelude_is_part_of_the_solution_file_but_not_of_the_model_code():
    prelude = ("from typing import *\n\nclass ListNode:\n    def __init__(self, val=0, next=None):\n"
               "        self.val, self.next = val, next\n    def __eq__(self, other):\n"
               "        return isinstance(other, ListNode) and (self.val, self.next) == (other.val, other.next)\n")
    reference = ("class Solution:\n    def first(self, head: Optional[ListNode]) -> List[int]:\n"
                 "        return [head.val] if head else []\n")
    tests = "def check(candidate):\n    assert candidate(ListNode(4, ListNode(5))) == [4]\n" \
            "    assert ListNode(1) == ListNode(1)\n"
    gold = {"harness": "check", "entry_point": "Solution().first", "tests": tests, "reference": reference,
            "prelude": prelude}
    assert code_job(reference, gold).reward == 1.0 and gold_ok("code", gold)
    without = {k: v for k, v in gold.items() if k != "prelude"}
    assert code_job(reference, without).error.startswith("tests failed") and not gold_ok("code", without)
    cheat = reference.replace("return [head.val] if head else []",
                              "class A:\n            def __eq__(self, o):\n                return True\n"
                              "        return A()")
    assert code_job(cheat, gold).error == "program defines __eq__"  # the prelude's __eq__ does not license it


def test_code_needs_a_python_tagged_block_for_the_reward():
    v = code_job(REF, CHECK, tag="")
    assert (v.reward, v.complied, v.correct_any) == (0.0, False, True)


# -- the pool, gold checks, malformed jobs ---------------------------------------------------------------------------

def test_verify_many_keeps_order_and_matches_judge():
    jobs = [Job("open", f"\\boxed{{{i}}}", "3", "box") for i in range(6)]
    assert [v.reward for v in verify_many(jobs)] == [0, 0, 0, 1, 0, 0]


def test_a_job_stuck_in_c_code_is_killed_at_its_deadline():
    pool = Pool(2)
    try:
        pool.run([Job("open", "\\boxed{1}", "1", "box")] * 2)  # start both processes (imports take seconds)
        t0 = time.monotonic()
        out = pool.run([Job("open", "\\boxed{10^{10^{10}}}", "1", "box"), Job("open", "\\boxed{1}", "1", "box")],
                       timeout=3)
        assert out[0] == Verdict(0.0, False, False, None, None, "timeout") and out[1].reward == 1.0
        assert time.monotonic() - t0 < 8
        assert pool.run([Job("open", "\\boxed{2}", "2", "box")])[0].reward == 1.0  # the replacement works
    finally:
        pool.close()


def test_malformed_jobs_are_refused():
    for job in [Job("open", "x", "1", "code"), Job("code", "x", CHECK, "box"), Job("riddle", "x", "1", "box"),
                Job("open", "x", "1", "final_line", BOTH), Job("choice", "x", {}, "box", BOTH),
                Job("open", "x", "1", "box", ("program", "magic"))]:
        with pytest.raises(ValueError):
            verify_many([job])


def test_gold_check():
    items = [("open", "\\frac{1}{2}"), ("open", "\\frac{1}{"), ("choice", {"options": OPTIONS, "answer": "D"}),
             ("verdict", {"labels": ["True", "False"], "answer": "False"}), ("code", PYTEST),
             ("code", {**PYTEST, "reference": "def add(a, b):\n    return a - b\n"})]
    out = gold_check(items)
    assert out[0] is None and out[1] is not None and out[2] is None and out[3] is None and out[4] is None
    assert out[5].startswith("wrapper code: tests failed")
    assert gold_ok("open", "18") and not gold_ok("open", "")
    with pytest.raises(ValueError):
        gold_check([("choice", {"options": OPTIONS, "answer": "E"})])
