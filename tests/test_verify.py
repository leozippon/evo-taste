"""Verifier behaviour: false credit is the error that matters most, so most cases here are answers that must be
rejected (near misses, hedges, the wrong wrapper, an answer only in the reasoning, programs that cheat or hang)."""

import time

import pytest

from evotaste.verify import (HELDOUT_WRAPPERS, RULES_VERSION, TRAIN_WRAPPERS, WRAPPERS, Job, Verdict, gold_check,
                             gold_ok, is_undecided, verify_many)
from evotaste.verify.equivalence import equivalent
from evotaste.verify import judge as judge_module
from evotaste.verify.judge import judge, resolve_choice, resolve_label
from evotaste.verify.pool import COMPARISON_CPU_S, Pool
from evotaste.verify.sandbox import EXIT_S, WALL_FACTOR
from evotaste.verify.wrappers import ANSWER_KINDS, for_kind

BOTH = ("direct", "program")


def open_job(text, gold, wrapper="box", channels=("direct",)):
    return judge(Job("open", text, gold, wrapper, channels))


# -- equivalence of open answers ------------------------------------------------------------------------------------

@pytest.mark.parametrize("gold,answer", [
    ("\\frac{1}{2}", "0.5"), ("0.5", "\\frac{1}{2}"), ("0.5", "1/2"), ("17", "17.0"), ("17", "16.999999999999996"),
    ("18", "\\$18"), ("12", "12 \\text{ cm}"), ("5", "x = 5"), ("5", "2 + 3 = 5"), ("x^2+2x+1", "(x+1)^2"),
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
    # exact values written differently
    ("\\frac{1}{\\sqrt{2}}", "\\frac{\\sqrt{2}}{2}"), ("\\sqrt{5+2\\sqrt{6}}", "\\sqrt{2}+\\sqrt{3}"),
    ("2^{10}", "1024"), ("5!", "120"), ("\\binom{5}{2}", "10"), ("\\log_2 8", "3"),
    ("\\frac{\\pi}{2}", "\\frac{2\\pi}{4}"), ("\\sqrt[3]{8}", "2"),
    ("3+2\\sqrt{2}", "(1+\\sqrt{2})^2"), ("4^{1006}", "2^{2012}"), ("100\\cdot 99!", "100!"), ("2i", "(1+i)^2"),
    ("\\frac{2012\\cdot 2011}{2}", "\\binom{2012}{2}"), ("\\sqrt{2}-1", "\\frac{1}{1+\\sqrt{2}}"), ("e^{2\\ln 3}", "9"),
])
def test_equal_values_in_other_notation_are_accepted(gold, answer):
    assert equivalent(answer, gold)


@pytest.mark.parametrize("gold,answer", [
    # near-miss numbers
    ("17", "18"), ("17", "-17"), ("17", "17.1"), ("1000", "1000.004"), ("0.5", "0.51"), ("0.5", "0.6"),
    ("3.14", "3.15"), ("3.14", "3.145"), ("3.14", "3.1"), ("0.0333", "0.0334"), ("1234.5", "1234.55"),
    ("1.33", "1.34"), ("0.9", "0.94"), ("0.9", "0.86"),  # the last stated digit; the 2% cap
    ("\\frac{1}{3}", "0.333"), ("\\pi", "3.14"), ("\\frac{\\sqrt{2}}{2}", "0.7071"),
    # huge or tiny exact values that differ (audit D1, census false credits)
    ("2^{2012}", "2^{2012}-1"), ("100!", "100!+1"), ("\\frac{2^{2012}+1}{3}", "\\frac{2^{2012}-1}{3}"),
    ("1+\\frac{1}{2^{2020}}", "1"), ("2^{2014}", "2^{2014} - 1"), ("(2^{2003}-2)/3", "\\frac{2^{2003}+1}{3}"),
    ("4^{2017}-2 \\cdot 3^{2017}+2^{2017}", "2^{4034} - 2\\cdot 3^{2017}"),
    ("1-(1/2)^{2016}", "1 + \\frac{\\ln 2}{1-\\ln 2}2^{-2016}"),
    ("\\sqrt{2}\\cdot 2^{2012}", "\\sqrt{2}\\cdot 2^{2012}+1"),
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


# A reference written as a number wants the arithmetic carried out: under training, an expression whose value is
# right would be a shortcut past the final computation.
@pytest.mark.parametrize("gold,answer", [
    ("395", "17 \\cdot 23 + 4"), ("5", "2 + 3"), ("395", "x = 17 \\cdot 23 + 4"), ("4086462", "2021 \\times 2022"),
    ("1024", "2^{10}"), ("120", "5!"), ("10", "\\binom{5}{2}"), ("3", "\\log_2 8"), ("2", "\\sqrt{4}"),
    ("2", "\\sqrt[3]{8}"), ("9", "e^{2\\ln 3}"), ("2", "4 \\div 2"), ("2", "0.5 \\cdot 4"), ("18", "\\$3 \\cdot 6"),
    ("\\frac{3}{4}", "\\frac{1}{2} + \\frac{1}{4}"), ("-\\frac{2}{39203}", "-\\frac{2}{199 \\cdot 197}"),
    ("900900", "\\frac{16!}{4! \\cdot 4! \\cdot 8!}"), ("90", "\\int_{0}^{6} 15 \\, dy"),
    ("0.0000001024", "\\frac{1}{5^{10}}"), ("1.33", "1 + \\frac{1}{3}"), ("50\\%", "25 \\cdot 2\\%"),
])
def test_an_expression_still_to_be_evaluated_does_not_answer_a_number(gold, answer):
    assert not equivalent(answer, gold)


def test_an_unevaluated_element_or_printed_expression_does_not_answer_a_number():
    assert not equivalent("17 \\cdot 23 + 4, 2", "2, 395") and equivalent("395, 2", "2, 395")
    assert open_job("\\boxed{(17 \\cdot 23 + 4, 2)}", {"answer": "(395, 2)", "type": "tuple"}).reward == 0.0
    v = open_job("```python\nprint('17*23+4')\n```", "395", channels=BOTH)
    assert (v.channel, v.reward) == ("program", 0.0)


@pytest.mark.parametrize("gold,answer", [
    ("0.75", "\\frac{3}{4}"), ("\\frac{3}{4}", "3/4"), ("\\frac{3}{4}", "\\dfrac{6}{8}"), ("0.75", "\\tfrac34"),
    ("3.5", "3\\frac{1}{2}"), ("-0.75", "-\\frac{3}{4}"), ("-3", "(-3)"), ("300000", "3 \\times 10^{5}"),
    ("3.2\\times 10^{-19}", "3.2 \\cdot 10^{-19}"), ("0.00001", "1e-05"), ("1234567", "1,234,567"),
    ("10000", "10,\\!000"), ("0.5", "50\\%"), ("18", "\\$18.00"), ("12", "12 \\text{ cm}"), ("30", "30^\\circ"),
    ("2003", "2003_6"), ("3:7", "3 : 7"), ("5", "2 + 3 = 5"),
    # a decimal reference stands for a rounded value: an exact irrational form still answers it
    ("3.14", "\\pi"), ("1.618", "\\frac{1 + \\sqrt{5}}{2}"), ("1.41", "\\sqrt{2}"), ("3.7", "\\log_2 13"),
])
def test_a_number_in_final_form_answers_a_number(gold, answer):
    assert equivalent(answer, gold)


@pytest.mark.parametrize("gold,answer", [("2^{10}", "4^{5}"), ("5!", "2 \\cdot 3 \\cdot 4 \\cdot 5"),
                                         ("\\sqrt{12}", "2\\sqrt{3}")])
def test_a_reference_written_as_an_expression_takes_any_rewriting(gold, answer):
    assert equivalent(answer, gold)


# Colon notation (a time or a ratio) and lists of statement labels are matched as written: 6:00 is no 6/0 (so every
# h:00 was equal), (2)(3) no sum (so 5 and (1)(4) answered it).
@pytest.mark.parametrize("gold,answer,ok", [
    ("6:00", "6:00", True), ("6:00", "06:00", True), ("05\\!:\\!00", "5:00", True), ("6:00", "4:00", False),
    ("6:00", "6", False), ("3:30 PM", "3:30 \\text{ PM}", True), ("4:40 a.m.", "4:40 AM", True),
    ("3:30 PM", "1:10 PM", False), ("10:40", "\\frac{1}{4}", False), ("4 : 13", "AB : BD = 4 : 13", True),
    ("4 : 13", "8 : 26", False), ("3 : 4 : 5", "3 : 20", False),
    # a two-term ratio whose second term is no minutes keeps its value
    ("2 : 5", "\\frac{2}{5}", True), ("2: 1", "2", True), ("2 : 5", "4 : 10", True), ("2 : 5", "5 : 2", False),
    ("(2)(3)", "(2)(3)", True), ("(2)(3)", "(2), (3)", True), ("(2)(3)", "2, 3", True), ("(3)", "3", True),
    ("(2)(3)", "5", False), ("(2)(3)", "(1)(4)", False), ("(2)(3)", "(2)", False), ("(3)", "3.0", False),
])
def test_times_ratios_and_statement_labels_are_matched_as_written(gold, answer, ok):
    assert equivalent(answer, gold) == ok


@pytest.mark.parametrize("gold,answer", [("1000", "1\\,000"), ("1234567", "1\\,234\\,567"), ("1000.5", "1\\,000.5")])
def test_thin_spaces_between_digit_groups_are_thousands_separators(gold, answer):
    assert equivalent(answer, gold) and not equivalent(answer, "0")


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
    ("print(7, flush=True)\nwhile True:\n    pass", "program timeout"),  # the CPU limit decides
    ("import os\nprint(7, flush=True)\nwhile True:\n    try:\n        os.fork()\n    except OSError:\n        pass",
     "program timeout"),
    ("import time\nprint(7, flush=True)\ntime.sleep(30)", "program timeout"),  # slept: decided
])
def test_failing_hanging_or_forking_programs_are_incorrect(program, error, monkeypatch):
    monkeypatch.setattr(judge_module, "PROGRAM_TIMEOUT_S", 0.25)  # 1 s of CPU, 5 s of wall clock
    t0 = time.monotonic()
    v = open_job(f"```python\n{program}\n```", "7", channels=BOTH)
    assert (v.reward, v.correct_any, v.channel) == (0.0, False, "program")
    assert v.error.startswith(error) and time.monotonic() - t0 < WALL_FACTOR * 0.25 + EXIT_S + 5
    assert is_undecided(v.error) == error.startswith("undecided")


@pytest.mark.parametrize("output,gold", [("pi/4", "\\frac{\\pi}{4}"), ("3*pi", "3\\pi"), ("E", "e"), ("exp(1)", "e"),
                                         ("2 + 3*I", "2+3i"), ("oo", "\\infty"), ("sqrt(2)/2", "\\frac{\\sqrt2}{2}"),
                                         ("1/3", "\\frac{1}{3}"), ("log(2)", "\\ln 2"), ("[-1, 3]", "3, -1"),
                                         ("0.3333333333333333", "\\frac13"), ("x**2 + 1", "x^2+1")])
def test_a_printed_python_or_sympy_value_is_read_as_such(output, gold):
    v = open_job(f"```python\nprint({output!r})\n```", gold, channels=BOTH)
    assert (v.reward, v.channel, v.extracted) == (1.0, "program", output)


@pytest.mark.parametrize("output", ["The answer is 4.", "Louise needs to bake 4 more cakes.", "4 cakes",
                                    "__import__('os').system('true')", "(4).real", "[4][0]", "2**2**2**2**2"])
def test_a_sentence_or_code_printed_by_a_program_is_not_an_answer(output):
    assert open_job(f"```python\nprint({output!r})\n```", "4", channels=BOTH).reward == 0.0


# -- choice and verdict ---------------------------------------------------------------------------------------------

OPTIONS = ["12", "15", "the empty set", "43"]


@pytest.mark.parametrize("content", ["B", "(B)", "[B]", "B.", "B)", "B:", "Option B", "option (B)", "\\text{B}",
                                     "\\textbf{(B)}", "**B**", "$B$", "B. 15", "(B) 15", "B: 15", "15", " B ",
                                     # the letter decides, whatever the style of the text after it
                                     "B. \\frac{30}{2}", "B) fifteen", "(B) $15$", "Option B: 15.0"])
def test_choice_letter_dressings_and_exact_text_resolve(content):
    assert resolve_choice(content, OPTIONS) == 1


@pytest.mark.parametrize("content", ["B or C", "B, C", "B and C", "b", "E", "B. 43", "B 12", "Option", "1",
                                     "15 or 12", "The answer is B", "the empty", "", "(B), (C)", "B. or C",
                                     "(B) (C)", "B. 15, C", "B: the empty set"])
def test_choice_answers_naming_no_single_option_are_rejected(content):
    assert resolve_choice(content, OPTIONS) is None


def test_choice_audit_cases():
    assert resolve_choice("B. \\frac{2}{5}", ["1/2", "2/5", "3/5"]) == 1
    assert resolve_choice("C. 7.2\\%", ["5%", "6%", "7.2%"]) == 2
    assert resolve_choice("D. 180 \\text{ Kcal}", ["90 Kcal", "120 Kcal", "150 Kcal", "180 Kcal"]) == 3


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


def test_slow_and_sleeping_programs_are_decided_timeouts():
    t0 = time.monotonic()
    v = code_job("while True:\n    pass\n" + REF, {**ASSERT, "time_limit_s": 1})
    assert (v.reward, v.error) == (0.0, "timeout") and time.monotonic() - t0 < WALL_FACTOR * 1 + EXIT_S
    v = code_job("import time\ntime.sleep(30)\n" + REF, {**ASSERT, "time_limit_s": 0.25})
    assert (v.reward, v.error) == (0.0, "timeout")


@pytest.mark.parametrize("gold", [PYTEST, ASSERT, CHECK])
def test_a_solution_followed_by_an_example_block_is_judged_on_both(gold):
    example = "print(add(2, 3))  # Output: 5"
    v = judge(Job("code", f"```python\n{REF}```\nExample:\n```python\n{example}\n```", gold, "code"))
    assert v.reward == 1.0 and v.extracted == f"{REF}\n\n{example}\n"
    wrong_last = "def add(a, b):\n    return a - b\n"  # a last block that defines the tested name decides alone
    v = judge(Job("code", f"```python\n{REF}```\n```python\n{wrong_last}```", gold, "code"))
    assert v.reward == 0.0 and v.extracted == wrong_last


def test_an_unclosed_last_python_block_is_no_answer():
    v = judge(Job("code", f"```python\n{REF}```\nNow faster:\n```python\ndef add(a, b):\n", CHECK, "code"))
    assert (v.reward, v.complied, v.correct_any) == (0.0, False, False)


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


def test_a_comparison_that_exhausts_its_cpu_budget_is_decided_not_verified():
    """Census answers (math_train/intermediate_algebra/1209) and the form behind the selection census's undecided
    verdicts: power towers that sympy computes in C, where no signal reaches. The budget is CPU time, so the verdict
    does not depend on the machine's load. (Against their numeral references the towers are now rejected unevaluated,
    so an expression reference makes the comparison run.)"""
    pool = Pool(2)
    try:
        pool.run([Job("open", "\\boxed{1}", "1", "box")] * 2)  # start both processes (imports take seconds)
        out = pool.run([Job("open", "So \\boxed{\\frac{2}{2^{2^{2007}} + 1}}", "\\sqrt{2}", "box"),
                        Job("open", "\\boxed{10^{10^{9}}}", "\\sqrt{2}", "box"), Job("open", "\\boxed{1}", "1", "box")])
        for v, extracted in zip(out[:2], ["\\frac{2}{2^{2^{2007}} + 1}", "10^{10^{9}}"]):
            assert v == Verdict(0.0, False, True, "direct", extracted, f"comparison exceeded its CPU budget "
                                f"({COMPARISON_CPU_S} s)") and not is_undecided(v.error)
        assert out[2].reward == 1.0
        assert pool.run([Job("open", "\\boxed{2}", "2", "box")])[0].reward == 1.0  # the replacement works
    finally:
        pool.close()


def test_a_wall_clock_deadline_with_cpu_budget_to_spare_is_undecided():
    pool = Pool(1)
    try:
        pool.run([Job("open", "\\boxed{1}", "1", "box")])
        v = pool.run([Job("open", "\\boxed{10^{10^{10}}}", "\\sqrt{2}", "box")], timeout=2)[0]
        assert is_undecided(v.error) and "starved" in v.error and v.reward == 0.0
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


@pytest.mark.parametrize("gold,reason", [
    ("1, 2", "needs a type"), ("(1, 2)", "needs a type"), ("[0.333, 0.25]", "needs a type"),
    ("\\log 100", "logarithm"), ("2\\log x", "logarithm"),
])
def test_gold_check_refuses_ambiguous_references(gold, reason):
    assert reason in gold_check([("open", gold)])[0]


@pytest.mark.parametrize("gold", ["[0, 1)", "(-\\infty, 2]", "\\{1, 2\\}", "\\log 1", "\\log_2 8", "\\ln 2",
                                  {"answer": "1, 2", "type": "list"}, {"answer": "(1, 2)", "type": "interval"},
                                  {"answer": "[0.333, 0.25]", "type": "tuple"}, {"answer": "7", "type": "number"},
                                  "6:00", "4:40 a.m.", "(1)(2)(4)"])
def test_gold_check_accepts_unambiguous_and_typed_references(gold):
    assert gold_ok("open", gold)


def test_gold_check_refuses_a_repeated_correct_option_but_not_another_repeat():
    assert "ambiguous" in gold_check([("choice", {"options": ["5", "5 ", "6"], "answer": "A"})])[0]
    assert gold_ok("choice", {"options": ["5", "5", "6"], "answer": "C"})


@pytest.mark.parametrize("answer,gold,ok", [
    ("0.25, 0.333", {"answer": "[0.333, 0.25]", "type": "tuple"}, False),
    ("[0.333, 0.25]", {"answer": "[0.333, 0.25]", "type": "tuple"}, True),
    ("(0.333, 0.25)", {"answer": "[0.333, 0.25]", "type": "tuple"}, True),
    ("0.25, 0.333", {"answer": "[0.333, 0.25]", "type": "list"}, True),
    ("\\{0.25, 0.333\\}", {"answer": "0.333, 0.25", "type": "list"}, True),
    ("0.333", {"answer": "[0.333, 0.25]", "type": "list"}, False),
    ("1, 2", {"answer": "(1, 2)", "type": "interval"}, False),
    ("(1, 2)", {"answer": "(1, 2)", "type": "interval"}, True),
    ("[1, 2]", {"answer": "(1, 2)", "type": "interval"}, False),
    ("-1, 2, 5", {"answer": "-1, 2, 5", "type": "tuple"}, True),  # "from least to greatest"
    ("5, 2, -1", {"answer": "-1, 2, 5", "type": "tuple"}, False),
    ("1, 2", {"answer": "1", "type": "number"}, False),
])
def test_typed_references(answer, gold, ok):
    assert open_job(f"\\boxed{{{answer}}}", gold).reward == float(ok)


def test_an_unclosed_last_marker_does_not_fall_back():
    # an unclosed last box: neither the program nor the earlier box answers
    v = open_job("\\boxed{8}\n```python\nprint(7)\n```\nso \\boxed{7", "7", channels=BOTH)
    assert (v.reward, v.complied, v.correct_any) == (0.0, False, False)
    v = open_job("<answer>7</answer> then <answer>8", "7", wrapper="answer_tag")
    assert (v.reward, v.complied) == (0.0, False)
    v = open_job("@@7@@ then @@8", "7", wrapper="at")
    assert (v.reward, v.complied) == (0.0, False)
    v = open_job("```answer\n7\n```\n```answer\n8\n", "7", wrapper="fence")
    assert (v.reward, v.complied) == (0.0, False)
    v = open_job("```python\nprint(7)\n```\n```python\nprint(8)\n", "7", channels=("program",))
    assert (v.reward, v.complied) == (0.0, False)


def test_rules_version_is_exposed():
    assert isinstance(RULES_VERSION, str) and RULES_VERSION
