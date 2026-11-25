"""Mathematical equivalence of an open answer with its reference.

A reference may carry a type (TYPES). Typed "tuple" and "list" references are split into elements, whatever their
brackets, and compared in order ("tuple") or by a one-to-one matching ("list"); the other types are compared whole.
An untyped reference that is a bare comma list ("1, 2") or a parenthesised tuple ("(1, 2)") is split at its top-level
commas and compared element by element: two bare lists in any order, otherwise in order, so a bare list answers a
tuple and the other way round; the lengths must agree. (gold_check refuses untyped references whose list, tuple or
interval reading is ambiguous: see `ambiguity`.)

Whole contents are parsed by math-verify's LaTeX parser (as the content of a box: no free-text extraction), with its
own wall-clock alarms off, so that a slow parse or comparison is bounded by the pool's deadline (undecided) and never
silently "not equal". A program's output line that is a Python expression built only from numbers, arithmetic and
sympy's printed names (pi, E, I, oo, sqrt, log, ...) is also read as such, without eval, when its LaTeX reading
does not verify. Numbers, and tuples, sets,
intervals and matrices of numbers, are compared here; anything with variables or relations is left to math-verify's
symbolic comparison. math-verify's numeric comparison is bypassed for numbers because it rounds to six decimal places
in absolute terms (so 1e-7 equals 2e-7) and chops tiny exact values (so 2^-60 equals 2^-61).

Numeric rules (|.| is the complex modulus, g the reference value, a the answer value):
- reference with a decimal literal ("0.25", "3.2 \\times 10^{-19}"): |a - g| < half a unit in the last significant
  digit of the reference's least precise decimal literal, and |a - g| < 2e-2 |g| (so 4/3 answers 1.33);
- exact reference, answer with a decimal: |a - g| <= 1e-6 |g|;
- both exact: exactly equal (rationals compared as such; otherwise a difference that sympy shows to be zero, by
  evaluation, simplification or its minimal polynomial; anything undecided is not equal);
- a zero reference against a decimal: |a| <= 1e-12.
A percentage X% has the value X/100 and also matches X (math-verify's convention).
"""

from __future__ import annotations

import ast
import functools
import logging
import re
import warnings
from dataclasses import dataclass

import sympy
from math_verify import LatexExtractionConfig, parse, verify
from sympy.core.relational import Relational

TYPES = ("number", "expression", "tuple", "list", "interval")
MAX_CHARS = 500  # longer contents are not taken for one answer
MAX_EXPONENT = 10_000  # largest integer exponent in the Python reading of an output line
DECIMAL_REL = sympy.Rational(2, 100)  # exact thresholds: a boundary case must not depend on binary rounding
EXACT_DECIMAL_REL = sympy.Rational(1, 10 ** 6)
ZERO_ABS = sympy.Rational(1, 10 ** 12)
DIGITS = 60

warnings.filterwarnings("ignore", module="math_verify")
logging.getLogger("math_verify").setLevel(logging.ERROR)  # it warns once that its timeouts are off
_CONFIG = [LatexExtractionConfig(boxed_match_priority=0)]
_DELIMITERS = (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]"), ("**", "**"), ("`", "`"))
_DECIMAL = re.compile(r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)?\.\d+(?![\d.])")
_SCIENTIFIC = re.compile(r"(?<![\w.])(\d+\.\d*|\.\d+)[eE]([+-]?\d+)(?![\w.])")
_LONE_SCIENTIFIC = re.compile(r"([+-]?\d+)[eE]([+-]?\d+)")
_COMMAND = re.compile(r"\\(?:text|textbf|textrm|mathrm|mathbf)\s*\{(.*)\}", re.S)
_IMAGINARY = re.compile(r"(?<=\d)i\b")
_LONE_I = re.compile(r"(?<![A-Za-z\\])i(?![A-Za-z])")  # i as a token of its own, not inside a word such as sin
_THOUSANDS = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_LOG = re.compile(r"\\log(?![_A-Za-z])")
_OPEN, _CLOSE = "([{", ")]}"
_X = sympy.Symbol("x")


class GoldError(ValueError):
    """The reference does not parse: the item cannot be verified."""


@dataclass(frozen=True)
class Parsed:
    expr: object  # sympy expression, set, interval, relation or matrix; decimals as exact rationals
    decimal: bool  # written with a decimal literal (a point, or e-notation)
    sig: int | None  # significant digits of the least precise decimal literal


def reference(gold) -> tuple[str, str | None]:
    """(text, type) of an open gold: a string, or {"answer": text, "type": one of TYPES}."""
    if isinstance(gold, str):
        return gold, None
    if not isinstance(gold, dict) or not isinstance(gold.get("answer"), str) or gold.get("type") not in TYPES:
        raise ValueError(f"an open gold is a string or {{'answer': str, 'type': one of {TYPES}}}: {gold!r}")
    return gold["answer"], gold["type"]


# -- reading a content ----------------------------------------------------------------------------------------------

def _strip(s: str) -> str:
    """Surrounding math delimiters, markdown emphasis and backticks, a \\text-like command around the whole answer,
    and a trailing period removed."""
    s = s.strip().replace("−", "-")
    prev = None
    while s != prev:
        prev = s
        for a, b in _DELIMITERS:
            if len(s) > len(a) + len(b) and s.startswith(a) and s.endswith(b):
                s = s[len(a):-len(b)].strip()
        if (m := _COMMAND.fullmatch(s)) and balanced(m.group(1)):
            s = m.group(1).strip()
        if s.endswith("."):
            s = s[:-1].rstrip()
    return s


def balanced(s: str) -> bool:
    """Braces balance and never close more than they opened."""
    depth = 0
    for ch in s:
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0


def _latex(s: str) -> str:
    """Python's ** and e-notation (with a point in the mantissa, or as the whole answer) as LaTeX powers; a
    coefficient before i spaced, so that math-verify reads 3i, 3 i and i alike as the imaginary unit."""
    s = _IMAGINARY.sub(" i", s.replace("**", "^"))
    if m := _LONE_SCIENTIFIC.fullmatch(s):
        return f"{m.group(1)}\\times 10^{{{m.group(2)}}}"
    return _SCIENTIFIC.sub(lambda m: f"{m.group(1)}\\times 10^{{{m.group(2)}}}", s)


def _significant(mantissa: str) -> int:
    return len(mantissa.replace(",", "").replace(".", "").lstrip("+-").lstrip("0"))


_NAMES = {"pi": sympy.pi, "E": sympy.E, "I": sympy.I, "oo": sympy.oo}
_FUNCTIONS = {"sqrt": sympy.sqrt, "cbrt": sympy.cbrt, "root": sympy.root, "exp": sympy.exp, "log": sympy.log,
              "ln": sympy.log, "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan, "asin": sympy.asin,
              "acos": sympy.acos, "atan": sympy.atan, "Abs": sympy.Abs, "abs": sympy.Abs,
              "factorial": sympy.factorial, "binomial": sympy.binomial, "Rational": sympy.Rational,
              "Integer": sympy.Integer}
_OPERATORS = {ast.Add: lambda x, y: x + y, ast.Sub: lambda x, y: x - y, ast.Mult: lambda x, y: x * y,
              ast.Div: lambda x, y: x / y, ast.Pow: lambda x, y: x ** y}


class _NotPython(Exception):
    pass


def _python(node, src: str):
    """A sympy value for a whitelisted Python expression tree, built without eval."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return sympy.Integer(node.value) if type(node.value) is int else \
            sympy.Rational(ast.get_source_segment(src, node) or repr(node.value))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _python(node.operand, src)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _python(node.left, src), _python(node.right, src)
        if isinstance(node.op, ast.Pow) and not (right.is_Number and abs(right) <= MAX_EXPONENT):
            raise _NotPython
        return _OPERATORS[type(node.op)](left, right)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS
            and not node.keywords):
        return _FUNCTIONS[node.func.id](*(_python(a, src) for a in node.args))
    raise _NotPython


def _python_reading(s: str):
    try:
        return _python(ast.parse(s, mode="eval").body, s)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, _NotPython):
        return None


def parse_answer(s: str, python: bool = False) -> Parsed | None:
    """A content as one expression, read as LaTeX (box content) or, with `python`, as a Python/sympy expression;
    None if it does not parse that way."""
    s = _strip(s)
    if not s or len(s) > MAX_CHARS:
        return None
    if python:
        if (expr := _python_reading(s)) is None:
            return None
    else:
        found = parse(f"\\boxed{{{_latex(s)}}}", extraction_config=_CONFIG, fallback_mode="no_fallback",
                      extraction_mode="first_match", parsing_timeout=None)
        exprs = [e for e in found if isinstance(e, (sympy.Basic, sympy.MatrixBase))]
        if not exprs:
            return None
        expr = exprs[0]
        imaginary = {x: sympy.I for x in expr.free_symbols if isinstance(x, sympy.Symbol) and x.name == "i"}
        if imaginary and _LONE_I.search(s):  # math-verify leaves i a symbol, so that (1+i)^2 would not equal 2i
            replaced = expr.xreplace(imaginary)
            if not isinstance(replaced, sympy.logic.boolalg.BooleanAtom):
                expr = replaced
    floats = expr.atoms(sympy.Float)
    if floats:  # the parser's binary floats back to the decimals they were read from (15 significant digits)
        expr = expr.xreplace({f: sympy.Rational(str(f)) for f in floats})
    mantissas = _DECIMAL.findall(s) + [m.group(1) for m in [_LONE_SCIENTIFIC.fullmatch(s)] if m]
    sig = min(map(_significant, mantissas)) if mantissas else None
    return Parsed(expr, bool(mantissas or floats), sig)


@functools.lru_cache(maxsize=4096)
def parse_gold(gold: str) -> Parsed:
    parsed = parse_answer(gold)
    if parsed is None:
        raise GoldError(f"reference does not parse: {gold!r}")
    return parsed


# -- comparing values -----------------------------------------------------------------------------------------------

def _is_number(x) -> bool:
    return isinstance(x, sympy.Expr) and not x.free_symbols and bool(x.is_number)


def _values(x) -> list:
    """The value of a number, and for a percentage X% also X."""
    pct = sympy.UnevaluatedExpr(sympy.Rational(1, 100))
    if isinstance(x, sympy.Mul) and len(x.args) == 2 and x.args[1] == pct:
        return [x.args[0] / 100, x.args[0]]
    return [x.doit()]


def _exponent(x) -> int:
    """floor(log10(x)) for a positive number, exact at powers of ten."""
    e = int(sympy.floor(sympy.log(x, 10)))
    while sympy.Integer(10) ** (e + 1) <= x:
        e += 1
    while sympy.Integer(10) ** e > x:
        e -= 1
    return e


def exact_equal(g, a) -> bool:
    """Two exact numbers are equal: as rationals, or by a difference that sympy shows to be zero (its own
    evaluation, simplification, or for algebraic numbers the minimal polynomial). Undecided means not equal."""
    if g.is_Rational and a.is_Rational:
        return bool(g == a)
    d = g - a
    if d == 0:
        return True
    if d.is_zero is not None:
        return bool(d.is_zero)
    if sympy.simplify(d) == 0:
        return True
    try:
        return sympy.minimal_polynomial(d, _X) == _X
    except Exception:  # noqa: BLE001 - not algebraic, or not decidable by sympy
        return False


def _close(g, a, gold: Parsed, gold_decimal: bool, answer_decimal: bool) -> bool:
    if g == a:
        return True
    if not (gold_decimal or answer_decimal):
        return exact_equal(g, a)
    if g.is_Rational and a.is_Rational:  # decimals are rationals here: compare exactly
        gv, av = g, a
    else:
        gv, av = sympy.N(g, DIGITS), sympy.N(a, DIGITS)
    if not (gv.is_finite and av.is_finite):
        return False
    diff, scale = abs(gv - av), abs(gv)
    if scale == 0:
        return bool(abs(av) <= ZERO_ABS)
    if gold_decimal:
        tol = DECIMAL_REL * scale
        if gold.sig:
            tol = min(tol, sympy.Rational(1, 2) * sympy.Integer(10) ** (_exponent(scale) - gold.sig + 1))
        return bool(diff < tol)
    return bool(diff <= EXACT_DECIMAL_REL * scale)


def _numbers(x) -> bool:
    """A number, or a tuple, set, interval or matrix whose entries are all numbers."""
    if isinstance(x, sympy.MatrixBase):
        return all(_is_number(e) for e in x)
    if isinstance(x, (sympy.Tuple, sympy.FiniteSet)):
        return all(_numbers(e) for e in x.args)
    if isinstance(x, sympy.Interval):
        return _is_number(x.start) and _is_number(x.end)
    return _is_number(x)


def _same(g, a, gold: Parsed, gd: bool, ad: bool) -> bool:
    """g and a satisfy _numbers: equal type and shape, and close entries (sets: a one-to-one matching)."""
    if isinstance(g, sympy.MatrixBase) or isinstance(a, sympy.MatrixBase):
        return (isinstance(g, sympy.MatrixBase) and isinstance(a, sympy.MatrixBase) and g.shape == a.shape
                and all(_same(x, y, gold, gd, ad) for x, y in zip(g, a)))
    if type(g) is not type(a) and not (_is_number(g) and _is_number(a)):
        return False
    if isinstance(g, sympy.Tuple):
        return len(g) == len(a) and all(_same(x, y, gold, gd, ad) for x, y in zip(g.args, a.args))
    if isinstance(g, sympy.Interval):
        return ((g.left_open, g.right_open) == (a.left_open, a.right_open)
                and _same(g.start, a.start, gold, gd, ad) and _same(g.end, a.end, gold, gd, ad))
    if isinstance(g, sympy.FiniteSet):
        rest = list(a.args)
        if len(rest) != len(g.args):
            return False
        for x in g.args:
            hit = next((i for i, y in enumerate(rest) if _same(x, y, gold, gd, ad)), None)
            if hit is None:
                return False
            rest.pop(hit)
        return True
    if g.is_infinite or a.is_infinite:
        return g == a
    return any(_close(x, y, gold, gd, ad) for x in _values(g) for y in _values(a))


def _unwrap(g, a):
    """Follow the reference's form for equations, as math-verify does: an answer `x = 5` or `2 + 3 = 5` stands for
    its last right-hand side when the reference is no relation, and a reference `x = 5` is answered by `5`.
    Inequalities are never unwrapped."""
    if isinstance(a, sympy.Eq) and not isinstance(g, Relational):
        while isinstance(a, sympy.Eq):
            a = a.rhs
    elif isinstance(g, sympy.Eq) and isinstance(g.lhs, sympy.Symbol) and not isinstance(a, Relational):
        g = g.rhs
    return g, a


def _whole(answer: str, gold: str, program: bool) -> bool:
    """A program's output line counts in its LaTeX reading or, failing that, its Python reading (E is a label in
    the first and Euler's number in the second; pi/4 is only meaningful in the second)."""
    ref = parse_gold(gold)
    return any(got is not None and _compare(ref, got)
               for got in (parse_answer(answer, python) for python in ((False, True) if program else (False,))))


def _compare(ref: Parsed, got: Parsed) -> bool:
    g, a = _unwrap(ref.expr, got.expr)
    if _numbers(g) and _numbers(a):
        return _same(g, a, ref, ref.decimal, got.decimal)
    if _numbers(g) or _numbers(a):  # a number against something with variables
        return False
    return bool(verify(g, a, timeout_seconds=None))


# -- lists and tuples -----------------------------------------------------------------------------------------------

def _top_level(s: str) -> list[str]:
    """s split at the commas outside all brackets and braces."""
    items, depth, start = [], 0, 0
    for i, ch in enumerate(s):
        depth += (ch in _OPEN) - (ch in _CLOSE)
        if ch == "," and depth == 0:
            items.append(s[start:i].strip())
            start = i + 1
    return items + [s[start:].strip()]


def _closes_at_end(s: str) -> bool:
    """The bracket opening s is closed by its last character."""
    depth = 0
    for i, ch in enumerate(s):
        depth += (ch in _OPEN) - (ch in _CLOSE)
        if depth == 0:
            return i == len(s) - 1
    return False


def _plain(s: str) -> str:
    s = re.sub(r"\\(?:left|right)(?![A-Za-z])", "", _strip(s))
    return re.sub(r"(?<=\d),\\!(?=\d{3})", "", s)


def as_list(s: str, program: bool = False) -> tuple[str, list[str]] | None:
    """("bare", items) for a comma list of two or more items, ("paren", items) for one in parentheses, else None. A
    number with thousands separators (1,000 or 1,\\!000) is no list, and neither is an interval with a bracket; a
    program's printed Python list ([1, 2]) is a bare list."""
    s = _plain(s)
    if _THOUSANDS.fullmatch(s):
        return None
    form = "bare"
    if s.startswith("(") and s.endswith(")") and _closes_at_end(s):
        form, s = "paren", s[1:-1]
    elif program and s.startswith("[") and s.endswith("]") and _closes_at_end(s):
        s = s[1:-1]
    items = _top_level(s)
    return (form, items) if len(items) >= 2 and all(items) else None


def _elements(s: str, sets: bool) -> list[str] | None:
    """The elements of a typed tuple or list, written bare or in (), [] (or, for a list, \\{ \\})."""
    s = _plain(s)
    for a, b in (("(", ")"), ("[", "]")) + ((("\\{", "\\}"),) if sets else ()):
        if s.startswith(a) and s.endswith(b) and _closes_at_end(s[len(a) - 1:]):
            s = s[len(a):-len(b)]
            break
    items = _top_level(s)
    return items if all(items) else None


def _matched(answers: list[str], golds: list[str], program: bool) -> bool:
    """A one-to-one matching of equivalent elements (greedy)."""
    rest = list(answers)
    for g in golds:
        hit = next((i for i, a in enumerate(rest) if equivalent(a, g, program=program)), None)
        if hit is None:
            return False
        rest.pop(hit)
    return not rest


def equivalent(answer: str, gold: str, kind: str | None = None, program: bool = False) -> bool:
    """Whether the answer content equals the reference text, of type `kind` (None: untyped). `program`: the answer
    is a program's output line. Raises GoldError if the reference does not parse."""
    if kind in ("tuple", "list"):
        refs, gots = _elements(gold, kind == "list"), _elements(answer, kind == "list")
        if refs is None or gots is None or len(refs) != len(gots):
            return False
        if kind == "list":
            return _matched(gots, refs, program)
        return all(equivalent(a, g, program=program) for a, g in zip(gots, refs))
    if kind is None:
        ref_list, got_list = as_list(gold), as_list(answer, program)
        if ref_list and got_list:
            (ref_form, refs), (got_form, gots) = ref_list, got_list
            if len(refs) != len(gots):
                return False
            if ref_form == got_form == "bare":
                return _matched(gots, refs, program)
            return all(equivalent(a, g, program=program) for a, g in zip(gots, refs))
    return _whole(answer, gold, program)


def ambiguity(gold: str, kind: str | None) -> str | None:
    """Why the reference cannot be verified as intended, if detectably so: an untyped list, tuple or closed-interval
    reading, or a logarithm without a base whose two readings (natural, decimal) differ."""
    if kind is None:
        s = _plain(gold)
        bracketed = s.startswith("[") and s.endswith("]") and _closes_at_end(s) and len(_top_level(s[1:-1])) > 1
        if as_list(gold) or bracketed:
            return "ambiguous: a list, tuple or interval reference needs a type (tuple, list or interval)"
    if _LOG.search(gold):
        natural, decimal = _LOG.sub(r"\\ln", gold), _LOG.sub(r"\\log_{10}", gold)
        try:
            if not equivalent(natural, decimal, kind) or not equivalent(decimal, natural, kind):
                return "ambiguous: a logarithm without a base reads differently as natural and decimal"
        except GoldError:
            return "ambiguous: a logarithm without a base does not parse in both readings"
    return None
