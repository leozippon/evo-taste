"""Mathematical equivalence of an open answer with its reference.

Both sides are parsed by math-verify's LaTeX parser (as the content of a box: no free-text extraction). Numbers, and
tuples, sets, intervals and matrices of numbers, are compared here; anything with variables or relations is left to
math-verify's symbolic comparison. math-verify's own numeric comparison is bypassed for numbers because it rounds to
six decimal places in absolute terms (so 1e-7 equals 2e-7) and chops tiny exact values (so 2^-60 equals 2^-61).

Numeric rules (|.| is the complex modulus, g the reference value, a the answer value):
- reference with a decimal literal ("0.25", "3.2 \\times 10^{-19}"): |a - g| < min(half a unit in the last
  significant digit of the reference's least precise decimal literal, 1e-3 |g|);
- exact reference, answer with a decimal: |a - g| <= 1e-6 |g|;
- both exact: equal to 40 significant digits (60-digit evaluation);
- a zero reference against a decimal: |a| <= 1e-12.
A percentage X% has the value X/100 and also matches X (math-verify's convention).
"""

from __future__ import annotations

import functools
import re
import warnings
from dataclasses import dataclass

import sympy
from math_verify import LatexExtractionConfig, parse, verify
from sympy.core.relational import Relational

MAX_CHARS = 500  # longer contents are not taken for one answer
DECIMAL_REL = 1e-3
EXACT_DECIMAL_REL = 1e-6
EXACT_REL = 1e-40
ZERO_ABS = 1e-12
DIGITS = 60

warnings.filterwarnings("ignore", module="math_verify")
_CONFIG = [LatexExtractionConfig(boxed_match_priority=0)]
_DELIMITERS = (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]"), ("**", "**"), ("`", "`"))
_DECIMAL = re.compile(r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)?\.\d+(?![\d.])")
_SCIENTIFIC = re.compile(r"(?<![\w.])(\d+\.\d*|\.\d+)[eE]([+-]?\d+)(?![\w.])")
_LONE_SCIENTIFIC = re.compile(r"([+-]?\d+)[eE]([+-]?\d+)")


class GoldError(ValueError):
    """The reference does not parse: the item cannot be verified."""


@dataclass(frozen=True)
class Parsed:
    expr: object  # sympy expression, set, interval, relation or matrix; decimals as exact rationals
    decimal: bool  # written with a decimal literal (a point, or e-notation)
    sig: int | None  # significant digits of the least precise decimal literal


def _strip(s: str) -> str:
    """Surrounding math delimiters, markdown emphasis and backticks, and a trailing period removed."""
    s = s.strip().replace("−", "-")
    prev = None
    while s != prev:
        prev = s
        for a, b in _DELIMITERS:
            if len(s) > len(a) + len(b) and s.startswith(a) and s.endswith(b):
                s = s[len(a):-len(b)].strip()
        if s.endswith("."):
            s = s[:-1].rstrip()
    return s


def _latex(s: str) -> str:
    """Python's ** and e-notation (with a point in the mantissa, or as the whole answer) as LaTeX powers."""
    s = s.replace("**", "^")
    if m := _LONE_SCIENTIFIC.fullmatch(s):
        return f"{m.group(1)}\\times 10^{{{m.group(2)}}}"
    return _SCIENTIFIC.sub(lambda m: f"{m.group(1)}\\times 10^{{{m.group(2)}}}", s)


def _significant(mantissa: str) -> int:
    return len(mantissa.replace(",", "").replace(".", "").lstrip("+-").lstrip("0"))


def parse_answer(s: str) -> Parsed | None:
    """The content of a wrapper (or a program's output line) as one expression, or None if it does not parse."""
    s = _strip(s)
    if not s or len(s) > MAX_CHARS:
        return None
    found = parse(f"\\boxed{{{_latex(s)}}}", extraction_config=_CONFIG, fallback_mode="no_fallback",
                  extraction_mode="first_match", parsing_timeout=5)
    exprs = [e for e in found if isinstance(e, (sympy.Basic, sympy.MatrixBase))]
    if not exprs:
        return None
    expr = exprs[0]
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


def _close(g, a, gold: Parsed, gold_decimal: bool, answer_decimal: bool) -> bool:
    if g == a:
        return True
    gv, av = sympy.N(g, DIGITS), sympy.N(a, DIGITS)
    if not (gv.is_finite and av.is_finite):
        return False
    diff, scale = abs(gv - av), abs(gv)
    if gold_decimal or answer_decimal:
        if scale == 0:
            return bool(abs(av) <= ZERO_ABS)
        if gold_decimal:
            tol = DECIMAL_REL * scale
            if gold.sig:
                tol = min(tol, sympy.Rational(1, 2) * sympy.Integer(10) ** (_exponent(scale) - gold.sig + 1))
            return bool(diff < tol)
        return bool(diff <= EXACT_DECIMAL_REL * scale)
    return bool(diff <= EXACT_REL * max(scale, abs(av)))


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


def equivalent(answer: str, gold: str) -> bool:
    """Whether the answer content equals the reference. Raises GoldError if the reference does not parse."""
    ref = parse_gold(gold)
    got = parse_answer(answer)
    if got is None:
        return False
    g, a = _unwrap(ref.expr, got.expr)
    if _numbers(g) and _numbers(a):
        return _same(g, a, ref, ref.decimal, got.decimal)
    if _numbers(g) or _numbers(a):  # a number against something with variables
        return False
    return bool(verify(g, a, timeout_seconds=5))
