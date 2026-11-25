"""Traits of a single response, computed from its record alone: counted traits (how a member works) and layout traits
(how it writes), kept apart; and the three views of a response the authorship classifier reads.

Counted traits: `checks`, lines of explicit checking; for kind code, counts over the syntax tree of the program the
verifier ran (`extracted`): `code_lines`, `loops`, `comprehensions`, `recursion`, `imports`, `helpers` (NaN when there
is no parsable program). Response length (`tokens`) and the answer channel (`program`) are record fields
(evotaste.analysis.data.FIELDS). Layout traits: `bold`, `headers`, `lists`, `line_words`, `latex`, `rules`.
"""

from __future__ import annotations

import ast
import re

import numpy as np

CHECK = re.compile(r"\b(?:check(?:s|ed|ing)?|double-check\w*|verif\w+|confirm\w*|plug\w*|substitut\w* back|validat\w+"
                   r"|satisf\w+|sanity)\b", re.I)
WORD = re.compile(r"[^\W\d_]+")
BOLD = re.compile(r"\*\*[^*\n]+\*\*")
HEADER = re.compile(r"^(?:#{1,6}\s|\*\*[^*]+\*\*:?$)")
LIST = re.compile(r"^(?:[-*•+]\s+\S|\(?\d+[.)]\s|step\s*\d+\b)", re.I)
RULE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,}|={3,})$")
LATEX = re.compile(r"\\[A-Za-z]+|\\[()\[\]]|\$+")

CODE_TRAITS = ("code_lines", "loops", "comprehensions", "recursion", "imports", "helpers")
COUNTED = ("tokens", "program", "checks") + CODE_TRAITS
LAYOUT = ("bold", "headers", "lists", "line_words", "latex", "rules")


def _per(n: float, d: float, k: float = 100.0) -> float:
    return k * n / d if d else 0.0


def code_traits(program: str | None) -> dict[str, float]:
    try:
        tree = ast.parse(program) if program else None
    except (SyntaxError, ValueError):
        tree = None
    if tree is None:
        return dict.fromkeys(CODE_TRAITS, np.nan)
    nodes = list(ast.walk(tree))
    defs = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    def calls_itself(f) -> bool:
        return any(isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id == f.name)
                                                or (isinstance(n.func, ast.Attribute) and n.func.attr == f.name))
                   for n in ast.walk(f))

    lines = [s for line in program.splitlines() if (s := line.strip()) and not s.startswith("#")]
    return {"code_lines": float(len(lines)),
            "loops": float(sum(isinstance(n, (ast.For, ast.AsyncFor, ast.While)) for n in nodes)),
            "comprehensions": float(sum(isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
                                        for n in nodes)),
            "recursion": float(any(calls_itself(f) for f in defs)),
            "imports": float(sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes)),
            "helpers": float(max(len(defs) - 1, 0))}


def layout_traits(text: str) -> dict[str, float]:
    lines = [s for line in text.splitlines() if (s := line.strip())]
    chars = sum(not c.isspace() for c in text)
    return {"bold": _per(len(BOLD.findall(text)), chars), "headers": _per(sum(bool(HEADER.match(s)) for s in lines),
                                                                          len(lines), 1.0),
            "lists": _per(sum(bool(LIST.match(s)) for s in lines), len(lines), 1.0),
            "line_words": _per(len(WORD.findall(text)), len(lines), 1.0),
            "latex": _per(len(LATEX.findall(text)), chars), "rules": float(sum(bool(RULE.match(s)) for s in lines))}


def response_traits(r: dict) -> dict[str, float]:
    """Counted and layout traits of one response record (tokens and program are record fields)."""
    text = r["text"]
    out = {"checks": float(sum(bool(CHECK.search(line)) for line in text.splitlines()))}
    out |= code_traits(r["extracted"]) if r["kind"] == "code" else dict.fromkeys(CODE_TRAITS, np.nan)
    return out | layout_traits(text)


# -- classifier views ---------------------------------------------------------------------------------------------

def raw_view(text: str) -> str:
    """The text with every run of digits replaced by 0 (the same items' numbers would otherwise be shared)."""
    return re.sub(r"\d+", "0", text)


_MARKUP = [re.compile(p, re.M) for p in (r"\\[A-Za-z]+", r"\\[()\[\]{}]", r"[$`|{}]", r"\*\*|__", r"^\s*#{1,6}\s*",
                                         r"^\s*(?:[-*•+]|\(?\d+[.)])\s+", r"^\s*(?:-{3,}|\*{3,}|_{3,}|={3,})\s*$")]


def stripped_view(text: str) -> str:
    """The text without layout and notation: LaTeX commands, math delimiters, braces, backticks and table bars,
    markdown emphasis, headers, list markers and rules removed; line structure, case and digits flattened."""
    for rx in _MARKUP:
        text = rx.sub(" ", text)
    return re.sub(r"\d+", "0", " ".join(text.split()).lower())
