"""Answer wrappers: how a response marks its final answer, the instruction that asks for it, and its extractor.

`box` is the default for kinds open, choice and verdict, `code` for kind code. The training variants replace `box` in
guardrail prompts; the held-out variants are never used in training and measure compliance with unseen wrappers.

Rules shared by all wrappers:
- A wrapper's form is matched literally (case included), apart from surrounding whitespace.
- Line wrappers look only at the last non-empty line of the response; delimiter and block wrappers take their last
  occurrence anywhere. An unclosed last occurrence (unbalanced braces, an unterminated fence) yields nothing; an
  earlier occurrence is never used instead, because the last one superseded it.
- The content must be non-empty. The content of an answer wrapper (kinds open, choice, verdict) must not contain
  another answer wrapper's markup (a box inside `Final answer:`, tags around a box, ...): such a response follows a
  different format than the one asked for. Programs are exempt: they may print or mention anything.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from typing import Callable

ANSWER_KINDS = frozenset({"open", "choice", "verdict"})
PYTHON_TAGS = frozenset({"python", "py", "python3"})

# A fenced block: an opening line ``` with an optional info string, the body, and a closing line ``` (both may be
# indented). The body ends at the first closing line.
_FENCE = re.compile(r"^[ \t]*```[ \t]*([^\n`]*?)[ \t]*\n(.*?)^[ \t]*```[ \t]*$", re.M | re.S)


@dataclass(frozen=True)
class Found:
    content: str
    end: int  # offset in the response just after the match; the final answer is the match that ends last


def fenced_blocks(text: str) -> list[tuple[str, str, int]]:
    """(lowercased first word of the info string, dedented body, end offset) of each closed fenced block."""
    out = []
    for m in _FENCE.finditer(text):
        tag = m.group(1).split()[0].lower() if m.group(1).split() else ""
        out.append((tag, textwrap.dedent(m.group(2)), m.end()))
    return out


def python_block(text: str, tolerant: bool = False) -> Found | None:
    """The last fenced block tagged python, py or python3 (also an untagged one if tolerant)."""
    blocks = [b for b in fenced_blocks(text) if b[0] in PYTHON_TAGS or (tolerant and b[0] == "")]
    if not blocks or not blocks[-1][1].strip():
        return None
    return Found(blocks[-1][1], blocks[-1][2])


def _last_line(text: str) -> tuple[str, int] | None:
    body = text.rstrip()
    if not body:
        return None
    return body[body.rfind("\n") + 1:].strip(), len(body)


def _line(pattern: str) -> Callable[[str], Found | None]:
    rx = re.compile(pattern)

    def extract(text: str) -> Found | None:
        last = _last_line(text)
        m = rx.fullmatch(last[0]) if last else None
        return Found(m.group(1).strip(), last[1]) if m else None
    return extract


def _last_match(pattern: str, flags: int = 0) -> Callable[[str], Found | None]:
    rx = re.compile(pattern, flags)

    def extract(text: str) -> Found | None:
        matches = list(rx.finditer(text))
        return Found(matches[-1].group(1).strip(), matches[-1].end()) if matches else None
    return extract


_BOX = re.compile(r"\\boxed\s*\{")


def _boxed(text: str) -> Found | None:
    """The last \\boxed{...}, if its braces balance."""
    starts = [m.end() for m in _BOX.finditer(text)]
    if not starts:
        return None
    depth = 1
    for j in range(starts[-1], len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return Found(text[starts[-1]:j].strip(), j + 1)
    return None


def _json_line(text: str) -> Found | None:
    """The last non-empty line is a JSON object with the single key "answer" and a string or number value."""
    last = _last_line(text)
    if not last or not last[0].startswith("{"):
        return None
    try:
        obj = json.loads(last[0], parse_int=str, parse_float=str)  # numbers keep their written form
    except ValueError:
        return None
    if not isinstance(obj, dict) or list(obj) != ["answer"] or not isinstance(obj["answer"], str):
        return None
    return Found(obj["answer"].strip(), last[1])


def _answer_fence(text: str) -> Found | None:
    blocks = [b for b in fenced_blocks(text) if b[0] == "answer"]
    return Found(blocks[-1][1].strip(), blocks[-1][2]) if blocks else None


def _code(text: str) -> Found | None:
    return python_block(text)


@dataclass(frozen=True)
class Wrapper:
    id: str
    kinds: frozenset[str]
    instruction: str  # one sentence asking for this form; it says "your final answer" and leaves what that is to the prompt
    render: Callable[[str], str]  # a compliant response carrying this content (for gold checks and tests)
    extract: Callable[[str], Found | None]  # the raw match; use `find` for the full rule
    markup: re.Pattern | None  # this wrapper's markup, which no other answer wrapper's content may contain

    def find(self, text: str) -> Found | None:
        """The content this wrapper extracts from a response, with the shared rules applied."""
        found = self.extract(text)
        if found is None or not found.content.strip():
            return None
        if self.kinds == ANSWER_KINDS and any(w.markup is not None and w.id != self.id and w.markup.search(
                found.content) for w in WRAPPERS.values() if w.kinds == ANSWER_KINDS):
            return None
        return found


def _w(id, kinds, instruction, render, extract, markup):
    return Wrapper(id, frozenset(kinds), instruction, render, extract, re.compile(markup) if markup else None)


_X = "where X is your final answer"
_REGISTRY = [
    _w("box", ANSWER_KINDS, "Put your final answer within \\boxed{}.", lambda x: f"\\boxed{{{x}}}", _boxed,
       r"\\boxed"),
    _w("code", {"code"}, "Give your complete solution as one fenced Python code block (```python ... ```).",
       lambda x: f"```python\n{x}\n```", _code, None),
    # training variants
    _w("final_line", ANSWER_KINDS, f"End your response with a line of the form `Final answer: X`, {_X}.",
       lambda x: f"Final answer: {x}", _line(r"Final answer:(.*)"), r"Final answer:"),
    _w("json", ANSWER_KINDS, f'End your response with a line containing only the JSON object {{"answer": "X"}}, {_X}.',
       lambda x: json.dumps({"answer": x}), _json_line, r'\{\s*"answer"\s*:'),
    _w("hash", ANSWER_KINDS, f"End your response with a line of the form `#### X`, {_X}.",
       lambda x: f"#### {x}", _line(r"####\s(.*)"), r"####"),
    _w("fence", ANSWER_KINDS, f"End your response with a fenced block tagged answer (a line ```answer, then X, then "
       f"a line ```), {_X}.", lambda x: f"```answer\n{x}\n```", _answer_fence, None),  # ``` is code's markup
    _w("at", ANSWER_KINDS, f"Write your final answer between double at signs, as @@X@@, at the end of your response.",
       lambda x: f"@@{x}@@", _last_match(r"@@([^@\n]+)@@"), r"@@"),
    _w("sentence", ANSWER_KINDS, f"End your response with the sentence `The answer is X.`, {_X}.",
       lambda x: f"The answer is {x}.", _line(r"The answer is (.*?)\.?"), r"The answer is"),
    # held-out variants
    _w("answer_tag", ANSWER_KINDS, "Put your final answer between <answer> and </answer> tags.",
       lambda x: f"<answer>{x}</answer>", _last_match(r"<answer>(.*?)</answer>", re.S), r"</?answer>"),
    _w("upper_line", ANSWER_KINDS, f"End your response with a line of the form `ANSWER: X`, {_X}.",
       lambda x: f"ANSWER: {x}", _line(r"ANSWER:(.*)"), r"ANSWER:"),
    _w("quote_line", ANSWER_KINDS, "End your response with a line that starts with `> ` followed by your final answer.",
       lambda x: f"> {x}", _line(r">\s(.*)"), None),
]
WRAPPERS: dict[str, Wrapper] = {w.id: w for w in _REGISTRY}
DEFAULT = {"open": "box", "choice": "box", "verdict": "box", "code": "code"}
TRAIN_WRAPPERS = ("final_line", "json", "hash", "fence", "at", "sentence")
HELDOUT_WRAPPERS = ("answer_tag", "upper_line", "quote_line")


def for_kind(kind: str) -> list[Wrapper]:
    """All wrappers that apply to a kind, default first (the tie order of tolerant extraction)."""
    return [w for w in WRAPPERS.values() if kind in w.kinds]
