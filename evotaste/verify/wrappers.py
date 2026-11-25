"""Answer wrappers: how a response marks its final answer, the instruction that asks for it, and its extractor.

`box` is the default for kinds open, choice and verdict, `code` for kind code. The training variants replace `box` in
guardrail prompts; the held-out variants are never used in training and measure compliance with unseen wrappers.

Rules shared by all wrappers:
- A wrapper's form is matched literally (case included), apart from surrounding whitespace.
- Line wrappers look only at the last non-empty line of the response; delimiter and block wrappers take their last
  occurrence anywhere. An unclosed last opening marker (unbalanced braces, an unterminated tag or fence) yields
  nothing: an earlier occurrence is never used instead, because the last one superseded it, and the judge does not
  fall back to another wrapper or channel that ends before it (`Unclosed`).
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


@dataclass(frozen=True)
class Unclosed:
    start: int  # offset of the last opening marker, which nothing closes


def fenced_blocks(text: str) -> list[tuple[str, str, int]]:
    """(lowercased first word of the info string, dedented body, end offset) of each closed fenced block."""
    out = []
    for m in _FENCE.finditer(text):
        tag = m.group(1).split()[0].lower() if m.group(1).split() else ""
        out.append((tag, textwrap.dedent(m.group(2)), m.end()))
    return out


def _tagged_opening(tags) -> re.Pattern:
    return re.compile(r"^[ \t]*```[ \t]*(?i:" + "|".join(sorted(tags)) + r")\b[^\n]*$", re.M)


_PYTHON_OPENING = _tagged_opening(PYTHON_TAGS)


def python_blocks(text: str, tolerant: bool = False) -> list[Found] | Unclosed:
    """The closed fenced blocks tagged python, py or python3 (also untagged ones if tolerant), in order; Unclosed if a
    tagged opening line follows the last of them."""
    blocks = [Found(body, end) for tag, body, end in fenced_blocks(text)
              if tag in PYTHON_TAGS or (tolerant and tag == "")]
    after = blocks[-1].end if blocks else 0
    if (m := _PYTHON_OPENING.search(text, after)) is not None:
        return Unclosed(m.start())
    return blocks


def python_block(text: str, tolerant: bool = False) -> Found | Unclosed | None:
    """The last fenced Python block (see python_blocks), Unclosed, or None (also for an empty one)."""
    blocks = python_blocks(text, tolerant)
    if isinstance(blocks, Unclosed):
        return blocks
    return blocks[-1] if blocks and blocks[-1].content.strip() else None


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


def _last_match(pattern: str, opening: str, flags: int = 0) -> Callable[[str], Found | Unclosed | None]:
    rx, opener = re.compile(pattern, flags), re.compile(opening)

    def extract(text: str) -> Found | Unclosed | None:
        matches = list(rx.finditer(text))
        after = matches[-1].end() if matches else 0
        if (m := opener.search(text, after)) is not None:
            return Unclosed(m.start())
        return Found(matches[-1].group(1).strip(), matches[-1].end()) if matches else None
    return extract


_BOX = re.compile(r"\\boxed\s*\{")


def _boxed(text: str) -> Found | Unclosed | None:
    """The last \\boxed{...}; Unclosed if its braces do not balance."""
    matches = list(_BOX.finditer(text))
    if not matches:
        return None
    depth = 1
    for j in range(matches[-1].end(), len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return Found(text[matches[-1].end():j].strip(), j + 1)
    return Unclosed(matches[-1].start())


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


_ANSWER_OPENING = _tagged_opening({"answer"})


def _answer_fence(text: str) -> Found | Unclosed | None:
    blocks = [b for b in fenced_blocks(text) if b[0] == "answer"]
    after = blocks[-1][2] if blocks else 0
    if (m := _ANSWER_OPENING.search(text, after)) is not None:
        return Unclosed(m.start())
    return Found(blocks[-1][1].strip(), blocks[-1][2]) if blocks else None


def _code(text: str) -> Found | Unclosed | None:
    return python_block(text)


@dataclass(frozen=True)
class Wrapper:
    id: str
    kinds: frozenset[str]
    instruction: str  # one sentence asking for this form; "your final answer" is left for the prompt to define
    render: Callable[[str], str]  # a compliant response carrying this content (for gold checks and tests)
    extract: Callable[[str], Found | Unclosed | None]  # the raw match; use `locate` or `find` for the full rule
    markup: re.Pattern | None  # this wrapper's markup, which no other answer wrapper's content may contain

    def locate(self, text: str) -> Found | Unclosed | None:
        """The content this wrapper extracts from a response with the shared rules applied, or Unclosed."""
        found = self.extract(text)
        if isinstance(found, Unclosed):
            return found
        if found is None or not found.content.strip():
            return None
        if self.kinds == ANSWER_KINDS and any(w.markup is not None and w.id != self.id and w.markup.search(
                found.content) for w in WRAPPERS.values() if w.kinds == ANSWER_KINDS):
            return None
        return found

    def find(self, text: str) -> Found | None:
        """The content this wrapper extracts from a response (an unclosed last marker gives none)."""
        found = self.locate(text)
        return found if isinstance(found, Found) else None


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
       lambda x: f"@@{x}@@", _last_match(r"@@([^@\n]+)@@", r"@@"), r"@@"),
    _w("sentence", ANSWER_KINDS, f"End your response with the sentence `The answer is X.`, {_X}.",
       lambda x: f"The answer is {x}.", _line(r"The answer is (.*?)\.?"), r"The answer is"),
    # held-out variants
    _w("answer_tag", ANSWER_KINDS, "Put your final answer between <answer> and </answer> tags.",
       lambda x: f"<answer>{x}</answer>", _last_match(r"<answer>(.*?)</answer>", r"<answer>", re.S),
       r"</?answer>"),
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
