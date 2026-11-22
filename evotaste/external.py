"""External benchmarks behind the task interface: GSM8K, MATH-500 and MMLU-Pro.

Each item belongs to exactly one part (foundation, blind1, blind2), decided by a seeded hash of its normalised question
text, so identical questions under different ids (MMLU-Pro has 357 such texts) share a part, in every source. Which
parts exist per source is fixed here, which suite uses which source is fixed in configs/suites.yaml. Blind parts are
returned only with a receipt from the blind guard, and only the requested part is ever turned into tasks.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re

from evotaste.tasks import BlindReceipt, Task, check_receipt

SPLIT_SEED = 20261122
PARTS = {
    "gsm8k_train": ("foundation",),
    "gsm8k_test": ("blind1", "blind2"),
    "math500": ("blind1", "blind2"),
    "mmlu_pro": ("foundation", "blind1", "blind2"),
}
_HF = {  # source -> (dataset id, config, split)
    "gsm8k_train": ("openai/gsm8k", "main", "train"),
    "gsm8k_test": ("openai/gsm8k", "main", "test"),
    "math500": ("HuggingFaceH4/MATH-500", None, "test"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, "test"),
}
LETTERS = "ABCDEFGHIJ"


# MMLU-Pro questions that repeat a MATH-500 problem (normalised text); they stay in MATH-500 only, because MATH-500
# has no foundation part to share. GSM8K train, GSM8K test and MATH-500 share no question text with each other.
MMLU_PRO_DROPPED = {"7744", "8289", "8591", "8823", "8830"}


def _hash(key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SPLIT_SEED}/{key}".encode()).digest()[:8], "big")


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def part_of(source: str, question: str) -> str:
    """Part of an item, a function of its question text only: the same text gets the same vault in every source,
    and in MMLU-Pro a third of the texts go to the foundation part instead."""
    h = _hash(normalized(question))
    vault = ("blind1", "blind2")[h % 2]
    if PARTS[source] == ("foundation",):
        return "foundation"
    if "foundation" in PARTS[source] and (h // 2) % 3 == 0:
        return "foundation"
    return vault


def _rows(source: str):
    """(item id, row) pairs of a source, read from the local Hugging Face cache."""
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    name, config, split = _HF[source]
    ds = load_dataset(name, config, split=split)
    if source.startswith("gsm8k"):
        return [(str(i), r) for i, r in enumerate(ds)]
    if source == "math500":
        return [(r["unique_id"], r) for r in ds]
    return [(str(r["question_id"]), r) for r in ds if str(r["question_id"]) not in MMLU_PRO_DROPPED]


def question_text(r: dict) -> str:
    return r["problem"] if "problem" in r else r["question"]


def _task(source: str, item_id: str, r: dict) -> Task:
    uid = f"{source}/{item_id}"
    if source.startswith("gsm8k"):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        return Task(uid, source, r["question"], gold)
    if source == "math500":
        gold = r["answer"]
        return Task(uid, source, r["problem"], gold)
    options = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(r["options"]))
    question = f"{r['question']}\n\nOptions:\n{options}\n\nAnswer with the letter of the correct option."
    gold = r["answer"]
    return Task(uid, source, question, gold)


@functools.lru_cache(maxsize=None)
def _part_tasks(source: str, part: str) -> tuple[Task, ...]:
    rows = [(i, r) for i, r in _rows(source) if part_of(source, question_text(r)) == part]
    rows.sort(key=lambda x: _hash(f"{source}/{x[0]}"))
    return tuple(_task(source, i, r) for i, r in rows)


def items(source: str, part: str, receipt: BlindReceipt | None = None) -> list[Task]:
    """Tasks of one part of a source, in a fixed seeded order. Vault parts need the blind guard's receipt."""
    if part not in PARTS[source]:
        raise ValueError(f"{source} has no part {part!r} (parts: {PARTS[source]})")
    if part != "foundation":
        check_receipt(receipt, part)
    return list(_part_tasks(source, part))


# ---------------------------------------------------------------------------------------------------------------
# Scorers (binary). They receive the extracted <answer> content.

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def score_number(answer: str, gold: str) -> float:
    """GSM8K: the answer must contain exactly one number (commas and $ ignored) equal to the gold number."""
    numbers = _NUMBER.findall(answer.replace(",", "").replace("$", ""))
    return float(len(numbers) == 1 and abs(float(numbers[0]) - float(gold)) < 1e-6)


def _math(s: str):
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse

    s = s.replace("\\$", "").strip()  # MATH answers escape currency signs
    if "$" not in s and "\\boxed" not in s:
        s = f"${s}$"
    return parse(s, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])


def score_math(answer: str, gold: str) -> float:
    """MATH-500: mathematical equivalence of the parsed expressions (math-verify)."""
    from math_verify import verify

    target = _math(gold)
    if not target:
        raise ValueError(f"gold answer does not parse: {gold!r}")
    return float(verify(target, _math(answer)))


_LETTER = re.compile(r"^\(?([A-J])\)?(?:[\s.:)]|$)")


def score_letter(answer: str, gold: str) -> float:
    """MMLU-Pro: the answer starts with the gold option letter (optionally in parentheses)."""
    m = _LETTER.match(answer.strip())
    return float(m is not None and m.group(1) == gold)


SCORERS = {"gsm8k_train": score_number, "gsm8k_test": score_number, "math500": score_math, "mmlu_pro": score_letter}
