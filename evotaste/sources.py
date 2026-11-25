"""Round 2 dataset sources: loading from the local Hugging Face cache and normalisation to items.

Every loader returns the candidate items of one source in a fixed order, normalised to a common form: the problem as
the model sees it (options included, answer instruction not), a kind-specific gold reference, and the text that decides
the item's part. Sampled sources draw their sample with a fixed seed. Nothing here assigns parts or filters items; that
is `external.py`. Datasets are read offline from the cache; `python scripts/make_suites_r2.py --download` fills it once
through the configured mirror.
"""

from __future__ import annotations

import ast
import functools
import os
import random
import re
from dataclasses import dataclass

from evotaste.tasks import stable_seed

SAMPLE_SEED = 20261125
LETTERS = "ABCDEFGHIJ"
VERDICT_PROOFWRITER = ("True", "False", "Unknown")
VERDICT_FOLIO = ("True", "False", "Uncertain")
VERDICT_FLD = ("PROVED", "DISPROVED", "UNKNOWN")


@dataclass(frozen=True)
class Item:
    source: str  # source id (Task.family)
    id: str  # item id within the source
    domain: str  # "math" | "code" | "logic"
    kind: str  # "open" | "choice" | "verdict" | "code" | "procedural"
    problem: str  # what the model is asked, options included, without the answer instruction
    gold: object  # kind-specific reference (see tasks.Task)
    stem: str  # text compared for de-duplication: the problem without its options or label list
    group: str  # text whose hash decides the part (the question; the theory for ProofWriter)
    held_out: bool = False  # from the official test or validation split of a training source

    @property
    def uid(self) -> str:
        return f"{self.source}/{self.id}"


@dataclass(frozen=True)
class Source:
    id: str
    domain: str
    role: str  # "train" (training source), "held_out" (all items held out from a training source), "unseen"
    unit: str  # assignment unit; unseen sources of one unit go to the same part
    test_split: bool  # training source with an official held-out split (else 5% / 5% by text hash)
    load: object  # () -> list[Item]


def _load(name: str, config: str | None = None, split: str | None = None, revision: str | None = None):
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    from datasets.utils.logging import set_verbosity_error

    set_verbosity_error()  # offline, every load warns that the hub is unreachable
    return load_dataset(name, config, split=split, revision=revision)


def _rng(*key) -> random.Random:
    return random.Random(stable_seed(SAMPLE_SEED, *key))


def _sample(n_rows: int, k: int, *key) -> list[int]:
    """k row indices out of n_rows, in increasing order (all rows if k >= n_rows)."""
    return sorted(_rng(*key).sample(range(n_rows), k)) if k < n_rows else list(range(n_rows))


def options_block(options: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))


def _choice(source, item_id, domain, stem, options, answer_index, held_out=False, context="") -> Item:
    """A multiple-choice item: options rendered as `A. text` lines; gold is the option list and the right letter."""
    if not 0 <= answer_index < len(options) or len(options) > len(LETTERS) or len(options) < 2:
        raise ValueError(f"{source}/{item_id}: answer index {answer_index} for {len(options)} options")
    head = f"{context}\n\n{stem}" if context else stem
    problem = f"{head}\n\nOptions:\n{options_block(options)}"
    gold = {"options": list(options), "answer": LETTERS[answer_index]}
    return Item(source, item_id, domain, "choice", problem, gold, head, head, held_out)


def _verdict(source, item_id, domain, problem, labels, answer, held_out=False, group=None, stem=None) -> Item:
    """A verdict item; `stem` is its content without fixed wording (the theory and statement, the premises and
    conclusion), so that the shared wording does not make different items look alike."""
    if answer not in labels:
        raise ValueError(f"{source}/{item_id}: verdict {answer!r} not among {labels}")
    return Item(source, item_id, domain, "verdict", problem, {"labels": list(labels), "answer": answer},
                stem or problem, group or problem, held_out)


def _open(source, item_id, domain, problem, gold, held_out=False) -> Item:
    return Item(source, item_id, domain, "open", problem, gold, problem, problem, held_out)


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (balanced braces), or None."""
    start = text.rfind("\\boxed")
    while start != -1:
        i = start + len("\\boxed")
        while i < len(text) and text[i] == " ":
            i += 1
        if i < len(text) and text[i] == "{":
            depth = 0
            for j in range(i, len(text)):
                depth += {"{": 1, "}": -1}.get(text[j], 0)
                if depth == 0:
                    return text[i + 1:j].strip()
            return None
        start = text.rfind("\\boxed", 0, start)
    return None


# ---------------------------------------------------------------------------------------------------------------
# Math

MATH_CONFIGS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                "prealgebra", "precalculus")


def _math(split: str) -> list[Item]:
    source = "math_train" if split == "train" else "math_test"
    out = []
    for config in MATH_CONFIGS:
        for i, r in enumerate(_load("EleutherAI/hendrycks_math", config, split)):
            out.append(_open(source, f"{config}/{i}", "math", r["problem"], last_boxed(r["solution"]),
                             held_out=split == "test"))
    return out


@functools.lru_cache(maxsize=None)
def math500_problems() -> tuple[str, ...]:
    """MATH-500 problem texts; all 500 are in the MATH test split, where they keep their Round 1 parts."""
    return tuple(r["problem"] for r in _load("HuggingFaceH4/MATH-500", None, "test"))


def _deepscaler() -> list[Item]:
    rows = _load("agentica-org/DeepScaleR-Preview-Dataset", None, "train")
    return [_open("deepscaler", str(i), "math", r["problem"], r["answer"]) for i, r in enumerate(rows)]


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def _gsm8k(split: str) -> list[Item]:
    source = f"gsm8k_{split}"
    return [_open(source, str(i), "math", r["question"], gsm8k_gold(r["answer"]), held_out=split == "test")
            for i, r in enumerate(_load("openai/gsm8k", "main", split))]


AQUA_TRAIN_SAMPLE = 10_000
_AQUA_OPTION = re.compile(r"^\s*\(?([A-E])\s*\)\s*(.*)$", re.S)


def _aqua() -> list[Item]:
    out = []
    for split in ("train", "validation", "test"):
        rows = _load("deepmind/aqua_rat", "raw", split)
        idx = _sample(len(rows), AQUA_TRAIN_SAMPLE, "aqua") if split == "train" else range(len(rows))
        for i in idx:
            r = rows[i]
            parsed = [_AQUA_OPTION.match(o) for o in r["options"]]
            if not all(parsed) or [m.group(1) for m in parsed] != list(LETTERS[:len(parsed)]):
                continue  # options not labelled A), B), ... in order: no reliable gold
            out.append(_choice("aqua", f"{split}/{i}", "math", r["question"].strip(),
                               [m.group(2).strip() for m in parsed], LETTERS.index(r["correct"].strip()),
                               held_out=split != "train"))
    return out


def _theoremqa() -> list[Item]:
    """Items with a picture are dropped; `option` answers refer to options written into the question text and are
    dropped too; `bool` answers become True/False verdicts."""
    out = []
    for i, r in enumerate(_load("TIGER-Lab/TheoremQA", None, "test")):
        if r["Picture"] is not None and str(r["Picture"]) != "None":
            continue
        kind, answer = r["Answer_type"], str(r["Answer"]).strip()
        if kind == "option":
            continue
        if kind == "bool":
            out.append(_verdict("theoremqa", str(i), "math", r["Question"], ("True", "False"), answer))
        else:
            out.append(_open("theoremqa", str(i), "math", r["Question"], answer))
    return out


_AGIEVAL_CHOICE = re.compile(r"^\(([A-J])\)\s*(.*)$", re.S)


def _agieval(source: str, dataset: str, domain: str) -> list[Item]:
    """AGIEval sets: `query` repeats the options and ends with "A: Among A through E, the answer is"; the clean question
    is the text before "Answer Choices:" without the leading "Q: ", options come from `choices`, gold from `gold`."""
    out = []
    for i, r in enumerate(_load(dataset, None, "test")):
        stem = r["query"].split("Answer Choices:")[0].strip()
        stem = re.sub(r"^Q:\s*", "", stem)
        stem = re.sub(r"\s*Q:\s*", "\n\n", stem).strip()  # passage, then the question (LSAT)
        parsed = [_AGIEVAL_CHOICE.match(c.strip()) for c in r["choices"]]
        if not all(parsed) or len(r["gold"]) != 1:
            continue
        out.append(_choice(source, str(i), domain, stem, [m.group(2).strip() for m in parsed], r["gold"][0]))
    return out


def _number(x) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else repr(x)


def _svamp() -> list[Item]:
    out = []
    for split in ("train", "test"):
        for r in _load("ChilleD/SVAMP", None, split):
            out.append(_open("svamp", r["ID"], "math", r["question_concat"].strip(), _number(r["Answer"])))
    return out


def _omni() -> list[Item]:
    """All of Omni-MATH; the items that DeepScaleR contains are removed by de-duplication (`external.py`)."""
    return [_open("omni_rest", str(i), "math", r["problem"], r["answer"])
            for i, r in enumerate(_load("KbsdJames/Omni-MATH", None, "test"))]


# ---------------------------------------------------------------------------------------------------------------
# Code. Gold in the verifier's format (evotaste/verify/judge.py): {"harness": "pytest" | "assert" | "check", "tests",
# "reference", "entry_point"}, plus "signature" (what the answer instruction names) and, for LeetCode, "prelude" (the
# imports and helper classes its problems assume, to be prepended to the program). Training items keep the verifier's
# default limits (10 s, 1 GiB); evaluation sources get room, since a few of their references need more than 1 GiB on
# the largest inputs or 10 s on a loaded machine.
EVAL_CODE_LIMITS = {"time_limit_s": 30, "memory_mb": 2048}


def _code(source, item_id, problem, gold) -> Item:
    if SOURCES[source].role == "unseen":
        gold = {**gold, **EVAL_CODE_LIMITS}
    return Item(source, item_id, "code", "code", problem, gold, problem, problem)


def _kodcode() -> list[Item]:
    out = []
    for r in _load("KodCode/KodCode-Light-RL-10K", None, "train"):
        info = ast.literal_eval(r["test_info"]) if isinstance(r["test_info"], str) else r["test_info"]
        if len(info) != 1:
            continue
        gold = {"harness": "pytest", "tests": r["test"], "reference": r["solution"],
                "entry_point": info[0]["function_name"], "signature": info[0]["function_declaration"].strip()}
        out.append(_code("kodcode", r["question_id"], r["question"].strip(), gold))
    return out


def _humaneval_plus() -> list[Item]:
    out = []
    for r in _load("evalplus/humanevalplus", None, "test"):
        problem = f"Complete the following Python function.\n\n```python\n{r['prompt'].strip()}\n```"
        gold = {"harness": "check", "tests": r["test"], "reference": r["prompt"] + r["canonical_solution"],
                "entry_point": r["entry_point"], "signature": None}
        out.append(_code("humaneval_plus", r["task_id"].split("/")[-1], problem, gold))
    return out


def _mbpp_plus() -> list[Item]:
    """The prompt and the first of the original assertions, as MBPP is usually posed; MBPP+'s own test calls the
    function by name."""
    out = []
    for r in _load("evalplus/mbppplus", None, "test"):
        tests = ast.literal_eval(r["test_list"]) if isinstance(r["test_list"], str) else r["test_list"]
        imports = ast.literal_eval(r["test_imports"]) if isinstance(r["test_imports"], str) else r["test_imports"]
        defined = re.findall(r"^def ([A-Za-z_]\w*)\(", r["code"], re.M)
        called = [n for n in defined if re.search(rf"\b{n}\(", tests[0])]
        if not called:
            continue
        signature = re.search(rf"^def {called[0]}\(.*?\)\s*(?:->[^:]*)?:", r["code"], re.M | re.S)
        problem = f"{r['prompt'].strip()}\n\nYour code should pass this test:\n{tests[0]}"
        gold = {"harness": "assert", "tests": "\n".join([*imports, r["test"]]), "reference": r["code"],
                "entry_point": called[0], "signature": signature.group(0) if signature else None}
        out.append(_code("mbpp_plus", str(r["task_id"]), problem, gold))
    return out


LEETCODE_MIN_ASSERTS = 5


def _leetcode() -> list[Item]:
    """Both splits. The generated tests are one assertion per line; those expecting None (the reference's reply to an
    invalid input, in 68 train problems) are dropped, and a problem keeps at least LEETCODE_MIN_ASSERTS assertions or
    is dropped. The dataset's import and helper preamble is the prelude; the tests end by calling `check` on the entry
    point (`Solution().method`)."""
    out = []
    for split in ("train", "test"):
        for r in _load("newfacade/LeetCodeDataset", None, split):
            head, *body = r["test"].rstrip().splitlines()
            asserts = [line for line in body if line.strip()]
            kept = [line for line in asserts if "== None" not in line]
            if (head.strip() != "def check(candidate):" or not all(a.startswith("    assert ") for a in asserts)
                    or len(kept) < LEETCODE_MIN_ASSERTS):
                continue
            problem = (f"{r['problem_description'].strip()}\n\nUse this starter code:\n```python\n"
                       f"{r['starter_code'].rstrip()}\n```")
            tests = "\n".join([head, *kept]) + f"\n\n\ncheck({r['entry_point']})\n"
            gold = {"harness": "assert", "tests": tests, "reference": r["completion"], "entry_point": r["entry_point"],
                    "signature": None, "prelude": r["prompt"]}
            out.append(_code("leetcode", r["task_id"], problem, gold))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Logic and science


def _logiqa() -> list[Item]:
    out = []
    for split in ("train", "test"):
        for i, r in enumerate(_load("lucasmccabe/logiqa", None, split, revision="refs/convert/parquet")):
            options = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
            out.append(_choice("logiqa", f"{split}/{i}", "logic", r["query"].strip(), [o.strip() for o in options],
                               int(r["correct_option"]), held_out=split == "test", context=r["context"].strip()))
    return out


PROOFWRITER_CONFIGS = ("depth-0", "depth-1", "depth-2", "depth-3", "depth-5")
PROOFWRITER_PER_DEPTH = {"train": 1000, "test": 200}  # question depths 0-5


def _proofwriter() -> list[Item]:
    """A sample across question depths 0-5 (equal numbers per depth), at most one question per theory, from the
    synthetic open-world configurations; train and test are ProofWriter's own theory-disjoint splits, and the part of
    a test item is decided by its theory, so all questions on one theory share a part."""
    out = []
    for split, per_depth in PROOFWRITER_PER_DEPTH.items():
        rows = _load("tasksource/proofwriter", None, split)
        meta = rows.select_columns(["config", "QDep"])
        by_depth: dict[int, list[int]] = {}
        for i, (c, d) in enumerate(zip(meta["config"], meta["QDep"])):
            if c in PROOFWRITER_CONFIGS and int(d) <= 5:
                by_depth.setdefault(int(d), []).append(i)
        chosen, seen = {}, set()
        for d in sorted(by_depth):
            order = by_depth[d][:]
            _rng("proofwriter", split, d).shuffle(order)
            taken = 0
            for i in order:
                if taken == per_depth:
                    break
                r = rows[i]
                if r["theory"] in seen:
                    continue
                seen.add(r["theory"])
                chosen[i] = r
                taken += 1
        for i in sorted(chosen):
            r = chosen[i]
            problem = (f"Facts and rules:\n{r['theory'].strip()}\n\nStatement: {r['question'].strip()}\n\n"
                       "Assume only the facts and rules above (anything they do not settle is unknown). Is the "
                       "statement True, False or Unknown?")
            out.append(_verdict("proofwriter", f"{split}/{i}", "logic", problem, VERDICT_PROOFWRITER, r["answer"],
                                held_out=split == "test", group=r["theory"],
                                stem=f"{r['theory'].strip()}\n{r['question'].strip()}"))
    return out


def _arc() -> list[Item]:
    out = []
    for split in ("train", "test"):
        for r in _load("allenai/ai2_arc", "ARC-Challenge", split):
            labels = list(r["choices"]["label"])
            if r["answerKey"] not in labels:
                continue
            out.append(_choice("arc", r["id"], "logic", r["question"].strip(), list(r["choices"]["text"]),
                               labels.index(r["answerKey"]), held_out=split == "test"))
    return out


# MMLU-Pro questions that repeat a MATH-500 problem; dropped since Round 1, which had no foundation part of MATH-500 to
# share them with.
MMLU_PRO_DROPPED = {"7744", "8289", "8591", "8823", "8830"}


def _mmlu_pro() -> list[Item]:
    out = []
    for r in _load("TIGER-Lab/MMLU-Pro", None, "test"):
        if str(r["question_id"]) in MMLU_PRO_DROPPED:
            continue
        item = _choice("mmlu_pro", str(r["question_id"]), "logic", r["question"], list(r["options"]),
                       LETTERS.index(r["answer"]))
        out.append(item)
    return out


def _folio() -> list[Item]:
    out = []
    for split in ("train", "validation"):
        for r in _load("tasksource/folio", None, split):
            problem = (f"Premises:\n{r['premises'].strip()}\n\nConclusion: {r['conclusion'].strip()}\n\n"
                       "Based only on the premises, is the conclusion True, False or Uncertain?")
            out.append(_verdict("folio", f"{split}/{r['example_id']}", "logic", problem, VERDICT_FOLIO, r["label"],
                                stem=f"{r['premises'].strip()}\n{r['conclusion'].strip()}"))
    return out


FLD_SAMPLE = 1000


def _fld() -> list[Item]:
    rows = _load("hitachi-nlp/FLD.v2", None, "test")
    out = []
    for i in _sample(len(rows), FLD_SAMPLE, "fld"):
        r = rows[i]
        facts = re.sub(r"\s*(sent\d+:)", r"\n\1", r["context"]).strip()
        problem = (f"Facts:\n{facts}\n\nHypothesis: {r['hypothesis'].strip()}\n\nUsing only these facts, is the "
                   "hypothesis PROVED, DISPROVED or UNKNOWN (neither provable nor disprovable)?")
        out.append(_verdict("fld", str(i), "logic", problem, VERDICT_FLD, r["proof_label"],
                            stem=f"{facts}\n{r['hypothesis'].strip()}"))
    return out


# BBH tasks: (source id, configs of the unit, kind). Verdict tasks carry their label set; dyck_languages and
# word_sorting have free-form string targets, compared exactly after normalisation (kind "procedural", see tasks.py).
# A Dyck answer may be the completed sequence or only its closing part (the reference is the completed sequence, which
# is balanced and so can be boxed); a word-sorting answer is its list of words, however separated.
BBH = [
    ("bbh_boolean_expressions", ("boolean_expressions",), "verdict"),
    ("bbh_dyck_languages", ("dyck_languages",), "exact"),
    ("bbh_formal_fallacies", ("formal_fallacies",), "verdict"),
    ("bbh_logical_deduction", ("logical_deduction_three_objects", "logical_deduction_five_objects",
                               "logical_deduction_seven_objects"), "choice"),
    ("bbh_multistep_arithmetic_two", ("multistep_arithmetic_two",), "open"),
    ("bbh_navigate", ("navigate",), "verdict"),
    ("bbh_object_counting", ("object_counting",), "open"),
    ("bbh_tracking_shuffled_objects", ("tracking_shuffled_objects_three_objects",
                                       "tracking_shuffled_objects_five_objects",
                                       "tracking_shuffled_objects_seven_objects"), "choice"),
    ("bbh_web_of_lies", ("web_of_lies",), "verdict"),
    ("bbh_word_sorting", ("word_sorting",), "exact"),
    ("bbh_date_understanding", ("date_understanding",), "choice"),
    ("bbh_temporal_sequences", ("temporal_sequences",), "choice"),
]
_BBH_LABELS = {"bbh_boolean_expressions": ("True", "False"), "bbh_formal_fallacies": ("valid", "invalid"),
               "bbh_navigate": ("Yes", "No"), "bbh_web_of_lies": ("Yes", "No")}
_BBH_OPTION = re.compile(r"^\(([A-R])\)\s*(.*)$")


def _bbh(source: str, configs: tuple[str, ...], kind: str) -> list[Item]:
    out = []
    for config in configs:
        for i, r in enumerate(_load("lukaemon/bbh", config, "test")):
            item_id = f"{config}/{i}" if len(configs) > 1 else str(i)
            text, target = r["input"].strip(), r["target"].strip()
            stem, _, options = text.partition("\nOptions:\n")
            if kind == "choice":
                parsed = [_BBH_OPTION.match(line.strip()) for line in options.splitlines() if line.strip()]
                letter = re.fullmatch(r"\(([A-R])\)", target)
                if not parsed or not all(parsed) or letter is None:
                    raise ValueError(f"{source}/{item_id}: unparsable options or target {target!r}")
                out.append(_choice(source, item_id, "logic", stem.strip(), [m.group(2).strip() for m in parsed],
                                   [m.group(1) for m in parsed].index(letter.group(1))))
            elif kind == "verdict":
                out.append(_verdict(source, item_id, "logic", stem.strip(), _BBH_LABELS[source], target))
            elif kind == "open":
                out.append(_open(source, item_id, "logic", stem.strip(), target))
            elif source == "bbh_dyck_languages":  # the completed sequence, or only its closing part
                opened = stem.split("Input:")[-1].strip()
                gold = {"exact": f"{opened} {target}", "accept": [target], "compare": "brackets"}
                out.append(Item(source, item_id, "logic", "procedural", stem.strip(), gold, stem.strip(), stem.strip()))
            else:  # word sorting: the sorted words, whatever separates them
                gold = {"exact": target, "accept": [], "compare": "words"}
                out.append(Item(source, item_id, "logic", "procedural", stem.strip(), gold, stem.strip(), stem.strip()))
    return out


MEDMCQA_SAMPLE = 1000
SCIENCEQA_SAMPLE = 1000


def _medmcqa() -> list[Item]:
    rows = _load("openlifescienceai/medmcqa", None, "validation")
    return [_choice("medmcqa", rows[i]["id"], "logic", rows[i]["question"].strip(),
                    [rows[i][k].strip() for k in ("opa", "opb", "opc", "opd")], int(rows[i]["cop"]))
            for i in _sample(len(rows), MEDMCQA_SAMPLE, "medmcqa")]


def _scienceqa() -> list[Item]:
    """Text-only questions of the test split (no image); the hint, when present, is the context."""
    rows = _load("derek-thomas/ScienceQA", None, "test", revision="refs/convert/parquet")
    text_only = [i for i, image in enumerate(rows["image"]) if image is None]
    out = []
    for j in _sample(len(text_only), SCIENCEQA_SAMPLE, "scienceqa"):
        i = text_only[j]
        r = rows[i]
        out.append(_choice("scienceqa", str(i), "logic", r["question"].strip(), list(r["choices"]), int(r["answer"]),
                           context=(r["hint"] or "").strip()))
    return out


def _openbookqa() -> list[Item]:
    out = []
    for r in _load("allenai/openbookqa", "main", "test"):
        labels = list(r["choices"]["label"])
        out.append(_choice("openbookqa", r["id"], "logic", r["question_stem"].strip(), list(r["choices"]["text"]),
                           labels.index(r["answerKey"])))
    return out


def _qasc() -> list[Item]:
    out = []
    for r in _load("allenai/qasc", None, "validation"):
        labels = list(r["choices"]["label"])
        out.append(_choice("qasc", r["id"], "logic", r["question"].strip(), list(r["choices"]["text"]),
                           labels.index(r["answerKey"])))
    return out


# ---------------------------------------------------------------------------------------------------------------
# The source table (docs/dev/ROUND2_SPEC.md). Order matters: when copies of one question remain in the same part, the
# copy of the earliest source is kept (Round 1 sources are listed first so that the sealed vault keeps its own copies).


def _table() -> list[Source]:
    t = [
        Source("gsm8k_train", "math", "train", "gsm8k_train", True, lambda: _gsm8k("train")),
        Source("gsm8k_test", "math", "held_out", "gsm8k_test", True, lambda: _gsm8k("test")),
        Source("math_train", "math", "train", "math_train", True, lambda: _math("train")),
        Source("math_test", "math", "held_out", "math_test", True, lambda: _math("test")),
        Source("mmlu_pro", "logic", "train", "mmlu_pro", True, _mmlu_pro),
        Source("deepscaler", "math", "train", "deepscaler", False, _deepscaler),
        Source("aqua", "math", "train", "aqua", True, _aqua),
        Source("kodcode", "code", "train", "kodcode", False, _kodcode),
        Source("logiqa", "logic", "train", "logiqa", True, _logiqa),
        Source("proofwriter", "logic", "train", "proofwriter", True, _proofwriter),
        Source("arc", "logic", "train", "arc", True, _arc),
        Source("theoremqa", "math", "unseen", "theoremqa", False, _theoremqa),
        Source("sat_math", "math", "unseen", "sat_math", False,
               lambda: _agieval("sat_math", "hails/agieval-sat-math", "math")),
        Source("svamp", "math", "unseen", "svamp", False, _svamp),
        Source("omni_rest", "math", "unseen", "omni_rest", False, _omni),
        Source("humaneval_plus", "code", "unseen", "humaneval_plus", False, _humaneval_plus),
        Source("mbpp_plus", "code", "unseen", "mbpp_plus", False, _mbpp_plus),
        Source("leetcode", "code", "unseen", "leetcode", False, _leetcode),
        Source("folio", "logic", "unseen", "folio", False, _folio),
        Source("lsat_lr", "logic", "unseen", "lsat_lr", False,
               lambda: _agieval("lsat_lr", "hails/agieval-lsat-lr", "logic")),
        Source("lsat_ar", "logic", "unseen", "lsat_ar", False,
               lambda: _agieval("lsat_ar", "hails/agieval-lsat-ar", "logic")),
        Source("fld", "logic", "unseen", "fld", False, _fld),
    ]
    t += [Source(s, "logic", "unseen", s, False, functools.partial(_bbh, s, c, k)) for s, c, k in BBH]
    t += [Source("medmcqa", "logic", "unseen", "medmcqa", False, _medmcqa),
          Source("scienceqa", "logic", "unseen", "scienceqa", False, _scienceqa),
          Source("openbookqa", "logic", "unseen", "openbookqa", False, _openbookqa),
          Source("qasc", "logic", "unseen", "qasc", False, _qasc)]
    return t


SOURCES: dict[str, Source] = {s.id: s for s in _table()}
TRAINING_SOURCES = tuple(s.id for s in SOURCES.values() if s.role == "train")
# Datasets read by the loaders, for the one-time download through the mirror: (name, configs, revision).
DATASETS = [("EleutherAI/hendrycks_math", MATH_CONFIGS, None), ("HuggingFaceH4/MATH-500", (None,), None),
            ("agentica-org/DeepScaleR-Preview-Dataset", (None,), None), ("openai/gsm8k", ("main",), None),
            ("deepmind/aqua_rat", ("raw",), None), ("KodCode/KodCode-Light-RL-10K", (None,), None),
            ("lucasmccabe/logiqa", (None,), "refs/convert/parquet"), ("tasksource/proofwriter", (None,), None),
            ("allenai/ai2_arc", ("ARC-Challenge",), None), ("TIGER-Lab/MMLU-Pro", (None,), None),
            ("TIGER-Lab/TheoremQA", (None,), None), ("hails/agieval-sat-math", (None,), None),
            ("ChilleD/SVAMP", (None,), None), ("KbsdJames/Omni-MATH", (None,), None),
            ("evalplus/humanevalplus", (None,), None), ("evalplus/mbppplus", (None,), None),
            ("newfacade/LeetCodeDataset", (None,), None), ("tasksource/folio", (None,), None),
            ("hails/agieval-lsat-lr", (None,), None), ("hails/agieval-lsat-ar", (None,), None),
            ("hitachi-nlp/FLD.v2", (None,), None), ("lukaemon/bbh", tuple(c for _, cs, _ in BBH for c in cs), None),
            ("openlifescienceai/medmcqa", (None,), None), ("derek-thomas/ScienceQA", (None,), "refs/convert/parquet"),
            ("allenai/openbookqa", ("main",), None), ("allenai/qasc", (None,), None)]
