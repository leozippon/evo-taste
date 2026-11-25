"""Calibration of the verifiers on real references (no model involved):

    python -m evotaste.verify.selfcheck [--per-source 300] [--processes 32]

For a sample of each source: (a) the share of references that verify against themselves in every wrapper of their
kind (gold_check); (b) the share of perturbed references rejected (open: off by one either way, sign flipped,
doubled, and for decimals one unit more in the last digit; choice: another letter and another option's text;
verdict: every other label); (c) for code, the share of reference solutions passing their own tests and the time
per item; (d) the throughput of verify_many per kind. Reads only the local Hugging Face cache and only parts that
are not sealed: training splits, MMLU-Pro's validation split, and the references of HumanEval+, MBPP+ and TheoremQA
(the same gold self-check the build runs on every item). Results in docs/dev/VERIFY_NOTES.md.
"""

from __future__ import annotations

import argparse
import ast
import collections
import os
import random
import re
import time

import sympy

from evotaste import verify
from evotaste.verify import Job, LETTERS, WRAPPERS
from evotaste.verify.equivalence import _numbers, parse_answer
from evotaste.verify.pool import Pool

FILLER = "Let us work through the problem step by step, checking each condition as we go. " * 20


def _last_boxed(s: str) -> str | None:
    found = WRAPPERS["box"].find(s)
    return found.content if found else None


def _list(x):
    return ast.literal_eval(x) if isinstance(x, str) else list(x)


def load(per_source: int, seed: int = 0) -> dict[str, tuple[str, list]]:
    """source -> (kind, golds), a seeded sample of each source."""
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    rng = random.Random(seed)

    def sample(rows):
        rows = list(rows)
        return rng.sample(rows, min(per_source, len(rows)))

    out: dict[str, tuple[str, list]] = {}
    configs = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
               "prealgebra", "precalculus"]
    math_rows = [r for c in configs for r in load_dataset("EleutherAI/hendrycks_math", c, split="train")]
    out["math_train"] = ("open", [g for r in sample(math_rows) if (g := _last_boxed(r["solution"]))])
    out["deepscaler"] = ("open", [r["answer"] for r in sample(load_dataset(
        "agentica-org/DeepScaleR-Preview-Dataset", split="train"))])
    out["gsm8k_train"] = ("open", [r["answer"].split("####")[-1].strip().replace(",", "") for r in sample(
        load_dataset("openai/gsm8k", "main", split="train"))])
    out["theoremqa"] = ("open", [str(r["Answer"]) for r in sample(
        r for r in load_dataset("TIGER-Lab/TheoremQA", split="test") if r["Picture"] is None)])
    out["aqua"] = ("choice", [{"options": [re.sub(r"^[A-E]\)", "", o).strip() for o in _list(r["options"])],
                               "answer": r["correct"]} for r in sample(load_dataset("deepmind/aqua_rat", "raw",
                                                                                     split="train"))])
    out["logiqa"] = ("choice", [{"options": _list(r["options"]), "answer": LETTERS[int(r["correct_option"])]}
                                for r in sample(load_dataset("lucasmccabe/logiqa", split="train",
                                                             revision="refs/convert/parquet"))])
    arc = []
    for r in sample(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")):
        ch = r["choices"] if isinstance(r["choices"], dict) else ast.literal_eval(r["choices"])
        arc.append({"options": list(ch["text"]), "answer": LETTERS[list(ch["label"]).index(r["answerKey"])]})
    out["arc"] = ("choice", arc)
    out["mmlu_pro_val"] = ("choice", [{"options": _list(r["options"]), "answer": r["answer"]} for r in sample(
        load_dataset("TIGER-Lab/MMLU-Pro", split="validation"))])
    out["proofwriter"] = ("verdict", [{"labels": ["True", "False", "Unknown"], "answer": r["answer"]} for r in sample(
        load_dataset("tasksource/proofwriter", split="train").select(range(20000)))])
    out["kodcode"] = ("code", [{"harness": "pytest", "tests": r["test"], "reference": r["solution"]}
                               for r in sample(load_dataset("KodCode/KodCode-Light-RL-10K", split="train"))])
    out["humaneval_plus"] = ("code", [{"harness": "check", "tests": r["test"], "entry_point": r["entry_point"],
                                       "reference": r["prompt"] + r["canonical_solution"]}
                                      for r in sample(load_dataset("evalplus/humanevalplus", split="test"))])
    out["mbpp_plus"] = ("code", [{"harness": "assert", "tests": r["test"], "reference": r["code"]}
                                 for r in sample(load_dataset("evalplus/mbppplus", split="test"))])
    return out


def _bump_last_digit(gold: str) -> str | None:
    m = list(re.finditer(r"\d\.\d+|\.\d+", gold))
    if not m:
        return None
    end = m[-1].end() - 1
    return gold[:end] + str((int(gold[end]) + 1) % 10) + gold[end + 1:] if gold[end] != "9" else None


def perturbations(kind: str, gold) -> list[str]:
    """Wrong answers close to the reference, as box contents."""
    if kind == "choice":
        right = gold["options"][LETTERS.index(gold["answer"])].strip().lower()
        others = [i for i, o in enumerate(gold["options"])
                  if LETTERS[i] != gold["answer"] and o.strip().lower() != right]
        return [LETTERS[i] for i in others[:1]] + [gold["options"][i] for i in others[1:2]]
    if kind == "verdict":
        return [lab for lab in gold["labels"] if lab != gold["answer"]]
    parsed = parse_answer(gold)
    if parsed is None:
        return []
    g = parsed.expr
    out = []
    if _numbers(g) and isinstance(g, sympy.Expr):
        out = [g + 1, g - 1] + ([-g, 2 * g] if g != 0 else [])
    elif isinstance(g, sympy.Expr) and g.free_symbols:
        out = [g + 1, 2 * g]
    out = [sympy.latex(x) for x in out]
    if parsed.decimal and (bumped := _bump_last_digit(gold)):
        out.append(bumped)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-source", type=int, default=300)
    ap.add_argument("--processes", type=int, default=verify.pool.PROCESSES)
    args = ap.parse_args(argv)
    verify._POOL = Pool(args.processes)
    data = load(args.per_source)

    print(f"(a) references verifying against themselves in every wrapper; (b) perturbations rejected "
          f"[{args.processes} processes]")
    for source, (kind, golds) in data.items():
        t0 = time.monotonic()
        reasons = verify.gold_check([(kind, g) for g in golds])
        seconds = time.monotonic() - t0
        ok = [g for g, r in zip(golds, reasons) if r is None]
        line = f"  {source:15} {kind:8} {len(ok)}/{len(golds)} = {100 * len(ok) / len(golds):.1f}%"
        if kind == "code":
            line += f"  ({seconds / len(golds) * args.processes:.2f} s per item per process, {seconds:.0f} s wall)"
        else:
            jobs = [Job(kind, f"{FILLER}\n\\boxed{{{p}}}", g, "box") for g in ok for p in perturbations(kind, g)]
            verdicts = verify.verify_many(jobs)
            credited = [(j.gold, j.text.rsplit("boxed", 1)[1]) for j, v in zip(jobs, verdicts) if v.reward]
            line += f"; perturbed {len(jobs) - len(credited)}/{len(jobs)} rejected"
            if credited:
                line += f"  CREDITED: {credited[:5]}"
        print(line)
        failures = collections.Counter(re.sub(r"\d+", "#", r)[:90] for r in reasons if r)
        for reason, n in failures.most_common(4):
            print(f"      {n:4}  {reason}")
        examples = [g for g, r in zip(golds, reasons) if r][:6]
        if examples and kind != "code":
            print(f"      e.g. {examples}")

    print("\n(d) throughput of verify_many (responses with 1.6k characters of reasoning; open answers rewritten by "
          "sympy, so that no parse is cached)")
    open_golds = [(g, p) for s in ("math_train", "deepscaler", "gsm8k_train") for g in data[s][1]
                  if (p := parse_answer(g)) is not None]
    workloads = {
        "open, box": [Job("open", f"{FILLER}\n\\boxed{{{sympy.latex(p.expr)}}}", g, "box", ("direct", "program"))
                      for g, p in open_golds],
        "open, program": [Job("open", f"{FILLER}\n```python\nprint({g!r})\n```", g, "box", ("direct", "program"))
                          for g in data["gsm8k_train"][1]],
        "choice": [Job("choice", f"{FILLER}\n\\boxed{{{g['answer']}}}", g, "box") for s in ("aqua", "logiqa", "arc")
                   for g in data[s][1]],
        "verdict": [Job("verdict", f"{FILLER}\n\\boxed{{{g['answer']}}}", g, "box") for g in data["proofwriter"][1]],
        "code (kodcode)": [Job("code", WRAPPERS["code"].render(g["reference"]), g, "code") for g in data["kodcode"][1]],
    }
    for name, jobs in workloads.items():
        t0 = time.monotonic()
        verdicts = verify.verify_many(jobs)
        seconds = time.monotonic() - t0
        timeouts = sum(v.error == "timeout" for v in verdicts)
        print(f"  {name:16} {len(jobs):5} jobs in {seconds:6.1f} s = {len(jobs) / seconds:7.1f} jobs/s; "
              f"reward 1: {sum(v.reward for v in verdicts):.0f}; timeouts: {timeouts}")


if __name__ == "__main__":
    main()
