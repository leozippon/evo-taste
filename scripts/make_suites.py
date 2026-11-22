"""Assign reasoning-gym families to the train, selection and blind suites; writes configs/suites.yaml.

Rules (docs/dev/ENGINE_SPEC.md, "Suite assignment"): exclude unusable families; keep near-duplicate families
together; hold out one whole category each for selection, blind 1 and blind 2; deal the remaining units, stratified
by category and tier, to train/selection/blind1/blind2 in the ratio 2:1:1:1. Besides the calibration statistics,
every candidate family must pass the engine's own reward check on fresh instances: the gold answer wrapped in
<answer> tags earns reward 1, an empty or junk answer earns 0.

Usage (compute host, from the repository root): python scripts/make_suites.py
"""

import collections
import json
import random
import sys
from pathlib import Path

import yaml

from evotaste import external, tasks

CALIBRATION = Path("results/calibration/1.7b.jsonl")
CATALOGUE = Path("results/calibration/catalogue.json")
OUT = Path("configs/suites.yaml")
SEED = 20261122
MAX_MEAN_PROMPT_TOKENS = 700
CEILING_EXACT_RATE = 0.9
CHECK_INSTANCES = 64  # per family, for the reward check
SPECIAL_CONFIG = {"composite": "needs an explicit dataset list (special configuration)"}
DUPLICATES_EXTERNAL = {"gsm_symbolic": "duplicates the GSM8K foundation probe"}
NEAR_DUPLICATES = [
    ["chain_sum", "decimal_chain_sum"], ["basic_arithmetic", "decimal_arithmetic"],
    ["simple_integration", "intermediate_integration"], ["simple_equations", "polynomial_equations"],
    ["sudoku", "mini_sudoku"], ["gcd", "lcm"], ["countdown", "puzzle24"], ["arc_1d", "rearc", "arc_agi"],
    ["word_sequence_reversal", "spell_backward"], ["number_sorting", "word_sorting"],
    ["simple_geometry", "advanced_geometry"], ["string_manipulation", "string_synthesis", "string_splitting", "ab"],
    ["rotate_matrix", "spiral_matrix", "manipulate_matrix"],
]
HELD_OUT_CANDIDATES = ["algebra", "cognition", "graphs", "logic"]
MIN_UNITS_PER_CATEGORY = 4  # smaller categories are pooled into one stratum
DEAL = ["train", "selection", "train", "blind1", "blind2"]  # 2:1:1:1, continued across strata
EXTERNAL = {"foundation": ["gsm8k_train", "mmlu_pro"], "blind1": ["gsm8k_test", "math500", "mmlu_pro"],
            "blind2": ["gsm8k_test", "math500", "mmlu_pro"]}


def calibration_stats() -> dict[str, dict]:
    rows = collections.defaultdict(list)
    for line in CALIBRATION.open():
        r = json.loads(line)
        rows[r["dataset"]].append(r)
    stats = {}
    for name, rs in rows.items():
        n = len(rs)
        stats[name] = {
            "instances": n,
            "prompt_tokens_mean": round(sum(r["prompt_tokens"] for r in rs) / n, 1),
            "completion_tokens_mean": round(sum(r["completion_tokens"] for r in rs) / n, 1),
            "exact_rate": round(sum(r["score"] == 1.0 for r in rs) / n, 4),
            "score_mean": round(sum(r["score"] for r in rs) / n, 4),
            "no_answer_rate": round(sum(r["extracted"] is None for r in rs) / n, 4),
            "truncated_rate": round(sum(r.get("finish_reason") == "length" for r in rs) / n, 4),
        }
    return stats


def reward_check(check_tasks: list[tasks.Task]) -> dict:
    """Engine reward of the wrapped gold answer (must be 1) and of empty / junk answers (must be 0)."""
    def rewards(answer_of, keep=lambda t: True):
        kept = [t for t in check_tasks if keep(t)]
        return [s.reward for s in tasks.score_many([(t, f"<answer>{answer_of(t)}</answer>", "stop") for t in kept])]

    gold = rewards(lambda t: t.answer)
    empty = rewards(lambda t: "", lambda t: t.answer.strip())
    junk = rewards(lambda t: "zzz", lambda t: t.answer.strip() != "zzz")
    return {"instances": len(check_tasks), "gold_reward_min": min(gold), "gold_reward_mean": round(sum(gold) / len(gold), 4),
            "empty_reward_max": max(empty, default=0.0), "junk_reward_max": max(junk, default=0.0)}


def check_failure(check: dict) -> str | None:
    if check["gold_reward_min"] < 1.0:
        return f"reward check: wrapped gold answer earns reward 1 on only {check['gold_reward_mean']:.0%} of instances"
    if check["empty_reward_max"] > 0 or check["junk_reward_max"] > 0:
        return "reward check: an empty or junk answer earns reward 1"
    return None


def main():
    catalogue = {c["name"]: c for c in json.loads(CATALOGUE.read_text())}
    stats = calibration_stats()
    families, excluded = {}, {}
    for name in sorted(catalogue):
        c, s = catalogue[name], stats.get(name)
        reason = SPECIAL_CONFIG.get(name) or DUPLICATES_EXTERNAL.get(name)
        if reason is None and ("error" in c or s is None):
            reason = f"not calibrated ({c.get('error', 'no calibration records')})"
        if reason is None and s["prompt_tokens_mean"] > MAX_MEAN_PROMPT_TOKENS:
            reason = f"mean prompt {s['prompt_tokens_mean']:.0f} tokens > {MAX_MEAN_PROMPT_TOKENS}"
        if reason is None and c["gold_score_mean"] < 1.0:
            reason = f"scorer defective: stored gold answer scores {c['gold_score_mean']:.2f} on average"
        if reason is None and c["junk_score_max"] >= 0.5:
            reason = f"scorer defective: junk answer scores {c['junk_score_max']:.2f}"
        if reason is None and s["exact_rate"] >= CEILING_EXACT_RATE:
            reason = f"base model at ceiling (exact rate {s['exact_rate']:.2f})"
        check = None
        if reason is None:
            check = reward_check([tasks.instance("suitecheck", name, n) for n in range(CHECK_INSTANCES)])
            reason = check_failure(check)
        if reason is not None:
            excluded[name] = reason
            continue
        families[name] = {"category": c["category"], **s, "reward_check": check}
        print(f"{name}: ok {check}", flush=True)

    # Assignment units: near-duplicate groups (restricted to surviving families) and singletons.
    grouped = {f for g in NEAR_DUPLICATES for f in g}
    units = [[f for f in g if f in families] for g in NEAR_DUPLICATES] + [[f] for f in families if f not in grouped]
    units = sorted((sorted(u) for u in units if u), key=lambda u: u[0])
    for u in units:
        cats = {families[f]["category"] for f in u}
        if len(cats) != 1:
            sys.exit(f"near-duplicate unit {u} spans categories {cats}")
    unit_category = {tuple(u): families[u[0]]["category"] for u in units}

    rng = random.Random(SEED)
    held = sorted(HELD_OUT_CANDIDATES)
    rng.shuffle(held)
    held_out = {"selection": held[0], "blind1": held[1], "blind2": held[2], "pooled": held[3]}
    assignment = {}  # family -> suite
    for suite in ("selection", "blind1", "blind2"):
        for u in units:
            if unit_category[tuple(u)] == held_out[suite]:
                assignment.update({f: suite for f in u})
                for f in u:
                    families[f].update(stratum=f"held-out:{held_out[suite]}", unit=u)

    pool = [u for u in units if u[0] not in assignment]
    per_category = collections.Counter(unit_category[tuple(u)] for u in pool)
    strata = collections.defaultdict(list)
    for u in pool:
        cat = unit_category[tuple(u)]
        cat = cat if per_category[cat] >= MIN_UNITS_PER_CATEGORY else "pooled"
        tier = "reachable" if any(families[f]["exact_rate"] > 0 for f in u) else "floor"
        strata[f"{cat}/{tier}"].append(u)
    pointer = 0
    for name in sorted(strata):
        members = sorted(strata[name])
        rng.shuffle(members)
        for u in members:
            suite = DEAL[pointer % len(DEAL)]
            pointer += 1
            assignment.update({f: suite for f in u})
            for f in u:
                families[f].update(stratum=name, unit=u)
    for f in families:
        families[f]["suite"] = assignment[f]
        families[f]["tier"] = "reachable" if families[f]["exact_rate"] > 0 else "floor"

    suites = {s: sorted(f for f in families if assignment[f] == s) for s in ("train", "selection", "blind1", "blind2")}
    foundation_checks = {src: reward_check(external.items(src, "foundation")) for src in EXTERNAL["foundation"]}
    for src, check in foundation_checks.items():
        if check_failure(check):
            sys.exit(f"external foundation source {src}: {check_failure(check)}")

    header = (
        "# Generated by scripts/make_suites.py from the Qwen3-1.7B calibration; do not edit by hand.\n"
        "# Single source of the suite assignment for every component. Statistics below are those the script used.\n"
        "# Blind vaults (blind1, blind2) may only be read by evaluate.py through the blind guard.\n"
    )
    doc = {
        "generated": {"script": "scripts/make_suites.py", "calibration": str(CALIBRATION), "catalogue": str(CATALOGUE),
                      "seed": SEED, "rules": {"max_mean_prompt_tokens": MAX_MEAN_PROMPT_TOKENS,
                                              "ceiling_exact_rate": CEILING_EXACT_RATE,
                                              "reward_check_instances": CHECK_INSTANCES,
                                              "min_units_per_category": MIN_UNITS_PER_CATEGORY, "deal": DEAL}},
        "held_out_categories": held_out,
        "suites": suites,
        "external": EXTERNAL,
        "external_foundation_reward_check": foundation_checks,
        "families": {f: families[f] for f in sorted(families)},
        "excluded": excluded,
    }
    OUT.write_text(header + yaml.safe_dump(doc, sort_keys=False, width=120))
    print(json.dumps({s: len(v) for s, v in suites.items()}), "held out:", held_out, "excluded:", len(excluded))


if __name__ == "__main__":
    main()
