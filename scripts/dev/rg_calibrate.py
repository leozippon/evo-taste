"""reasoning-gym catalogue and base-model calibration.

  catalogue                    probe every registered dataset at its default config (subprocess per dataset, with timeout)
  run --model 1.7b --port P    one sample per instance through a running vLLM server -> results/calibration/<model>.jsonl
  table                        write docs/dev/REASONING_GYM_CALIBRATION.md from the catalogue and the runs
"""

import argparse
import asyncio
import json
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path("results/calibration")
N, SEED, GEN_TIMEOUT_S = 60, 1234, 600
MARK = "PROBE_RESULT "  # datasets may print to stdout themselves
DOC = Path("docs/dev/REASONING_GYM_CALIBRATION.md")


def knobs(name, config) -> str:
    """Curriculum attributes with their default-config values; config fields when there is no curriculum."""
    from reasoning_gym.coaching.attributes import RangeAttributeDefinition
    from reasoning_gym.factory import CURRICULA

    def fmt(v):
        if isinstance(v, (list, tuple, dict)):
            return "/".join(map(str, v)) if len(v) <= 3 else f"[{len(v)} values]"
        return str(v)

    if name in CURRICULA:
        parts = []
        for a in CURRICULA[name]().attributes.values():
            if isinstance(a, RangeAttributeDefinition):
                parts.append(f"{a.name}={fmt(getattr(config, a.lower_field_name))}–{fmt(getattr(config, a.upper_field_name))}")
            else:
                parts.append(f"{a.name}={fmt(getattr(config, a.field_name))}")
        return ", ".join(parts)
    fields = {k: v for k, v in vars(config).items() if k not in ("seed", "size")}
    return "(no curriculum) " + ", ".join(f"{k}={fmt(v)}"[:40] for k, v in fields.items())[:200]


def probe(name: str) -> dict:
    """Generate N instances, time it, and probe score_answer with perturbed gold answers."""
    import reasoning_gym
    from reasoning_gym.dataset import ProceduralDataset
    from reasoning_gym.factory import DATASETS

    cls, cfg_cls = DATASETS[name]
    rec = {"name": name, "category": cls.__module__.split(".")[1], "default_scorer": cls.score_answer is ProceduralDataset.score_answer}
    t0 = time.perf_counter()
    ds = reasoning_gym.create_dataset(name, size=N, seed=SEED)
    entries = [ds[i] for i in range(N)]
    rec["gen_s"] = round(time.perf_counter() - t0, 2)
    rec["knobs"] = knobs(name, ds.config)
    rec["example"] = entries[0]["question"]
    probes = set()

    def safe(answer, e):
        try:
            return float(ds.score_answer(answer, e))
        except Exception:
            probes.add("error")
            return 0.0

    gold = []
    for i, e in enumerate(entries[:20]):
        g = str(e["answer"])
        gold.append(safe(g, e))
        for c in (g + " x", "x " + g, g[:-1], g[1:], str(entries[(i + 1) % N]["answer"])):
            probes.add(round(safe(c, e), 3))
    rec["gold_score_mean"] = float(np.mean(gold))
    # A junk answer or a missing answer should score 0; anything else is a reward-design hazard.
    rec["junk_score_max"] = max(safe("zzz", e) for e in entries[:20])
    rec["none_score_max"] = max(safe(None, e) for e in entries[:20])
    rec["probe_scores"] = sorted(map(str, probes))
    return rec


def catalogue():
    from reasoning_gym.factory import DATASETS

    def one(name):
        try:
            p = subprocess.run([sys.executable, "-W", "ignore", __file__, "probe", name], capture_output=True, text=True,
                               timeout=GEN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return {"name": name, "error": f"timeout after {GEN_TIMEOUT_S}s"}
        if p.returncode != 0 or MARK not in p.stdout:
            return {"name": name, "error": (p.stderr.strip().splitlines() or ["no output"])[-1][:300]}
        return json.loads(p.stdout.rsplit(MARK, 1)[1].splitlines()[0])

    with ThreadPoolExecutor(24) as pool:
        recs = list(pool.map(one, sorted(DATASETS)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "catalogue.json").write_text(json.dumps(recs, indent=1))
    for r in recs:
        print(r["name"], r.get("category"), r.get("gen_s"), r.get("error", ""), r.get("probe_scores"))


class ScoreTimeout(Exception):
    pass


def score(ds, answer, entry):
    def alarm(*_):
        raise ScoreTimeout

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(10)
    try:
        return float(ds.score_answer(answer, entry)), None
    except ScoreTimeout:
        return 0.0, "score_timeout"
    except Exception as e:
        return 0.0, f"score_error: {type(e).__name__}: {e}"[:200]
    finally:
        signal.alarm(0)


def run(model_alias: str, port: int):
    import httpx
    import reasoning_gym
    from reasoning_gym.utils import extract_answer
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).parent))
    from common import MODELS, build_prompt

    model_id = MODELS[model_alias]
    tok = AutoTokenizer.from_pretrained(model_id)
    cat = [r for r in json.loads((OUT / "catalogue.json").read_text()) if "error" not in r]
    jobs = []
    for r in cat:
        ds = reasoning_gym.create_dataset(r["name"], size=N, seed=SEED)
        for i in range(N):
            e = ds[i]
            jobs.append((r["name"], i, ds, e, build_prompt(tok, e["question"])[1]))
    print(f"{len(jobs)} prompts built", flush=True)

    async def go():
        sem = asyncio.Semaphore(512)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=3600, limits=httpx.Limits(max_connections=None)) as client:
            async def one(job):
                body = {"model": model_id, "prompt": job[4], "max_tokens": 1024, "temperature": 1.0, "top_p": 1.0,
                        "seed": job[1], "return_token_ids": True}
                async with sem:
                    resp = await client.post("/v1/completions", json=body)
                return resp.json() if resp.status_code == 200 else {"error": resp.text[:300]}
            return await asyncio.gather(*(one(j) for j in jobs))

    t0 = time.perf_counter()
    results = asyncio.run(go())
    print(f"generation {time.perf_counter() - t0:.0f}s", flush=True)
    with open(OUT / f"{model_alias}.jsonl", "w") as f:
        for (name, i, ds, e, ids), res in zip(jobs, results):
            rec = {"model": model_id, "dataset": name, "idx": i, "prompt_tokens": len(ids), "gold": str(e["answer"])}
            if "error" in res:
                rec.update(error=res["error"], score=0.0, extracted=None, completion_tokens=0)
            else:
                ch = res["choices"][0]
                ans = extract_answer(ch["text"])
                s, err = score(ds, ans, e)
                rec.update(completion_tokens=len(ch["token_ids"]), finish_reason=ch["finish_reason"], extracted=ans,
                           score=s, completion=ch["text"], **({"error": err} if err else {}))
            f.write(json.dumps(rec) + "\n")


def table():
    import pandas as pd

    cat = {r["name"]: r for r in json.loads((OUT / "catalogue.json").read_text())}
    runs = {m: pd.read_json(OUT / f"{m}.jsonl", lines=True) for m in ("1.7b", "4b") if (OUT / f"{m}.jsonl").exists()}

    def stats(df):
        return {"acc": df.score.mean(), "one": (df.score == 1.0).mean(), "none": df.extracted.isna().mean(),
                "len": f"{df.completion_tokens.mean():.0f}/{np.quantile(df.completion_tokens, 0.95):.0f}",
                "frac": df.score[(df.score > 0) & (df.score < 1)].tolist(), "plen": df.prompt_tokens.max(),
                "err": int(df.get("error", pd.Series(dtype=str)).notna().sum())}

    rows, examples = [], []
    notes = {k: [] for k in ("failed", "slow", "long", "junk", "none", "gold", "raises", "errors")}
    for name, c in cat.items():
        if "error" in c:
            notes["failed"].append(f"`{name}` ({c['error']})")
            continue
        per = {m: stats(df[df.dataset == name]) for m, df in runs.items()}
        frac = [x for p in per.values() for x in p["frac"]] + [float(s) for s in c["probe_scores"] if s != "error"]
        frac = [x for x in frac if 0 < x < 1]
        scoring = ("exact" if not frac else "exact+ε" if max(frac) <= 0.05 else "partial") + ("*" if c["default_scorer"] else "")
        row = {"dataset": name, "category": c["category"], "scoring": scoring, "difficulty knobs (default)": c["knobs"]}
        for m, p in per.items():
            row.update({f"{m} acc": f"{p['acc']:.2f}", f"{m} =1": f"{p['one']:.2f}", f"{m} no-ans": f"{p['none']:.2f}",
                        f"{m} tok mean/p95": p["len"]})
        rows.append(row)
        examples.append({"dataset": name, "example question": " ".join(c["example"].split())[:150]})
        plen = max((p["plen"] for p in per.values()), default=0)
        if plen > 1500:
            notes["long"].append(f"`{name}` ({plen})")
        if c["gen_s"] > 10:
            notes["slow"].append(f"`{name}` ({c['gen_s']:.0f}s)")
        if c["junk_score_max"] >= 0.5:
            notes["junk"].append(f"`{name}` ({c['junk_score_max']:.2f})")
        if c["none_score_max"] > 0:
            notes["none"].append(f"`{name}` ({c['none_score_max']:.2f})")
        if c["gold_score_mean"] < 1.0:
            notes["gold"].append(f"`{name}` ({c['gold_score_mean']:.2f})")
        if "error" in c["probe_scores"]:
            notes["raises"].append(f"`{name}`")
        errs = sum(p["err"] for p in per.values())
        if errs:
            notes["errors"].append(f"`{name}` ({errs})")
    df = pd.DataFrame(rows).sort_values(["category", "dataset"])
    ex = pd.DataFrame(examples).set_index("dataset").loc[df.dataset].reset_index()
    def to_md(d):  # unpadded pipe table; cell text must not contain "|"
        cell = lambda v: str(v).replace("|", "/")
        lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
        return "\n".join(lines + ["| " + " | ".join(map(cell, r)) + " |" for r in d.itertuples(index=False)])
    labels = {
        "failed": "Fail to generate at the default config",
        "slow": f"Slow to generate (more than 10 s for {N} instances, measured with 24 datasets generating in parallel)",
        "long": "Prompts longer than 1500 tokens (max prompt tokens)",
        "junk": "Junk answer `zzz` scores at least 0.5 (scorer unusable as a reward without a fix)",
        "none": "A missing answer still scores above 0 (score)",
        "gold": "The stored gold answer scores below 1 (mean; the scorer checks validity and `answer` is not a valid solution string)",
        "raises": "`score_answer` raised an exception on a perturbed gold answer (guard reward calls)",
        "errors": "Request or scoring errors during calibration (count; see the jsonl `error` field)",
    }
    note_md = "\n".join(f"- {labels[k]}: {', '.join(v)}." for k, v in notes.items() if v) or "- None."
    overall = []
    for m, d in runs.items():
        done = d[d.finish_reason == "stop"]
        overall.append(f"- {m}: mean score {d.score.mean():.2f}, exactly 1.0 {(d.score == 1).mean():.2f}, no answer "
                       f"{d.extracted.isna().mean():.2f}, truncated at 1024 tokens {(d.finish_reason == 'length').mean():.2f}; "
                       f"among completed samples: mean score {done.score.mean():.2f}, no answer {done.extracted.isna().mean():.2f}.")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(TEMPLATE.format(n=N, seed=SEED, table=to_md(df), examples=to_md(ex), notes=note_md,
                                   overall="\n".join(overall), ndatasets=len(rows)))


TEMPLATE = """# reasoning-gym calibration

Base-model calibration of every reasoning-gym dataset at its default config: {n} instances per dataset (seed {seed}), one sample each at temperature 1.0 (top_p 1.0), max_tokens 1024, no thinking phase, reasoning-gym's `default` system prompt, answer taken from the last `<answer>…</answer>` and scored with the dataset's own `score_answer`. Models: Qwen3-1.7B (`enable_thinking=False`) and Qwen3-4B-Instruct-2507. Raw per-instance records are on the compute host under `results/calibration/` (`<model>.jsonl`, `catalogue.json`); regenerate this file with `scripts/dev/rg_calibrate.py`.

Columns: `acc` is mean score; `=1` the fraction scoring exactly 1.0; `no-ans` the fraction without an extractable answer; `tok mean/p95` completion tokens (1024 means truncated). Scoring is classified from the scores of model outputs and of perturbed gold answers: `exact` gives only 0 or 1, `exact+ε` also gives a small format credit (at most 0.05) to wrong but parseable answers, `partial` gives larger fractional credit; `*` marks datasets that use the base-class scorer, which gives `len(gold)/len(answer)` credit whenever the gold answer is a substring of the answer. Difficulty knobs are the dataset's curriculum attributes with their default-config values (`lo–hi` for ranges).

Over all {n} × {ndatasets} prompts:

{overall}

Almost every missing answer is a truncated sample: Qwen3-4B-Instruct-2507 writes long step-by-step solutions, so at max_tokens 1024 half of its samples never reach `<answer>`; its per-dataset accuracy here is a lower bound, and its RL runs need a larger token budget or a prompt that asks for brevity.

{table}

## Example questions

{examples}

## Notes

{notes}
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["catalogue", "probe", "run", "table"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--model", default="1.7b")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    if a.cmd == "probe":
        print(MARK + json.dumps(probe(a.name)))
    elif a.cmd == "catalogue":
        catalogue()
    elif a.cmd == "run":
        run(a.model, a.port)
    else:
        table()
