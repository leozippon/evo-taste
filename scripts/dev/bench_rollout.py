"""Rollout throughput of a running vLLM server: base only, one LoRA, and four LoRAs sharing the batch.

128 reasoning-gym prompts of 150-300 tokens, n=8 samples each (1024 sequences in flight), temperature 1.0,
max_tokens 512. Each configuration runs twice: "natural" (sequences stop at EOS; wall time is set by the few longest
samples) and "full" (ignore_eos, every sequence decodes 512 tokens; steady-state decode throughput).
Usage: python scripts/dev/bench_rollout.py --model 1.7b --port 8100 [--tag name]
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import reasoning_gym
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODELS, build_prompt, load_adapter, make_random_lora, served_model, unload_adapter

DATASETS = ["chain_sum", "basic_arithmetic", "gsm_symbolic", "letter_counting", "number_sorting", "word_sorting",
            "propositional_logic", "family_relationships", "countdown", "spell_backward", "simple_equations",
            "knights_knaves", "maze", "base_conversion", "time_intervals", "syllogism"]


def bench_prompts(tok, n: int, lo: int, hi: int) -> list[list[int]]:
    out, seed = [], 0
    while len(out) < n:
        for name in DATASETS:
            ids = build_prompt(tok, reasoning_gym.create_dataset(name, size=1, seed=seed)[0]["question"])[1]
            if lo <= len(ids) <= hi and len(out) < n:
                out.append(ids)
        seed += 1
    return out


async def rollout(base_url, prompts, models, n, max_tokens, ignore_eos):
    async with httpx.AsyncClient(base_url=base_url, timeout=3600, limits=httpx.Limits(max_connections=None)) as client:
        async def one(i, ids):
            body = {"model": models[i % len(models)], "prompt": ids, "n": n, "temperature": 1.0, "top_p": 1.0,
                    "max_tokens": max_tokens, "ignore_eos": ignore_eos}
            r = await client.post("/v1/completions", json=body)
            r.raise_for_status()
            return r.json()

        t0 = time.perf_counter()
        res = await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
        wall = time.perf_counter() - t0
    out_tok = sum(r["usage"]["completion_tokens"] for r in res)
    truncated = sum(c["finish_reason"] == "length" for r in res for c in r["choices"])
    return {"wall_s": round(wall, 1), "output_tokens": out_tok, "output_tok_per_s": round(out_tok / wall),
            "mean_completion_tokens": round(out_tok / (len(prompts) * n), 1),
            "truncated_frac": round(truncated / (len(prompts) * n), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="1.7b")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default="runs/dev/bench")
    ap.add_argument("--tag", default="default", help="label of the server configuration under test")
    args = ap.parse_args()
    base_url = f"http://127.0.0.1:{args.port}"
    model_id = MODELS[args.model]
    assert served_model(base_url) == model_id
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    prompts = bench_prompts(tok, args.num_prompts, 150, 300)
    lens = [len(p) for p in prompts]
    cpu_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    names = [f"bench{i}" for i in range(4)]
    for i, name in enumerate(names):
        path = out / name
        if not path.exists():
            make_random_lora(cpu_model, path, seed=100 + i)
        load_adapter(base_url, name, path)
    del cpu_model

    run = lambda models, full: asyncio.run(rollout(base_url, prompts, models, args.n, args.max_tokens, full))
    run([model_id] + names, False)  # warm-up: CUDA graphs, adapter slots
    report = {"model": model_id, "tag": args.tag, "prompts": len(prompts), "prompt_tokens_mean": sum(lens) / len(lens),
              "n": args.n, "max_tokens": args.max_tokens}
    for mode, full in [("natural", False), ("full", True)]:
        for label, models in [("base", [model_id]), ("lora_x1", names[:1]), ("lora_x4", names)]:
            report[f"{mode}_{label}"] = run(models, full)
            print(mode, label, report[f"{mode}_{label}"], flush=True)
    for name in names:
        unload_adapter(base_url, name)
    (out / f"rollout_{args.tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
