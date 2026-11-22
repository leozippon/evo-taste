"""Seconds per update on one realistic amplification batch, for the textbook path and the engine's paths.

The batch is the base model's rollouts on the training stream's first step (prompts x rollouts, temperature 1,
max_tokens 640), generated once through a running vLLM server and cached; every sequence gets a non-zero weight so
all of them enter the update. Paths:
  naive   full-vocabulary fp32 logits for whole padded sequences, fixed micro-batch of 4 (8 runs out of memory), no length sorting
  plain   engine, completion-only LM head, length-sorted token-budget micro-batches, one row per sequence
  shared  engine as plain, plus shared-prefix rows (each prompt once for its rollouts)

Usage: CUDA_VISIBLE_DEVICES=<gpu> python scripts/bench_update.py --attach http://127.0.0.1:<port> [--paths naive,plain,shared]
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from evotaste.config import Lora, Trainer
from evotaste.records import read_jsonl_gz, write_jsonl_gz
from evotaste.sampler import Request, Sampler
from evotaste.tasks import Streams, Suites, prompt_ids
from evotaste.trainer import Policy, Sequence

OUT = Path("runs/bench")
MODEL = "Qwen/Qwen3-1.7B"


def batch(url: str, prompts: int, rollouts: int) -> list[Sequence]:
    cache = OUT / f"batch_{prompts}x{rollouts}.jsonl.gz"
    if not cache.exists():
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)
        probes = SimpleNamespace(train_per_family=8, selection_per_family=0, foundation_per_source=0)
        tasks = Streams(Suites("configs/suites.yaml"), 1, probes).train_batch(0, prompts)
        ids = [prompt_ids(tok, t.question) for t in tasks]
        samples = Sampler(url, MODEL, "bench").generate([Request(MODEL, p, rollouts, 640) for p in ids])
        write_jsonl_gz(cache, [{"prompt": p, "completion": s.token_ids} for p, g in zip(ids, samples) for s in g])
    rng = random.Random(0)
    return [Sequence(r["prompt"], r["completion"], rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5))
            for r in read_jsonl_gz(cache)]


def naive_update(policy: Policy, member, seqs, lr, micro_batch=4):
    policy._activate(member)
    params = list(policy.params.values())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    tokens = sum(len(s.completion) for s in seqs)
    for i in range(0, len(seqs), micro_batch):
        mb = seqs[i:i + micro_batch]
        L = max(len(s.prompt) + len(s.completion) for s in mb)
        ids = torch.zeros(len(mb), L, dtype=torch.long)
        for b, s in enumerate(mb):
            ids[b, :len(s.prompt) + len(s.completion)] = torch.tensor(s.prompt + s.completion)
        logp = policy.model(input_ids=ids.cuda()).logits.float().log_softmax(-1)
        loss = 0.0
        for b, s in enumerate(mb):
            n, c = len(s.prompt), torch.tensor(s.completion, device="cuda")
            loss = loss - s.weight * logp[b, n - 1:n - 1 + len(c)].gather(1, c[:, None]).sum()
        (loss / tokens).backward()
        del logp
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attach", required=True)
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--paths", default="naive,plain,shared")
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    seqs = batch(args.attach, args.prompts, args.rollouts)
    comp = [len(s.completion) for s in seqs]
    report = {"sequences": len(seqs), "prompt_tokens_mean": sum(len(s.prompt) for s in seqs) / len(seqs),
              "completion_tokens_mean": sum(comp) / len(comp), "completion_tokens_max": max(comp),
              "truncated_at_640": sum(c == 640 for c in comp), "token_budget": args.token_budget, "paths": {}}
    print(json.dumps(report), flush=True)
    for path in args.paths.split(","):
        policy = Policy(MODEL, Lora(), Trainer(token_budget=args.token_budget, shared_prefix=path == "shared"))
        member = policy.new_member(seed=0, b_std=0.01)
        times = []
        for _ in range(args.repeats):
            m = copy.deepcopy(member)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            if path == "naive":
                naive_update(policy, m, seqs, 1e-5)
            else:
                policy.update(m, seqs, 1e-5)
            torch.cuda.synchronize()
            times.append(round(time.perf_counter() - t0, 2))
        report["paths"][path] = {"update_s": times, "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        print(path, report["paths"][path], flush=True)
        del policy
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"update_{args.prompts}x{args.rollouts}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
