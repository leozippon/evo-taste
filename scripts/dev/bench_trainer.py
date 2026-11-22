"""Trainer-side cost of one policy-gradient update with HF + PEFT (bf16) next to a resident vLLM server.

One update = forward + backward over --num-seqs sequences of prompt+completion tokens (LoRA parameters only,
gradient checkpointing, micro-batched), then a single AdamW step; then adapter save + load into vLLM.
Usage: python scripts/dev/bench_trainer.py --model 1.7b --port 8100 --micro-batches 4,8,16
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from common import LORA_TARGETS, MODELS, load_adapter, served_model, unload_adapter


def gpu_used_mib() -> int:
    """Total memory used on the (single) visible GPU by all processes, as nvidia-smi reports it."""
    idx = os.environ["CUDA_VISIBLE_DEVICES"]
    q = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True)
    return int(q.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="1.7b")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--num-seqs", type=int, default=512)
    ap.add_argument("--prompt-len", type=int, default=250)
    ap.add_argument("--comp-len", type=int, default=350)
    ap.add_argument("--micro-batches", default="4,8,16")
    ap.add_argument("--no-ckpt", action="store_true", help="disable gradient checkpointing")
    ap.add_argument("--out", default="runs/dev/bench")
    ap.add_argument("--handoff-dir", help="where adapter versions are saved for vLLM (default: --out); try /dev/shm")
    args = ap.parse_args()
    base_url = f"http://127.0.0.1:{args.port}"
    model_id = MODELS[args.model]
    assert served_model(base_url) == model_id
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    if not args.no_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, target_modules=LORA_TARGETS, lora_dropout=0.0))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-6)
    model.train()

    g = torch.Generator().manual_seed(0)
    L, C = args.prompt_len + args.comp_len, args.comp_len
    data = torch.randint(1000, 150000, (args.num_seqs, L), generator=g)
    adv = torch.randn(args.num_seqs, generator=g)
    report = {"model": model_id, "seqs": args.num_seqs, "seq_len": L, "comp_len": C,
              "grad_ckpt": not args.no_ckpt, "lora_params_M": sum(p.numel() for p in params) / 1e6,
              "gpu_used_mib_before": gpu_used_mib(), "runs": []}

    def seq_logp(x):
        logits = model(input_ids=x, logits_to_keep=C + 1).logits[:, :-1].float()
        tgt = x[:, -C:]
        return logits.gather(-1, tgt[..., None])[..., 0] - logits.logsumexp(-1)

    for mb in [int(x) for x in args.micro_batches.split(",") if x]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rec = {"micro_batch": mb}
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for i in range(0, args.num_seqs, mb):
                    seq_logp(data[i:i + mb].cuda())
            torch.cuda.synchronize()
            rec["score_only_s"] = round(time.perf_counter() - t0, 1)
            t0 = time.perf_counter()
            for i in range(0, args.num_seqs, mb):
                x = data[i:i + mb].cuda()
                loss = -(adv[i:i + mb].cuda()[:, None] * seq_logp(x)).sum() / (args.num_seqs * C)
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            rec["update_s"] = round(time.perf_counter() - t0, 1)
            rec["tokens_per_s"] = round(args.num_seqs * L / rec["update_s"])
            rec["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            rec["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
            rec["gpu_used_mib_after"] = gpu_used_mib()
        except torch.OutOfMemoryError as e:
            rec["oom"] = str(e).splitlines()[0][:200]
            opt.zero_grad(set_to_none=True)
        print(rec, flush=True)
        report["runs"].append(rec)
        if "oom" in rec:
            break

    # Adapter hand-off: save a new version to a fresh directory and load it under a fresh name.
    handoff = []
    handoff_dir = Path(args.handoff_dir) / args.model if args.handoff_dir else out
    for v in range(3):
        path = handoff_dir / f"trainer_v{v}"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.save_pretrained(path)
        save_s = time.perf_counter() - t0
        load_s = load_adapter(base_url, f"trainer_v{v}", path)
        t0 = time.perf_counter()
        r = httpx.post(f"{base_url}/v1/completions", timeout=60,
                       json={"model": f"trainer_v{v}", "prompt": [1, 2, 3], "max_tokens": 1})
        r.raise_for_status()
        first_req_s = time.perf_counter() - t0
        handoff.append({"save_s": round(save_s, 3), "load_s": round(load_s, 3), "first_request_s": round(first_req_s, 3)})
        unload_adapter(base_url, f"trainer_v{v}")
    report["handoff"] = {"dir": str(handoff_dir), "versions": handoff}
    print(handoff)
    suffix = ("_nockpt" if args.no_ckpt else "") + ("_handoff-" + Path(args.handoff_dir).name if args.handoff_dir else "")
    (out / f"trainer{suffix}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
