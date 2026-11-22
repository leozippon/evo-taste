"""Multi-LoRA smoke test against a running vLLM server (scripts/dev/serve_vllm.sh) on the same GPU.

(a) base / adapter A / adapter B produce different outputs; (b) hot-reload procedures: are the new weights served,
is KV from the old weights reused through the prefix cache, latency; (c) per-token logprobs of vLLM samples
recomputed with HF transformers + PEFT under the same adapter.
Greedy text equality is not a usable weight check: bf16 near-ties flip between cold and prefix-cached prefills.
Weights are therefore compared through teacher-forced prompt logprobs (which never read the prefix cache).
Usage: python scripts/dev/lora_smoke.py --model 1.7b --port 8100
"""

import argparse
import json
import shutil
from pathlib import Path

import httpx
import numpy as np
import reasoning_gym
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODELS, build_prompt, load_adapter, make_random_lora, served_model, unload_adapter


def questions(names: list[str], per: int, seed: int) -> list[str]:
    return [reasoning_gym.create_dataset(n, size=per, seed=seed)[i]["question"] for n in names for i in range(per)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="1.7b")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--out", default="runs/dev/lora_smoke")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    base_url = f"http://127.0.0.1:{args.port}"
    model_id = MODELS[args.model]
    assert served_model(base_url) == model_id, "server serves a different model"
    out = Path(args.out) / args.model
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    client = httpx.Client(base_url=base_url, timeout=600)
    report = {"model": model_id}

    tok = AutoTokenizer.from_pretrained(model_id)
    hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    pA = make_random_lora(hf, out / "A", seed=1)  # old weights
    pB = make_random_lora(hf, out / "B", seed=2)
    pW = make_random_lora(hf, out / "W", seed=3)  # new weights

    def copy(src: Path, name: str) -> Path:
        shutil.copytree(src, out / name)
        return out / name

    def complete(name, prompt, **kw):
        r = client.post("/v1/completions", json={"model": name, "prompt": prompt, "return_token_ids": True, **kw})
        if r.status_code != 200:
            raise RuntimeError(f"completion via {name} failed: {r.status_code} {r.text}")
        return r.json()

    probe = [build_prompt(tok, q)[1] for q in questions(["gsm_symbolic", "knights_knaves", "word_sorting", "maze"], 1, 100)]
    # Fixed continuation scored teacher-forced: the base model's greedy answer to each probe prompt.
    probe_full = [p + complete(model_id, p, max_tokens=64, temperature=0.0)["choices"][0]["token_ids"] for p in probe]

    def plp(name):
        """Prompt logprobs of the fixed continuations (computed from scratch: prompt_logprobs bypass the prefix cache)."""
        vals = []
        for p, full in zip(probe, probe_full):
            pl = complete(name, full, max_tokens=1, temperature=0.0, prompt_logprobs=0)["choices"][0]["prompt_logprobs"]
            vals += [pl[i][str(full[i])]["logprob"] for i in range(len(p), len(full))]
        return np.array(vals)

    dist = lambda x, y: float(np.abs(x - y).mean())

    def greedy(name):
        return [complete(name, p, max_tokens=48, temperature=0.0)["choices"][0]["token_ids"] for p in probe]

    # (a) base vs adapters; A2 = A's weights at another path gives the same-weights noise floor.
    load_adapter(base_url, "A", pA)
    load_adapter(base_url, "A2", copy(pA, "A2"))
    load_adapter(base_url, "B", pB)
    lp = {k: plp(k) for k in (model_id, "A", "A2", "B")}
    g = {k: greedy(k) for k in (model_id, "A", "B")}
    report["a_adapters"] = {
        "teacher_forced_mean_abs_logprob_diff": {
            "A_vs_A2_same_weights": dist(lp["A"], lp["A2"]), "A_vs_base": dist(lp["A"], lp[model_id]),
            "B_vs_base": dist(lp["B"], lp[model_id]), "A_vs_B": dist(lp["A"], lp["B"])},
        "greedy_outputs_differ_of_4": {"A_vs_base": sum(x != y for x, y in zip(g["A"], g[model_id])),
                                       "B_vs_base": sum(x != y for x, y in zip(g["B"], g[model_id])),
                                       "A_vs_B": sum(x != y for x, y in zip(g["A"], g["B"]))},
        "sample_greedy_text": {k: tok.decode(g[k][0])[:120] for k in g},
    }
    print(json.dumps(report["a_adapters"], indent=1), flush=True)

    # (b) hot reload. Each procedure starts from a name serving A's weights whose probe prompt sits in the prefix
    # cache, switches to W, then checks (i) which weights are served and (ii) whether the first request after the
    # switch reads prefix-cache blocks (which can only have been computed with the old weights).
    load_adapter(base_url, "ref", copy(pW, "ref"))
    lp_new, lp_old = plp("ref"), lp["A"]

    procs = {}

    def run_proc(label, name, switch):
        load_adapter(base_url, name, copy(pA, f"{name}_v1"))
        complete(name, probe[0], max_tokens=1, temperature=0.0)  # puts the probe prompt in the prefix cache
        served, latency = switch(name)
        usage = complete(served, probe[0], max_tokens=1, temperature=0.0)["usage"]
        hits = usage["prompt_tokens_details"]["cached_tokens"]  # needs --enable-prompt-tokens-details
        x = plp(served)
        procs[label] = {"latency_s": round(latency, 3), "dist_to_new": round(dist(x, lp_new), 4),
                        "dist_to_old": round(dist(x, lp_old), 4), "serves_new_weights": dist(x, lp_new) < 0.25 * dist(x, lp_old),
                        "stale_prefix_cache_tokens": int(hits)}
        print(label, procs[label], flush=True)

    def new_name(name):
        return f"{name}@v2", load_adapter(base_url, f"{name}@v2", copy(pW, f"{name}_v2"))

    def inplace_new_path(name):
        return name, load_adapter(base_url, name, copy(pW, f"{name}_v2"), inplace=True)

    def unload_load_new_path(name):
        t = unload_adapter(base_url, name)
        return name, t + load_adapter(base_url, name, copy(pW, f"{name}_v2"))

    def overwrite(name):
        dst = out / f"{name}_v1"
        for f in pW.iterdir():
            shutil.copy(f, dst / f.name)
        return dst

    def inplace_same_path(name):
        return name, load_adapter(base_url, name, overwrite(name), inplace=True)

    def unload_load_same_path(name):
        dst = overwrite(name)
        t = unload_adapter(base_url, name)
        return name, t + load_adapter(base_url, name, dst)

    def overwrite_only(name):
        overwrite(name)
        return name, 0.0

    run_proc("new_name_new_path", "p1", new_name)
    run_proc("inplace_new_path", "p2", inplace_new_path)
    run_proc("unload_load_new_path", "p3", unload_load_new_path)
    run_proc("inplace_same_path_overwritten", "p4", inplace_same_path)
    run_proc("unload_load_same_path_overwritten", "p5", unload_load_same_path)
    run_proc("overwrite_files_no_reload", "p6", overwrite_only)
    lat = [load_adapter(base_url, f"lat{i}", copy(pW, f"lat{i}")) for i in range(5)]
    unl = [unload_adapter(base_url, f"lat{i}") for i in range(5)]
    report["b_hot_reload"] = {"procedures": procs, "fresh_load_latency_s": [round(x, 3) for x in lat],
                              "unload_latency_s": [round(x, 3) for x in unl]}

    # (c) likelihood consistency through adapter B: 16 prompts x 4 samples, temperature 1, full softmax.
    names = ["gsm_symbolic", "knights_knaves", "family_relationships", "propositional_logic", "countdown", "maze",
             "syllogism", "time_intervals"]
    prompts = [build_prompt(tok, q) for q in questions(names, 2, 200)]
    samples = []
    for text, ids in prompts:
        # The string prompt must tokenize to exactly the ids the trainer builds.
        echo = complete("B", text, max_tokens=1, temperature=0.0)["choices"][0]["prompt_token_ids"]
        assert echo == ids, "vLLM tokenized the prompt string differently"
        res = complete("B", ids, n=4, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, logprobs=0,
                       return_tokens_as_token_ids=True)
        for ch in res["choices"]:
            comp = ch["token_ids"]
            assert [int(t.split(":")[1]) for t in ch["logprobs"]["tokens"]] == comp
            samples.append({"prompt": ids, "comp": comp, "vllm": np.array(ch["logprobs"]["token_logprobs"]),
                            "finish": ch["finish_reason"]})

    peft_bf16 = PeftModel.from_pretrained(hf, pB).eval()

    @torch.no_grad()
    def score(model, s):
        x = torch.tensor([s["prompt"] + s["comp"]], device="cuda")
        logits = model(input_ids=x, logits_to_keep=len(s["comp"]) + 1).logits[0, :-1].float()
        return logits.log_softmax(-1).gather(-1, torch.tensor(s["comp"], device="cuda")[:, None])[:, 0].cpu().numpy()

    for s in samples:
        s["hf_bf16"] = score(peft_bf16, s)
        with peft_bf16.disable_adapter():
            s["hf_base"] = score(peft_bf16, s)
    del peft_bf16, hf
    torch.cuda.empty_cache()
    hf32 = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map="cuda").eval()
    peft_fp32 = PeftModel.from_pretrained(hf32, pB).eval()
    for s in samples:
        s["hf_fp32"] = score(peft_fp32, s)

    def compare(key_a, key_b):
        tok_d = np.concatenate([s[key_a] - s[key_b] for s in samples])
        seq_d = np.array([np.sum(s[key_a]) - np.sum(s[key_b]) for s in samples])
        r = lambda v: round(float(v), 4)
        return {"tok_mean_abs": r(np.abs(tok_d).mean()), "tok_mean_signed": r(tok_d.mean()),
                "tok_p99_abs": r(np.quantile(np.abs(tok_d), 0.99)), "tok_max_abs": r(np.abs(tok_d).max()),
                "seq_mean_abs": r(np.abs(seq_d).mean()), "seq_mean_signed": r(seq_d.mean()),
                "seq_p95_abs": r(np.quantile(np.abs(seq_d), 0.95)), "seq_max_abs": r(np.abs(seq_d).max())}

    lens = [len(s["comp"]) for s in samples]
    report["c_consistency"] = {
        "sequences": len(samples), "tokens": sum(lens), "mean_len": float(np.mean(lens)), "max_len": max(lens),
        "finish_reasons": {r: sum(s["finish"] == r for s in samples) for r in {s["finish"] for s in samples}},
        "eos_in_token_ids": sum(s["comp"][-1] == tok.eos_token_id for s in samples),
        "vllm_vs_hf_bf16": compare("vllm", "hf_bf16"),
        "vllm_vs_hf_fp32": compare("vllm", "hf_fp32"),
        "hf_bf16_vs_hf_fp32": compare("hf_bf16", "hf_fp32"),
        "control_vllm_adapter_vs_hf_base": compare("vllm", "hf_base"),
    }
    print(json.dumps(report["c_consistency"], indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    for name in ["A", "A2", "B", "ref", "p1", "p1@v2", "p2", "p3", "p4", "p5", "p6"]:
        unload_adapter(base_url, name)


if __name__ == "__main__":
    main()
