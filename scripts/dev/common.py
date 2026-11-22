"""Shared helpers for the development scripts: prompt recipe, random LoRA adapters, vLLM adapter endpoints."""

import time
from pathlib import Path

import httpx
import torch
from reasoning_gym.utils import SYSTEM_PROMPTS

MODELS = {"1.7b": "Qwen/Qwen3-1.7B", "4b": "Qwen/Qwen3-4B-Instruct-2507"}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SYSTEM_PROMPT = SYSTEM_PROMPTS["default"]


def build_prompt(tokenizer, question: str) -> tuple[str, list[int]]:
    """Return the templated prompt string and its token ids (no thinking phase, no extra special tokens)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return text, ids


def make_random_lora(base_model, out_dir: Path, seed: int, r: int = 32, alpha: int = 64, b_std: float = 0.01):
    """Save a PEFT LoRA adapter whose B matrices are small random values (so the adapter is not a no-op).

    alpha != r on purpose: a LoRA-scaling mismatch between vLLM and PEFT would then be visible.
    The adapter is created on a temporary PEFT wrapper and removed again, leaving base_model unchanged.
    """
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=LORA_TARGETS, lora_dropout=0.0, bias="none")
    peft_model = get_peft_model(base_model, cfg)
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_B" in name:
                p.normal_(0.0, b_std)
    peft_model.save_pretrained(out_dir)
    peft_model.unload()
    return Path(out_dir)


def load_adapter(base_url: str, name: str, path: Path, inplace: bool = False) -> float:
    """Load (or with inplace=True, replace) an adapter in a running vLLM server; return latency in seconds."""
    t0 = time.perf_counter()
    r = httpx.post(f"{base_url}/v1/load_lora_adapter", timeout=120,
                   json={"lora_name": name, "lora_path": str(Path(path).resolve()), "load_inplace": inplace})
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        raise RuntimeError(f"load_lora_adapter {name} failed: {r.status_code} {r.text}")
    return dt


def unload_adapter(base_url: str, name: str) -> float:
    t0 = time.perf_counter()
    r = httpx.post(f"{base_url}/v1/unload_lora_adapter", json={"lora_name": name}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"unload_lora_adapter {name} failed: {r.status_code} {r.text}")
    return time.perf_counter() - t0


def served_model(base_url: str) -> str:
    """Base model name served by vLLM (the first entry of /v1/models without a parent)."""
    data = httpx.get(f"{base_url}/v1/models", timeout=10).json()["data"]
    return next(m["id"] for m in data if not m.get("parent"))
