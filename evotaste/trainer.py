"""PyTorch/PEFT policy: a frozen bf16 base model with one LoRA structure whose tensors are swapped per member.

Scoring, the group update and the distillation update share one forward path. Sequences are packed into rows: by
default a row holds a prompt once followed by as many of its completions as fit in the token budget, with an attention
mask that lets each completion see the prompt and itself only (shared-prefix reuse); with `shared_prefix=False` every
(prompt, completion) pair is its own row and plain causal attention is used. Output projections are applied only at
the positions that predict completion tokens, in chunks that are recomputed in the backward pass, so full-vocabulary
fp32 logits never exist for whole sequences. On GPU the LoRA matmuls run in bf16 under autocast, as in vLLM; the fp32
LoRA tensors are the optimizer's master copy. The frozen base model (the distillation teacher and the context score's
reference) is the same network with the adapter disabled.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from peft.helpers import disable_input_dtype_casting
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
HEAD_CHUNK = 1024  # completion positions per chunk of sampled-token log-probabilities
# Completion positions per chunk of the reverse KL: each chunk holds a few fp32 [chunk, vocabulary] tensors (about
# 155 MB each for Qwen3's 151,936 tokens), recomputed in the backward pass, whatever the completions' length.
KL_CHUNK = 256
CENTERED_SCALE = 0.5  # `centered` advantages: (return - group mean) / CENTERED_SCALE


@dataclass
class Sequence:
    prompt: list[int]
    completion: list[int]  # sampled token ids, including the end-of-turn token when generation stopped
    weight: float = 1.0  # advantage
    imitate: bool = False  # in the self-imitation term (an amplification rollout with reward 1)


@dataclass
class MemberWeights:
    """A member's trainable state, kept on the CPU between turns: fp32 LoRA tensors and AdamW state."""
    lora: dict[str, torch.Tensor]  # PEFT parameter name -> tensor
    optimizer: dict | None = None  # torch AdamW state_dict; None before the first update of a phase


def length_returns(rewards: list[float], tokens: list[int], cap: int, free: int, slope: float) -> list[float]:
    """reward - slope * clip((tokens - free) / (cap - free), 0, 1): nothing up to `free` completion tokens, then a
    deduction growing linearly to `slope` at the cap."""
    return [r - slope * min(max((n - free) / (cap - free), 0.0), 1.0) for r, n in zip(rewards, tokens)]


def advantages(returns: list[float], rule: str) -> list[float]:
    """Group-relative weights of one prompt's rollouts; exactly 0 for every rollout when all returns are equal.

    centered: (return - mean) / 0.5, so a small return difference (a length deduction) stays a small weight;
    standardized: (return - mean) / (population std + 1e-6), Round 1's rule, which scales any difference to unit size.
    """
    if max(returns) == min(returns):
        return [0.0] * len(returns)
    mean = sum(returns) / len(returns)
    if rule == "centered":
        return [(r - mean) / CENTERED_SCALE for r in returns]
    if rule == "standardized":
        std = math.sqrt(sum((r - mean) ** 2 for r in returns) / len(returns))
        return [(r - mean) / (std + 1e-6) for r in returns]
    raise ValueError(f"unknown advantage rule {rule!r}")


@dataclass
class _Row:
    ids: list[int]
    pos: list[int]
    seg: list[int]  # 0 = prompt, k + 1 = k-th completion of the row
    src: list[int]  # row positions whose hidden state predicts each target
    targets: list[int]
    spans: list[tuple[int, int, int]]  # (sequence index, start, end) into targets


def build_rows(seqs: list[Sequence], shared_prefix: bool, token_budget: int) -> list[_Row]:
    """Rows of the forward pass. With shared_prefix, completions of the same prompt follow one copy of it while the
    row stays within token_budget; a completion that does not fit starts a new row (a row with a single completion may
    exceed the budget). Packing long completions would only add masked attention work and memory."""
    groups: dict = {}
    for i, s in enumerate(seqs):
        if not s.completion:
            raise ValueError(f"sequence {i} has an empty completion")
        groups.setdefault(tuple(s.prompt) if shared_prefix else i, []).append(i)
    rows: list[_Row] = []
    for members in groups.values():
        prompt = seqs[members[0]].prompt
        n = len(prompt)
        row = None
        for i in members:
            c = seqs[i].completion
            if row is None or len(row.ids) + len(c) > token_budget:
                row = _Row(list(prompt), list(range(n)), [0] * n, [], [], [])
                rows.append(row)
            start = len(row.ids)
            row.spans.append((i, len(row.targets), len(row.targets) + len(c)))
            row.src += [n - 1] + list(range(start, start + len(c) - 1))
            row.targets += c
            row.ids += c
            row.pos += range(n, n + len(c))
            row.seg += [len(row.spans)] * len(c)
    return rows


def micro_batches(rows: list[_Row], token_budget: int) -> list[list[_Row]]:
    """Sort by length and fill micro-batches while (rows x longest row) stays within the token budget."""
    batches, current = [], []
    for row in sorted(rows, key=lambda r: len(r.ids), reverse=True):
        if current and (len(current) + 1) * len(current[0].ids) > token_budget:
            batches.append(current)
            current = []
        current.append(row)
    return batches + [current] if current else batches


def by_sequence(rows: list[_Row], values: torch.Tensor) -> dict[int, torch.Tensor]:
    """Per sequence index, its slice of `values` (one entry per target, rows concatenated in order)."""
    out, offset = {}, 0
    for row in rows:
        for i, a, b in row.spans:
            out[i] = values[offset + a:offset + b]
        offset += len(row.targets)
    return out


def _chunk_logprobs(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = (hidden @ weight.T).float()
    return logits.gather(1, targets[:, None])[:, 0] - logits.logsumexp(1)


def head_logprobs(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log softmax(hidden @ weight.T)[targets] in chunks; with grad enabled each chunk is recomputed in backward."""
    out = []
    for s in range(0, len(targets), HEAD_CHUNK):
        args = (hidden[s:s + HEAD_CHUNK], weight, targets[s:s + HEAD_CHUNK])
        out.append(checkpoint(_chunk_logprobs, *args, use_reentrant=False) if torch.is_grad_enabled()
                   else _chunk_logprobs(*args))
    return torch.cat(out)


def _chunk_kl(student: torch.Tensor, teacher: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    s = (student @ weight.T).float().log_softmax(-1)
    t = (teacher @ weight.T).float().log_softmax(-1)
    return (s.exp() * (s - t)).sum(-1)


def reverse_kl(student: torch.Tensor, teacher: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per position, KL(softmax(student @ W.T) || softmax(teacher @ W.T)) over the full vocabulary, from hidden
    states; KL_CHUNK positions at a time, each chunk recomputed in the backward pass when grad is enabled."""
    out = []
    for s in range(0, len(student), KL_CHUNK):
        args = (student[s:s + KL_CHUNK], teacher[s:s + KL_CHUNK], weight)
        out.append(checkpoint(_chunk_kl, *args, use_reentrant=False) if torch.is_grad_enabled() else _chunk_kl(*args))
    return torch.cat(out)


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


class Policy:
    def __init__(self, model_id: str, lora, trainer, device: str = "cuda", base_model=None):
        """`base_model` (tests) replaces loading `model_id`."""
        self.cfg = trainer
        self.device = torch.device(device)
        base = base_model if base_model is not None else AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16)
        base.to(self.device)
        if trainer.grad_checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.lora_config = LoraConfig(r=lora.rank, lora_alpha=lora.alpha, target_modules=LORA_TARGETS,
                                      lora_dropout=0.0, bias="none")
        self.model = get_peft_model(base, self.lora_config)
        self.model.train()
        inner = self.model.get_base_model()
        self.decoder, self.head_weight = inner.model, inner.lm_head.weight
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        if not self.params or any(p.dtype != torch.float32 for p in self.params.values()):
            raise RuntimeError("expected fp32 LoRA parameters")

    def release_memory(self) -> None:
        """Drop gradients and cached CUDA blocks, e.g. after an out-of-memory error, before trying again."""
        for p in self.params.values():
            p.grad = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # -- member state ----------------------------------------------------------------------------------------
    def new_member(self, seed: int, b_std: float = 0.0) -> MemberWeights:
        """PEFT's initialization (A: Kaiming-uniform, B: 0, so the member starts as the base model) drawn from
        `seed`; b_std > 0 (tests, benchmarks) draws B ~ N(0, b_std) instead."""
        g = torch.Generator().manual_seed(seed)
        lora = {}
        for name in sorted(self.params):
            t = torch.empty(self.params[name].shape, dtype=torch.float32)
            if ".lora_A." in name:
                torch.nn.init.kaiming_uniform_(t, a=math.sqrt(5), generator=g)
            elif b_std > 0:
                t.normal_(0.0, b_std, generator=g)
            else:
                t.zero_()
            lora[name] = t
        return MemberWeights(lora)

    @torch.no_grad()
    def _activate(self, member: MemberWeights) -> None:
        if member.lora.keys() != self.params.keys():
            raise ValueError("member LoRA tensors do not match the policy's LoRA structure")
        for name, p in self.params.items():
            p.copy_(member.lora[name], non_blocking=True)

    def save_adapter(self, member: MemberWeights, path: str | Path, dtype=torch.float32) -> None:
        """Write a PEFT adapter directory (adapter_config.json, adapter_model.safetensors) vLLM can load."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        tensors = {n.replace(".default.", "."): t.to(dtype).contiguous() for n, t in member.lora.items()}
        save_file(tensors, path / "adapter_model.safetensors")
        self.lora_config.save_pretrained(path)

    def load_adapter(self, path: str | Path) -> dict[str, torch.Tensor]:
        """Inverse of save_adapter (fp32 checkpoints); the key set must match this policy's LoRA structure."""
        tensors = load_file(Path(path) / "adapter_model.safetensors")
        lora = {n: tensors[n.replace(".default.", ".")].float() for n in self.params}
        if len(tensors) != len(lora):
            raise ValueError(f"{path}: adapter has {len(tensors)} tensors, expected {len(lora)}")
        return lora

    # -- forward ---------------------------------------------------------------------------------------------
    def _batches(self, seqs: list[Sequence], token_budget: int | None = None) -> list[list[_Row]]:
        budget = token_budget or self.cfg.token_budget
        return micro_batches(build_rows(seqs, self.cfg.shared_prefix, budget), budget)

    def _hidden(self, rows: list[_Row]) -> tuple[torch.Tensor, torch.Tensor]:
        """Final hidden states at the positions that predict each row's targets (rows concatenated), and the
        targets; attached to the graph when grad is enabled."""
        dev, B, L = self.device, len(rows), max(len(r.ids) for r in rows)
        ids = torch.zeros(B, L, dtype=torch.long)
        pos = torch.zeros(B, L, dtype=torch.long)
        seg = torch.full((B, L), -1, dtype=torch.long)
        src, targets = [], []
        for b, r in enumerate(rows):
            n = len(r.ids)
            ids[b, :n], pos[b, :n], seg[b, :n] = torch.tensor(r.ids), torch.tensor(r.pos), torch.tensor(r.seg)
            src += [b * L + i for i in r.src]
            targets += r.targets
        mask = None
        if any(max(r.seg) > 1 for r in rows):  # some row packs several completions behind one prompt
            seg = seg.to(dev)
            i = torch.arange(L, device=dev)
            causal = i[None, :, None] >= i[None, None, :]
            visible = (seg[:, :, None] == seg[:, None, :]) | (seg == 0)[:, None, :]
            mask = (causal & visible)[:, None]  # (B, 1, L, L), True = attend
        # LoRA matmuls in bf16 like vLLM's (the fp32 LoRA weights stay the optimizer's master copy).
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"), \
                disable_input_dtype_casting(self.model):
            hidden = self.decoder(input_ids=ids.to(dev), position_ids=pos.to(dev), attention_mask=mask,
                                  use_cache=False).last_hidden_state
        return hidden.reshape(B * L, -1)[torch.tensor(src, device=dev)], torch.tensor(targets, device=dev)

    def _forward(self, rows: list[_Row]) -> list[torch.Tensor]:
        """Per row, the fp32 log-probabilities of its targets (attached to the graph when grad is enabled)."""
        hidden, targets = self._hidden(rows)
        lp = head_logprobs(hidden, self.head_weight, targets)
        return list(lp.split([len(r.targets) for r in rows]))

    def _logprobs(self, seqs: list[Sequence], token_budget: int | None = None) -> list[torch.Tensor]:
        out: list = [None] * len(seqs)
        for batch in self._batches(seqs, token_budget):
            for row, lp in zip(batch, self._forward(batch)):
                for i, a, b in row.spans:
                    out[i] = lp[a:b].cpu()
        return out

    @torch.no_grad()
    def logprobs(self, member: MemberWeights, seqs: list[Sequence]) -> list[torch.Tensor]:
        """Per-token log-probabilities (CPU, fp32) of each sequence's completion under the member's weights."""
        self._activate(member)
        return self._logprobs(seqs)

    @torch.no_grad()
    def base_logprobs(self, seqs: list[Sequence], token_budget: int | None = None) -> list[torch.Tensor]:
        """Per-token log-probabilities (CPU, fp32) of each sequence's completion under the frozen base model; a
        forward-only pass may use a larger token budget than the trainer's."""
        with self.model.disable_adapter():
            return self._logprobs(seqs, token_budget)

    # -- updates ---------------------------------------------------------------------------------------------
    def _optimizer(self, member: MemberWeights, lr: float) -> torch.optim.AdamW:
        self._activate(member)
        opt = torch.optim.AdamW(list(self.params.values()), lr=lr, weight_decay=self.cfg.weight_decay)
        if member.optimizer is not None:  # a private copy: a failed step must leave the member's state untouched
            opt.load_state_dict(_to_cpu(member.optimizer))
            for group in opt.param_groups:
                group.update(lr=lr, weight_decay=self.cfg.weight_decay)
        opt.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        return opt

    def _step(self, member: MemberWeights, opt: torch.optim.AdamW) -> dict:
        """Clip, step, and copy the new weights and optimizer state out to the member (only now is it changed)."""
        params = list(self.params.values())
        grad_norm = torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm, error_if_nonfinite=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        member.lora = {n: p.detach().to("cpu", copy=True) for n, p in self.params.items()}
        member.optimizer = _to_cpu(opt.state_dict())
        peak = ({"peak_alloc_gib": round(torch.cuda.max_memory_allocated(self.device) / 2**30, 2)}
                if self.device.type == "cuda" else {})
        return {"grad_norm": float(grad_norm), **peak}

    def update(self, member: MemberWeights, seqs: list[Sequence], lr: float, self_imitation: float = 0.0) -> dict:
        """One AdamW step on

            -sum_{i: w_i != 0} w_i sum_t log pi(y_it) / sum_{i: w_i != 0} |y_i|
            + beta * (-sum_{i imitated} sum_t log pi(y_it) / sum_{i imitated} |y_i|)

        with beta = self_imitation. With beta = 0 the imitation flags are ignored and the computation is exactly the
        advantage-only update. Updates `member` in place (LoRA tensors and optimizer state); without any sequence in
        either term nothing changes.
        """
        t0 = time.perf_counter()
        active = [s for s in seqs if s.weight != 0.0 or (self_imitation > 0 and s.imitate)]
        adv_tokens = sum(len(s.completion) for s in active if s.weight != 0.0)
        imitated = [s for s in active if self_imitation > 0 and s.imitate]
        si_tokens = sum(len(s.completion) for s in imitated)
        stats = {"sequences": len(active), "tokens": sum(len(s.completion) for s in active),
                 "si_sequences": len(imitated), "si_tokens": si_tokens}
        if not active:
            return {"loss": 0.0, "adv_loss": 0.0, "si_loss": 0.0, "grad_norm": 0.0, **stats, "micro_batches": 0,
                    "update_s": 0.0}
        opt = self._optimizer(member, lr)
        batches = self._batches(active)
        adv_sum, si_sum = 0.0, 0.0
        for batch in batches:
            adv = si = torch.zeros((), device=self.device)
            for row, lp in zip(batch, self._forward(batch)):
                for i, a, b in row.spans:
                    if active[i].weight != 0.0:
                        adv = adv - active[i].weight * lp[a:b].sum()
                    if imitated and active[i].imitate:
                        si = si - lp[a:b].sum()
            loss = adv / adv_tokens if adv_tokens else adv
            adv_sum += loss.item()
            if imitated:
                si = si / si_tokens
                si_sum += si.item()
                loss = loss + self_imitation * si
            loss.backward()
        out = self._step(member, opt)
        return {"loss": adv_sum + self_imitation * si_sum, "adv_loss": adv_sum, "si_loss": si_sum, **out, **stats,
                "micro_batches": len(batches), "update_s": round(time.perf_counter() - t0, 3)}

    def distill(self, member: MemberWeights, seqs: list[Sequence], teacher_prompts: list[list[int]],
                lr: float) -> dict:
        """One AdamW step on the token-mean reverse KL over completion tokens,

            sum_i sum_t KL( pi(. | x_i, y_i<t) || base(. | c_i, y_i<t) ) / sum_i |y_i|,

        where x_i is seqs[i].prompt (the member's own prompt), c_i is teacher_prompts[i] (the same question behind a
        prompt that carries the context), y_i the member's sampled completion, and `base` the frozen base model: the
        adapter disabled, no gradient. The KL is over the full vocabulary at every completion position."""
        t0 = time.perf_counter()
        tokens = sum(len(s.completion) for s in seqs)
        opt = self._optimizer(member, lr)
        batches = self._batches(seqs)
        loss_sum = 0.0
        for batch in batches:
            idx = [i for row in batch for i, _, _ in row.spans]
            teacher = [Sequence(teacher_prompts[i], seqs[i].completion) for i in idx]
            with torch.no_grad(), self.model.disable_adapter():
                rows = build_rows(teacher, self.cfg.shared_prefix, self.cfg.token_budget)
                t_hidden = by_sequence(rows, self._hidden(rows)[0])
            s_hidden = by_sequence(batch, self._hidden(batch)[0])
            kl = reverse_kl(torch.cat([s_hidden[i] for i in idx]), torch.cat([t_hidden[j] for j in range(len(idx))]),
                            self.head_weight)
            loss = kl.sum() / tokens
            loss.backward()
            loss_sum += loss.item()
        out = self._step(member, opt)
        return {"loss": loss_sum, "kl": loss_sum, **out, "sequences": len(seqs), "tokens": tokens,
                "micro_batches": len(batches), "update_s": round(time.perf_counter() - t0, 3)}


def gpu_processes() -> str:
    """This GPU's memory and every process on the machine's GPUs, as nvidia-smi reports them now."""
    def query(*args):
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True).stdout.strip()

    return "\n".join([query("-i", os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                            "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv"),
                      query("--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv")])


def load_optimizer(path: Path) -> dict | None:
    return torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None


def adapter_sha256(path: str | Path) -> str:
    """Content hash of everything vLLM reads from an adapter directory: adapter_config.json (rank, alpha, target
    modules) and the weights. The identity used by evaluation records and the harvest manifest."""
    h = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        h.update(f"{name}\0".encode())
        with open(Path(path) / name, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()
