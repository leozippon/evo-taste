"""PyTorch/PEFT policy: a frozen bf16 base model with one LoRA structure whose tensors are swapped per member.

Scoring and the update share one forward path. Sequences are packed into rows: by default one row per prompt that
holds the prompt once followed by all of its completions, with an attention mask that lets each completion see the
prompt and itself only (shared-prefix reuse); with `shared_prefix=False` every (prompt, completion) pair is its own
row and plain causal attention is used. The LM head is applied only at the positions that predict completion tokens,
in chunks that are recomputed in the backward pass, so full-vocabulary fp32 logits never exist for whole sequences.
On GPU the LoRA matmuls run in bf16 under autocast, as in vLLM; the fp32 LoRA tensors are the optimizer's master copy.
"""

from __future__ import annotations

import hashlib
import math
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
HEAD_CHUNK = 1024  # completion positions per LM-head chunk


@dataclass
class Sequence:
    prompt: list[int]
    completion: list[int]  # sampled token ids, including the end-of-turn token when generation stopped
    weight: float = 1.0


@dataclass
class MemberWeights:
    """A member's trainable state, kept on the CPU between turns: fp32 LoRA tensors and AdamW state."""
    lora: dict[str, torch.Tensor]  # PEFT parameter name -> tensor
    optimizer: dict | None = None  # torch AdamW state_dict; None before the first update of a phase


def group_weights(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """Group-relative advantage of one prompt's rollouts: (r - mean) / (population std + eps); 0 if all equal."""
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    if std == 0.0:
        return [0.0] * len(rewards)
    return [(r - mean) / (std + eps) for r in rewards]


@dataclass
class _Row:
    ids: list[int]
    pos: list[int]
    seg: list[int]  # 0 = prompt, k + 1 = k-th completion of the row
    src: list[int]  # row positions whose hidden state predicts each target
    targets: list[int]
    spans: list[tuple[int, int, int]]  # (sequence index, start, end) into targets


def build_rows(seqs: list[Sequence], shared_prefix: bool) -> list[_Row]:
    groups: dict = {}
    for i, s in enumerate(seqs):
        if not s.completion:
            raise ValueError(f"sequence {i} has an empty completion")
        groups.setdefault(tuple(s.prompt) if shared_prefix else i, []).append(i)
    rows = []
    for members in groups.values():
        prompt = seqs[members[0]].prompt
        n = len(prompt)
        row = _Row(list(prompt), list(range(n)), [0] * n, [], [], [])
        for k, i in enumerate(members):
            c = seqs[i].completion
            start = len(row.ids)
            row.spans.append((i, len(row.targets), len(row.targets) + len(c)))
            row.src += [n - 1] + list(range(start, start + len(c) - 1))
            row.targets += c
            row.ids += c
            row.pos += range(n, n + len(c))
            row.seg += [k + 1] * len(c)
        rows.append(row)
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

    # -- member state ----------------------------------------------------------------------------------------
    def new_member(self, seed: int, b_std: float = 0.0) -> MemberWeights:
        """PEFT's initialization (A: Kaiming-uniform, B: 0) drawn from `seed`; b_std > 0 perturbs B ~ N(0, b_std)."""
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
    def _forward(self, rows: list[_Row]) -> list[torch.Tensor]:
        """Per row, the fp32 log-probabilities of its targets (attached to the graph when grad is enabled)."""
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
        h = hidden.reshape(B * L, -1)[torch.tensor(src, device=dev)]
        lp = head_logprobs(h, self.head_weight, torch.tensor(targets, device=dev))
        return list(lp.split([len(r.targets) for r in rows]))

    @torch.no_grad()
    def logprobs(self, member: MemberWeights, seqs: list[Sequence]) -> list[torch.Tensor]:
        """Per-token log-probabilities (CPU, fp32) of each sequence's completion under the member's weights."""
        self._activate(member)
        out: list = [None] * len(seqs)
        for batch in micro_batches(build_rows(seqs, self.cfg.shared_prefix), self.cfg.token_budget):
            for row, lp in zip(batch, self._forward(batch)):
                for i, a, b in row.spans:
                    out[i] = lp[a:b].cpu()
        return out

    def update(self, member: MemberWeights, seqs: list[Sequence], lr: float) -> dict:
        """One AdamW step on  -sum_i w_i sum_t log pi(y_it) / sum_i |y_i|  over sequences with w_i != 0.

        Updates `member` in place (LoRA tensors and optimizer state). Without any non-zero weight nothing changes.
        """
        t0 = time.perf_counter()
        active = [s for s in seqs if s.weight != 0.0]
        tokens = sum(len(s.completion) for s in active)
        if not active:
            return {"loss": 0.0, "grad_norm": 0.0, "sequences": 0, "tokens": 0, "micro_batches": 0, "update_s": 0.0}
        self._activate(member)
        params = list(self.params.values())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=self.cfg.weight_decay)
        if member.optimizer is not None:
            opt.load_state_dict(member.optimizer)
            for group in opt.param_groups:
                group.update(lr=lr, weight_decay=self.cfg.weight_decay)
        opt.zero_grad(set_to_none=True)
        loss, n_batches = self._accumulate(active)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm, error_if_nonfinite=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        member.lora = {n: p.detach().to("cpu", copy=True) for n, p in self.params.items()}
        member.optimizer = _to_cpu(opt.state_dict())
        return {"loss": loss, "grad_norm": float(grad_norm), "sequences": len(active), "tokens": tokens,
                "micro_batches": n_batches, "update_s": round(time.perf_counter() - t0, 3)}

    def _accumulate(self, active: list[Sequence]) -> tuple[float, int]:
        """Backward of the update loss into the LoRA gradients, micro-batch by micro-batch; returns (loss, batches)."""
        tokens = sum(len(s.completion) for s in active)
        batches = micro_batches(build_rows(active, self.cfg.shared_prefix), self.cfg.token_budget)
        loss_sum = 0.0
        for batch in batches:
            loss = 0.0
            for row, lp in zip(batch, self._forward(batch)):
                for i, a, b in row.spans:
                    loss = loss - active[i].weight * lp[a:b].sum()
            loss = loss / tokens
            loss.backward()
            loss_sum += loss.item()
        return loss_sum, len(batches)


def save_optimizer(member: MemberWeights, path: Path) -> None:
    torch.save(member.optimizer, path)


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
