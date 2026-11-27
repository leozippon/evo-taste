"""PyTorch/PEFT policy: a frozen bf16 base model with one LoRA structure whose tensors are swapped per member.

Scoring, the group update and the distillation update share one forward path. Sequences are packed into rows: by
default a row holds a prompt once followed by as many of its completions as fit in the token budget, with an attention
mask that lets each completion see the prompt and itself only (shared-prefix reuse); with `shared_prefix=False` every
(prompt, completion) pair is its own row and plain causal attention is used. Output projections are applied only at
the positions that predict completion tokens, in chunks that are recomputed in the backward pass, so full-vocabulary
fp32 logits never exist for whole sequences. On GPU the LoRA matmuls run in bf16 under autocast, as in vLLM; the fp32
LoRA tensors are the optimizer's master copy. The frozen base model (the distillation teacher, the reference of the
exaggeration target, of the context score and of the deviation) is the same network with the adapter disabled.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
import shutil
import statistics
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
    imitate: bool = False  # eligible for the imitation terms (self-imitation, exaggeration): a successful rollout
    replay: bool = False  # a stored earlier answer for replay imitation, not a rollout of this step


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


TASTE_MIN_SHAPED = 8  # a step with fewer rollouts in groups of two or more eligible ones is not shaped


def taste_factors(scores: list[list[float | None]], strength: float, clip: float) -> tuple[list[list[float]], dict]:
    """Reward factors of the taste-shaped reward for one step. scores[j][k] is the characteristic score of rollout k of
    prompt j if it is eligible (correct, and within the free length where that is required), else None.

    In each group with at least two eligible rollouts, d = s - the group's mean s over them (exactly 0 when their scores
    are equal, so that rounding cannot pass for a difference); sigma is the population standard deviation of all such d
    in the step; an eligible rollout's factor is 1 + strength x clip(d / sigma, -clip, clip). Every other rollout keeps
    1, and so does every rollout of a step with fewer than TASTE_MIN_SHAPED such d or with sigma = 0. The z-values do
    not depend on `strength`, so -strength mirrors the factors around 1. Returns the factors and {shaped, groups, sigma,
    skipped} (`skipped`: why the step was not shaped, else None)."""
    d = {}
    for j, group in enumerate(scores):
        eligible = [(k, s) for k, s in enumerate(group) if s is not None]
        if len(eligible) >= 2:
            same = len({s for _, s in eligible}) == 1
            mean = sum(s for _, s in eligible) / len(eligible)
            d.update({(j, k): 0.0 if same else s - mean for k, s in eligible})
    factors = [[1.0] * len(group) for group in scores]
    sigma = statistics.pstdev(list(d.values())) if d else 0.0
    if len(d) < TASTE_MIN_SHAPED or sigma == 0:
        why = (f"{len(d)} rollouts in groups with two or more eligible, fewer than {TASTE_MIN_SHAPED}"
               if len(d) < TASTE_MIN_SHAPED else "the scores do not vary within groups")
        return factors, {"shaped": 0, "groups": 0, "sigma": sigma if d else None, "skipped": why}
    for (j, k), x in d.items():
        factors[j][k] = 1.0 + strength * min(max(x / sigma, -clip), clip)
    return factors, {"shaped": len(d), "groups": len({j for j, _ in d}), "sigma": sigma, "skipped": None}


def length_residuals(scores: list[list[float | None]],
                     tokens: list[list[int]]) -> tuple[list[list[float | None]], dict]:
    """The length-neutral taste score: over the rollouts that `taste_factors` shapes (eligible ones in groups with two
    or more eligible), the least-squares fit s = a + b log(tokens), and their scores replaced by the residuals s - a -
    b log(tokens); every other score is returned unchanged. With fewer than two such rollouts nothing is fitted; with
    a single length the slope is 0. Returns the scores and {length_intercept, length_slope}."""
    points = [(j, k) for j, group in enumerate(scores) if sum(s is not None for s in group) >= 2
              for k, s in enumerate(group) if s is not None]
    if len(points) < 2:
        return scores, {"length_intercept": None, "length_slope": None}
    x = [math.log(tokens[j][k]) for j, k in points]
    y = [scores[j][k] for j, k in points]
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx if sxx > 0 else 0.0
    intercept = my - slope * mx
    out = [list(group) for group in scores]
    for (j, k), a, b in zip(points, x, y):
        out[j][k] = b - (intercept + slope * a)
    return out, {"length_intercept": intercept, "length_slope": slope}


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


def _chunk_exaggeration(member: torch.Tensor, base: torch.Tensor, weight: torch.Tensor, gamma: float) -> torch.Tensor:
    logits = (member @ weight.T).float()
    with torch.no_grad():  # the target is fixed: no gradient through the member's or the base model's logits
        target = (logits + gamma * (logits - (base @ weight.T).float())).log_softmax(-1)
    return (target.exp() * (target - logits.log_softmax(-1))).sum(-1)


def _over_chunks(fn, a: torch.Tensor, b: torch.Tensor, *args) -> torch.Tensor:
    """fn on KL_CHUNK positions of the hidden states a and b at a time, each chunk recomputed in the backward pass
    when grad is enabled, so that its fp32 [chunk, vocabulary] tensors never exist for more than one chunk."""
    out = []
    for s in range(0, len(a), KL_CHUNK):
        chunk = (a[s:s + KL_CHUNK], b[s:s + KL_CHUNK], *args)
        out.append(checkpoint(fn, *chunk, use_reentrant=False) if torch.is_grad_enabled() else fn(*chunk))
    return torch.cat(out)


def reverse_kl(student: torch.Tensor, teacher: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per position, KL(softmax(student @ W.T) || softmax(teacher @ W.T)) over the full vocabulary, from hidden
    states."""
    return _over_chunks(_chunk_kl, student, teacher, weight)


def exaggeration_kl(member: torch.Tensor, base: torch.Tensor, weight: torch.Tensor, gamma: float) -> torch.Tensor:
    """Per position, KL(q || softmax(l_m)) over the full vocabulary with l_m = member @ W.T, l_b = base @ W.T and the
    fixed target q = softmax(l_m + gamma (l_m - l_b)), computed without gradient: the gradient with respect to l_m is
    softmax(l_m) - q, which moves the member further along its own deviation from the base model."""
    return _over_chunks(_chunk_exaggeration, member, base, weight, gamma)


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
    def logprobs(self, member: MemberWeights, seqs: list[Sequence], token_budget: int | None = None) -> list[torch.Tensor]:
        """Per-token log-probabilities (CPU, fp32) of each sequence's completion under the member's weights."""
        self._activate(member)
        return self._logprobs(seqs, token_budget)

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

    def update(self, member: MemberWeights, seqs: list[Sequence], lr: float, self_imitation: float = 0.0,
               exaggeration=None, replay: float = 0.0) -> dict:
        """One AdamW step on the sum of four terms, each normalised by its own token count:

            -sum_{i: w_i != 0} w_i sum_t log pi(y_it) / sum_{i: w_i != 0} |y_i|
            + beta * (-sum_{i eligible} sum_t log pi(y_it) / sum_{i eligible} |y_i|)
            + weight * sum_{i eligible} sum_t KL(q_it || pi(. | x_i, y_i<t)) / sum_{i eligible} |y_i|
            + replay * (-sum_{i replayed} sum_t log pi(y_it) / sum_{i replayed} |y_i|)

        with beta = self_imitation, `exaggeration` (config.Exaggeration or None) giving gamma and weight, eligible the
        sequences flagged `imitate`, replayed those flagged `replay` (stored earlier answers, which take part in no
        other term), and q_it = softmax(l_m + gamma (l_m - l_b)) the fixed exaggerated target from the member's logits
        l_m and the base model's l_b (adapter disabled, no gradient) on the same prompt and completion. A term whose
        weight is 0 (or exaggeration with gamma 0, whose target is the member itself) is not computed at all, so the
        update is then exactly the one without it; the replay statistics are reported only when its weight is not 0.
        Updates `member` in place (LoRA tensors and optimizer state); without any sequence in any term nothing changes.
        """
        t0 = time.perf_counter()
        si_on = self_imitation > 0
        ex_on = exaggeration is not None and exaggeration.gamma != 0 and exaggeration.weight != 0
        rp_on = replay > 0
        if any(s.replay and (s.weight != 0.0 or s.imitate) for s in seqs):
            raise ValueError("a replayed answer takes part in the replay term only (weight 0, not imitated)")
        active = [s for s in seqs if s.weight != 0.0 or ((si_on or ex_on) and s.imitate) or (rp_on and s.replay)]
        eligible = [s.imitate for s in active]
        adv_tokens = sum(len(s.completion) for s in active if s.weight != 0.0)
        imitated_tokens = sum(len(s.completion) for s, e in zip(active, eligible) if e)
        replay_tokens = sum(len(s.completion) for s in active if s.replay)
        stats = {"sequences": len(active), "tokens": sum(len(s.completion) for s in active),
                 "adv_sequences": sum(s.weight != 0.0 for s in active), "adv_tokens": adv_tokens}
        for name, on in (("si", si_on), ("ex", ex_on)):
            stats.update({f"{name}_sequences": sum(eligible) if on else 0, f"{name}_tokens": imitated_tokens if on else 0})
        if rp_on:
            stats.update(rp_sequences=sum(s.replay for s in active), rp_tokens=replay_tokens)
        if not active:
            return {"loss": 0.0, "adv_loss": 0.0, "si_loss": 0.0, "ex_loss": 0.0, **({"rp_loss": 0.0} if rp_on else {}),
                    "grad_norm": 0.0, **stats, "micro_batches": 0, "update_s": 0.0}
        opt = self._optimizer(member, lr)
        batches = self._batches(active)
        adv_sum = si_sum = ex_sum = rp_sum = 0.0
        for batch in batches:
            mine = [i for row in batch for i, _, _ in row.spans if eligible[i]] if ex_on else []
            if mine:  # the base model on the very same rows first (same ids, positions and mask); only the hidden
                # states that predict eligible completion tokens are kept
                with torch.no_grad(), self.model.disable_adapter():
                    base_all = by_sequence(batch, self._hidden(batch)[0])
                    base_hidden = torch.cat([base_all[i] for i in mine])
                    del base_all
            hidden, targets = self._hidden(batch)
            lp = head_logprobs(hidden, self.head_weight, targets)
            adv = si = rp = torch.zeros((), device=self.device)
            for i, seq_lp in by_sequence(batch, lp).items():
                if active[i].weight != 0.0:
                    adv = adv - active[i].weight * seq_lp.sum()
                if si_on and eligible[i]:
                    si = si - seq_lp.sum()
                if rp_on and active[i].replay:
                    rp = rp - seq_lp.sum()
            loss = adv / adv_tokens if adv_tokens else adv
            adv_sum += loss.item()
            if si_on and imitated_tokens:
                si = si / imitated_tokens
                si_sum += si.item()
                loss = loss + self_imitation * si
            if rp_on and replay_tokens:
                rp = rp / replay_tokens
                rp_sum += rp.item()
                loss = loss + replay * rp
            if mine:
                member_hidden = by_sequence(batch, hidden)
                kl = exaggeration_kl(torch.cat([member_hidden[i] for i in mine]), base_hidden, self.head_weight,
                                     exaggeration.gamma)
                ex = kl.sum() / imitated_tokens
                ex_sum += ex.item()
                loss = loss + exaggeration.weight * ex
            loss.backward()
        out = self._step(member, opt)
        ex_weight = exaggeration.weight if ex_on else 0.0
        total = adv_sum + self_imitation * si_sum + ex_weight * ex_sum
        if rp_on:
            total += replay * rp_sum
        return {"loss": total, "adv_loss": adv_sum, "si_loss": si_sum, "ex_loss": ex_sum,
                **({"rp_loss": rp_sum} if rp_on else {}), **out, **stats, "micro_batches": len(batches),
                "update_s": round(time.perf_counter() - t0, 3)}

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


def scale_adapter(src: str | Path, dst: str | Path, factor: float) -> None:
    """Write to `dst` the adapter of `src` with its update to the base model multiplied by `factor`: every LoRA B
    tensor times the factor, the A tensors and adapter_config.json unchanged, so that (alpha / r) B A becomes factor
    times itself and nothing else differs."""
    src, dst = Path(src), Path(dst)
    tensors = load_file(src / "adapter_model.safetensors")
    other = [n for n in tensors if ".lora_A." not in n and ".lora_B." not in n]
    if other or not any(".lora_B." in n for n in tensors):
        raise ValueError(f"{src}: expected LoRA A and B tensors only, found also {other[:3]}")
    dst.mkdir(parents=True, exist_ok=False)
    save_file({n: t * factor if ".lora_B." in n else t for n, t in tensors.items()}, dst / "adapter_model.safetensors")
    shutil.copyfile(src / "adapter_config.json", dst / "adapter_config.json")


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
