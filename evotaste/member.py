"""Member life cycle: seeding, clone starts, amplification steps, probes, culling, checkpoints and resume.

A configured member id names a *slot*. The slot's first lineage has the slot id; after a cull, the k-th replacement
lineage is "<slot>~<k>" with a new seed and the same context (or parent). Each lineage lives in
runs/<run>/members/<lineage>/; its state.json is rewritten only at checkpoints and is the commit point for resume.
Every sampling request carries a seed derived from the lineage seed and the request's place (phase, step, prompt).
"""

from __future__ import annotations

import dataclasses
import logging
import random
import shutil
import time
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path

import torch

from evotaste import config
from evotaste.records import (append_jsonl, filter_jsonl, now, read_json, read_jsonl, read_jsonl_gz, write_json,
                              write_jsonl_gz)
from evotaste.sampler import Request, Sample
from evotaste.tasks import Streams, Suites, Task, load_strata, prompt_ids, score_many, stable_seed
from evotaste.trainer import MemberWeights, Sequence, advantages, gpu_processes, length_returns, load_optimizer

log = logging.getLogger(__name__)

# Policy-consistency check (docs/dev/ENGINE_NOTES.md): with the right adapter the per-token mean |trainer - sampler|
# log-probability is about 0.006 and the summed |difference| about 0.002 per token; serving the base model instead of
# a random adapter, or the reverse, gave 0.12 and 0.073. The limits sit between the two.
# A CUDA out-of-memory error in the trainer (a co-tenant took memory on the shared card) is retried from clean
# state: OOM_ATTEMPTS tries, OOM_WAIT_S apart (about 15 minutes in all), then the error propagates.
OOM_ATTEMPTS = 10
OOM_WAIT_S = 100
CHECK_TOKEN_MEAN_ABS_MAX = 0.04
CHECK_SEQ_ABS_PER_TOKEN_MAX = 0.015


def check_sample(lengths: list[int], rollouts: int, min_tokens: int, min_rollouts: int) -> list[int]:
    """Rollouts (indices into the step's batch, prompt-major) for the consistency check: the first rollout of each
    prompt, then the second of each, and so on, until both minimums are met, or all of them. A token-based size keeps
    an isolated near-tie token (about 2 nats in bf16) from deciding the check when answers are short."""
    order = [j * rollouts + k for k in range(rollouts) for j in range(len(lengths) // rollouts)]
    sample, tokens = [], 0
    for i in order:
        if tokens >= min_tokens and len(sample) >= min_rollouts:
            break
        sample.append(i)
        tokens += lengths[i]
    return sample


def _check_passes(stats: dict) -> bool:
    return stats["token_mean_abs"] <= CHECK_TOKEN_MEAN_ABS_MAX and stats["seq_abs_per_token"] <= CHECK_SEQ_ABS_PER_TOKEN_MAX


def retry_on_oom(policy, what: str, fn):
    """fn() with retries after CUDA out-of-memory errors. fn must not change anything before it succeeds (the
    trainer's scoring and updates copy a member's tensors and optimizer state in, and out only at the very end), so
    a retry computes exactly what an uninterrupted call would have."""
    for attempt in range(1, OOM_ATTEMPTS + 1):
        try:
            return fn()
        except torch.OutOfMemoryError as e:
            if attempt == OOM_ATTEMPTS:
                raise
            error = str(e).splitlines()[0]
        policy.release_memory()  # outside the except block, so that the failed frames' tensors are gone
        log.warning("%s: CUDA out of memory (attempt %d of %d: %s); retrying in %d s. GPU memory now:\n%s",
                    what, attempt, OOM_ATTEMPTS, error, OOM_WAIT_S, gpu_processes())
        time.sleep(OOM_WAIT_S)


class ConsistencyError(RuntimeError):
    """The trainer does not reproduce the sampler's log-probabilities: wrong or stale adapter, or a prompt mismatch."""


class Run:
    """Shared context of one run directory. The worker attaches `policy` and `sampler`."""

    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.cfg = config.load(self.dir / "config.yaml")
        self.contexts = config.load_contexts(self.cfg.contexts)
        self.suites = Suites(self.cfg.suites)
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model)
        self.policy = None
        self.sampler = None
        self.prompt = lru_cache(maxsize=8192)(self._prompt)

    @cached_property
    def streams(self) -> Streams:
        strata = self.cfg.strata and load_strata(self.cfg.strata.path, self.cfg.strata.sha256)
        return Streams(self.suites, self.cfg.seed, strata=strata)

    @cached_property
    def probe_tasks(self) -> list[Task]:
        return list(self.streams.probe_tasks)

    def _prompt(self, question: str, context: str | None = None) -> list[int]:
        """The member's prompt (no context), or the teacher's prompt carrying the named context."""
        if context is None:
            return prompt_ids(self.tokenizer, question)
        return prompt_ids(self.tokenizer, question, context=self.contexts[context])

    def check_probe_set(self) -> None:
        """Members' probes are compared with the base model's: the probe set rebuilt by this code must be the one the
        base model answered at prepare time (a code change that alters items would otherwise go unnoticed)."""
        base = [(r["uid"], r["question"]) for r in read_jsonl_gz(self.dir / "base" / "probe_responses.jsonl.gz")]
        if base != [(t.uid, t.question) for t in self.probe_tasks]:
            raise RuntimeError(f"{self.dir}: the probe set differs from the one the base model answered at prepare "
                               "time; the run was prepared by different code and cannot be continued with this one")

    @property
    def base_probe(self) -> dict:
        return read_json(self.dir / "base" / "probe_summary.json")


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def summarize_probe(records: list[dict]) -> dict:
    """Accuracy of one probe (macro over families, overall and by domain), per family, wrapper compliance per wrapper,
    and format statistics."""
    by_family: dict[str, list[float]] = {}
    domain_of, by_wrapper = {}, {}
    for r in records:
        by_family.setdefault(r["family"], []).append(r["reward"])
        domain_of[r["family"]] = r["domain"]
        by_wrapper.setdefault(r["wrapper"], []).append(r["complied"])
    family_acc = {f: _mean(v) for f, v in sorted(by_family.items())}
    by_domain = {d: _mean(a for f, a in family_acc.items() if domain_of[f] == d)
                 for d in sorted(set(domain_of.values()))}
    return {"n": len(records), "accuracy": _mean(family_acc.values()), "by_domain": by_domain, "by_family": family_acc,
            "complied": _mean(r["complied"] for r in records), "correct_any": _mean(r["correct_any"] for r in records),
            "by_wrapper": {w: _mean(v) for w, v in sorted(by_wrapper.items(), key=lambda x: str(x[0]))},
            "truncated": _mean(r["truncated"] for r in records),
            "completion_tokens": _mean(r["completion_tokens"] for r in records)}


def response_records(pairs: list[tuple[Task, Sample]], token_ids: bool = False) -> list[dict]:
    """One record per (task, sample), all scored in one batch; with token_ids the sampled ids are kept (probes)."""
    scores = score_many([(task, sample.text, sample.finish_reason) for task, sample in pairs])
    return [_record(task, sample, s, token_ids) for (task, sample), s in zip(pairs, scores)]


def _record(task: Task, sample: Sample, score, token_ids: bool) -> dict:
    return {"uid": task.uid, "family": task.family, "domain": task.domain, "kind": task.kind, "wrapper": task.wrapper,
            "stratum": task.stratum, "question": task.question, "text": sample.text,
            "completion_tokens": len(sample.token_ids), "finish_reason": sample.finish_reason,
            "logprob": sample.logprob, **dataclasses.asdict(score),
            **({"token_ids": sample.token_ids} if token_ids else {})}


def probe_requests(model: str, tasks: list[Task], prompt, max_tokens: int, *seed_parts) -> list[Request]:
    """One sample per probe item; item i is drawn with seed stable_seed("probe", *seed_parts, i)."""
    return [Request(model, prompt(t.question), 1, max_tokens, stable_seed("probe", *seed_parts, i))
            for i, t in enumerate(tasks)]


@dataclass
class Lineage:
    id: str
    slot: str
    generation: int  # 0 for the slot's first lineage, k for its k-th replacement
    context: str | None  # seeding context (a clone's is its parent's)
    seed: int
    parent: dict | None = None  # a clone's start: {"lineage": <parent lineage>, "step": <its step>}
    status: str = "new"  # new (no checkpoint) | active | done | culled
    step: int = 0  # amplification updates completed at the last checkpoint (the checkpoint's step)
    probes: list = field(default_factory=list)  # probe summaries up to the last checkpoint
    cull_reason: str | None = None
    replaced_by: str | None = None


def lineage_dir(run: Run, lineage_id: str) -> Path:
    return run.dir / "members" / lineage_id


def load_lineage(run: Run, lineage_id: str) -> Lineage | None:
    path = lineage_dir(run, lineage_id) / "state.json"
    return Lineage(**read_json(path)) if path.exists() else None


def first_lineage(cfg: config.RunConfig, slot: str, generation: int = 0, seed: int | None = None) -> Lineage:
    spec = cfg.member(slot)
    parent = {"lineage": spec.parent.member, "step": spec.parent.step} if spec.parent else None
    return Lineage(f"{slot}~{generation}" if generation else slot, slot, generation, cfg.context_of(spec),
                   spec.seed if seed is None else seed, parent)


def current_lineage(run: Run, slot: str) -> Lineage | None:
    """The slot's living (or finished) lineage, following replacements; None if the slot is exhausted."""
    lin = load_lineage(run, slot)
    if lin is None:
        lin = first_lineage(run.cfg, slot)
        write_json(lineage_dir(run, slot) / "state.json", dataclasses.asdict(lin))
    while lin.status == "culled":
        if lin.replaced_by is None:
            return None
        lin = load_lineage(run, lin.replaced_by)
        if lin is None:
            raise RuntimeError(f"slot {slot}: replacement lineage state missing")
    return lin


def ancestry_history(run: Run, lineage_id: str, before: int) -> list[dict]:
    """The amplification history behind a lineage's state at step `before`: its own lines before that step and, for a
    clone, its parent's before the clone point, and so on up the chain."""
    lines = [r for r in read_jsonl(lineage_dir(run, lineage_id) / "history.jsonl")
             if r["phase"] == "amp" and r["step"] < before]
    lin = load_lineage(run, lineage_id)
    if lin is not None and lin.parent is not None:
        lines += ancestry_history(run, lin.parent["lineage"], min(before, lin.parent["step"]))
    return lines


def load_checkpoint(policy, ckpt: Path, history: list[dict], step: int, owner: str) -> MemberWeights:
    """Weights and optimizer state of a checkpoint at `step`; refuses when the optimizer state is missing although
    `history` shows amplification updates before that step (a lost file must not silently restart AdamW)."""
    optimizer = load_optimizer(ckpt / "optimizer.pt")
    updated = any(r["phase"] == "amp" and r["step"] < step and r.get("sequences", 0) > 0 for r in history)
    if optimizer is None and updated:
        raise RuntimeError(f"{ckpt}: optimizer.pt is missing, but {owner} has made updates before step {step}")
    return MemberWeights(policy.load_adapter(ckpt), optimizer)


class Member:
    def __init__(self, run: Run, lin: Lineage):
        self.run, self.lin = run, lin
        self.spec = run.cfg.member(lin.slot)
        self.seeding, self.amp, self.culling = run.cfg.settings(self.spec)
        self.dir = lineage_dir(run, lin.id)
        # checkpoints this lineage keeps with optimizer state, for the clones that start from it now or later
        self.clone_points = run.cfg.clone_points(lin.id) if lin.generation == 0 else set()
        self.weights: MemberWeights | None = None
        self.version: str | None = None  # adapter version currently served for this member

    def __repr__(self):
        return f"Member({self.lin.id}, step {self.lin.step}, {self.lin.status})"

    @property
    def finished(self) -> bool:
        return self.lin.status in ("done", "culled")

    @property
    def is_clone(self) -> bool:
        return self.spec.parent is not None

    # -- resume ------------------------------------------------------------------------------------------------
    def restore(self) -> None:
        """Bring the lineage directory back to its last checkpoint and load the weights from it."""
        if self.finished:
            return
        if self.lin.status == "new":
            for sub in ("rollouts", "probe_responses", "checkpoints"):
                shutil.rmtree(self.dir / sub, ignore_errors=True)
            for name in ("history.jsonl", "probes.jsonl"):
                (self.dir / name).unlink(missing_ok=True)
            return
        s = self.lin.step
        filter_jsonl(self.dir / "history.jsonl", lambda r: r["phase"] == "seed" or r["step"] < s)
        filter_jsonl(self.dir / "probes.jsonl", lambda r: r["step"] <= s)
        for path in (self.dir / "rollouts").glob("step_*"):
            if int(path.name[5:11]) >= s:
                path.unlink()
        for path in (self.dir / "probe_responses").glob("step_*"):
            if int(path.name[5:11]) > s:
                path.unlink()
        for path in (self.dir / "checkpoints").iterdir():
            if path.name.endswith(".tmp") or int(path.name[5:11]) > s:
                shutil.rmtree(path)
        ckpt = self.checkpoint_dir(s)
        history = ancestry_history(self.run, self.lin.id, s)  # a clone's updates before its start are its parents'
        self.weights = load_checkpoint(self.run.policy, ckpt, history, s, self.lin.id)
        log.info("%s resumed from %s", self.lin.id, ckpt)

    def checkpoint_dir(self, step: int) -> Path:
        return self.dir / "checkpoints" / f"step_{step:06d}"

    # -- seeding -------------------------------------------------------------------------------------------------
    def begin_seeding(self) -> None:
        """Fresh LoRA tensors from the lineage seed (B = 0: the member starts as the base model)."""
        self.weights = self.run.policy.new_member(self.lin.seed)

    @property
    def seeding_steps(self) -> int:
        return 0 if self.seeding.method == "none" else self.seeding.steps

    def seed_tasks(self, i: int) -> list[Task]:
        return self.run.streams.seed_batch(i, self.seeding.batch, self.seeding.domains)

    def seed_requests(self, i: int, tasks: list[Task]) -> list[Request]:
        """opcd: one sample of the member itself per prompt, without the context; imitation: `samples` samples of the
        base model per prompt, behind the prompt that carries the context."""
        s, lin = self.seeding, self.lin
        if s.method == "opcd":
            return [Request(self.version, self.run.prompt(t.question), 1, s.max_tokens,
                            stable_seed("opcd", lin.seed, i, j)) for j, t in enumerate(tasks)]
        return [Request(self.run.cfg.model, self.run.prompt(t.question, lin.context), s.samples, s.max_tokens,
                        stable_seed("imitation", lin.seed, i, j)) for j, t in enumerate(tasks)]

    def seed_update(self, i: int, tasks: list[Task], samples: list[list[Sample]], generate_s: float,
                    batch_members: int) -> None:
        """Seeding update i on its samples, then (opcd) publish the new weights; one history line."""
        policy, lin, s = self.run.policy, self.lin, self.seeding
        flat = [x for g in samples for x in g]
        line = {"phase": "seed", "method": s.method, "step": i, "time": now(),
                "completion_tokens": _mean(len(x.token_ids) for x in flat),
                "truncated": _mean(x.finish_reason != "stop" for x in flat)}
        timing = {"generate_s": round(generate_s, 3), "batch_members": batch_members}
        if s.method == "opcd":
            seqs = [Sequence(self.run.prompt(t.question), g[0].token_ids) for t, g in zip(tasks, samples)]
            t0 = time.perf_counter()
            lps = [g[0].token_logprobs for g in samples]
            line["check"] = retry_on_oom(policy, f"{lin.id} seeding step {i} check",
                                         lambda: self.consistency_check(seqs, lps, 1))
            timing["check_s"] = round(time.perf_counter() - t0, 3)
            teacher = [self.run.prompt(t.question, lin.context) for t in tasks]
            stats = retry_on_oom(policy, f"{lin.id} seeding step {i}",
                                 lambda: policy.distill(self.weights, seqs, teacher, s.lr))
            timing["publish_s"] = round(sum(self.publish().values()), 3)
        else:
            rewards = iter(x.reward for x in score_many([(t, x.text, x.finish_reason)
                                                         for t, g in zip(tasks, samples) for x in g]))
            pool = []
            for t, g in zip(tasks, samples):
                correct = [x.token_ids for x in g if next(rewards) == 1.0]
                if correct:
                    pool.append((self.run.prompt(t.question), correct))
            if not pool:
                raise RuntimeError(f"{lin.id} seeding step {i}: no correct teacher sample among "
                                   f"{len(flat)} on {len(tasks)} prompts")
            rng = random.Random(stable_seed("imitation-draw", lin.seed, i))
            seqs = []
            for _ in range(s.batch):  # a prompt with a correct sample uniformly, then one of its correct samples
                prompt, completions = rng.choice(pool)
                seqs.append(Sequence(prompt, rng.choice(completions)))
            line.update(correct=sum(len(c) for _, c in pool), prompts_with_correct=len(pool))
            stats = retry_on_oom(policy, f"{lin.id} seeding step {i}", lambda: policy.update(self.weights, seqs, s.lr))
        timing["update_s"] = stats["update_s"]
        append_jsonl(self.dir / "history.jsonl", {**line, **stats, "timing": timing})

    def end_seeding(self) -> None:
        self.weights.optimizer = None  # amplification starts from a fresh AdamW state
        log.info("%s seeded (%s, %d updates, context %s)", self.lin.id, self.seeding.method, self.seeding_steps,
                 self.lin.context)

    # -- clones --------------------------------------------------------------------------------------------------
    def parent_checkpoint(self) -> Path | None:
        """The parent's checkpoint at the clone point once the parent has committed it; None until then."""
        p = self.lin.parent
        parent = load_lineage(self.run, p["lineage"])
        if parent is None or parent.status == "new" or parent.step < p["step"]:
            if parent is not None and parent.status == "culled":
                raise RuntimeError(f"{self.lin.id}: parent {p['lineage']} was culled at step {parent.step}, before "
                                   f"the clone point {p['step']}")
            return None
        ckpt = lineage_dir(self.run, p["lineage"]) / "checkpoints" / f"step_{p['step']:06d}"
        if not (ckpt / "adapter_model.safetensors").exists():
            raise RuntimeError(f"{self.lin.id}: parent {p['lineage']} is at step {parent.step} but kept no checkpoint "
                               f"at the clone point {p['step']}")
        return ckpt

    def start_clone(self, ckpt: Path) -> None:
        """The parent's weights and optimizer state at the clone point, and its step counter. The clone commits its
        own copy of that checkpoint with its first probe."""
        p = self.lin.parent
        history = ancestry_history(self.run, p["lineage"], p["step"])
        self.weights = load_checkpoint(self.run.policy, ckpt, history, p["step"], p["lineage"])
        self.lin.step = p["step"]
        log.info("%s cloned from %s at step %d", self.lin.id, p["lineage"], p["step"])

    def publish(self) -> dict:
        """Hand the current weights to the sampler as a new adapter version (bf16, as vLLM serves them); returns the
        seconds spent saving, loading, unloading the previous version and removing its directory."""
        self.version, parts = self.run.sampler.publish(
            self.lin.id, lambda path: self.run.policy.save_adapter(self.weights, path, torch.bfloat16))
        return parts

    def retire(self) -> None:
        self.run.sampler.retire(self.lin.id)
        self.version = None

    # -- amplification --------------------------------------------------------------------------------------------
    def step_tasks(self) -> list[Task]:
        return self.run.streams.train_batch(self.lin.step, self.amp.prompts)

    def rollout_requests(self, tasks: list[Task]) -> list[Request]:
        return [Request(self.version, self.run.prompt(t.question), self.amp.rollouts, self.amp.max_tokens,
                        stable_seed("rollout", self.lin.seed, self.lin.step, j)) for j, t in enumerate(tasks)]

    def amplify(self, tasks: list[Task], samples: list[list[Sample]], generate_s: float, batch_members: int):
        """One group update on this step's rollouts; then publish, record, and advance the step counter."""
        t, policy, amp = self.lin.step, self.run.policy, self.amp
        t0 = time.perf_counter()
        seqs, sampler_lps, records = [], [], []
        groups = 0
        flat = response_records([(task, s) for task, group in zip(tasks, samples) for s in group])
        for j, (task, group) in enumerate(zip(tasks, samples)):
            recs, flat = flat[:len(group)], flat[len(group):]
            returns = length_returns([r["reward"] for r in recs], [r["completion_tokens"] for r in recs],
                                     amp.max_tokens, amp.length.free, amp.length.slope)
            weights = advantages(returns, amp.advantage)
            groups += any(w != 0.0 for w in weights)
            prompt = self.run.prompt(task.question)
            for k, (s, r, g, w) in enumerate(zip(group, recs, returns, weights)):
                seqs.append(Sequence(prompt, s.token_ids, w, imitate=r["reward"] == 1.0))
                sampler_lps.append(s.token_logprobs)
                records.append({"step": t, "prompt": j, "k": k, **r, "return": g, "weight": w})
        score_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        check = retry_on_oom(policy, f"{self.lin.id} step {t} check",
                             lambda: self.consistency_check(seqs, sampler_lps, amp.rollouts))
        check_s = time.perf_counter() - t0
        stats = retry_on_oom(policy, f"{self.lin.id} step {t} update",
                             lambda: policy.update(self.weights, seqs, amp.lr, amp.self_imitation))
        publish = self.publish()
        t0 = time.perf_counter()
        write_jsonl_gz(self.dir / "rollouts" / f"step_{t:06d}.jsonl.gz", records)
        by_family: dict[str, list[float]] = {}
        for r in records:
            by_family.setdefault(r["family"], []).append(r["reward"])
        append_jsonl(self.dir / "history.jsonl", {
            "phase": "amp", "step": t, "time": now(), "version": self.version,
            "reward": _mean(r["reward"] for r in records), "return": _mean(r["return"] for r in records),
            "reward_by_family": {f: _mean(v) for f, v in sorted(by_family.items())},
            "completion_tokens": _mean(r["completion_tokens"] for r in records),
            "complied": _mean(r["complied"] for r in records), "truncated": _mean(r["truncated"] for r in records),
            "groups": groups, "prompts": len(tasks), **stats, "check": check,
            "timing": {"generate_s": round(generate_s, 3), "batch_members": batch_members,
                       "score_s": round(score_s, 3), "check_s": round(check_s, 3), "update_s": stats["update_s"],
                       "publish_s": round(sum(publish.values()), 3), "publish": publish,
                       "write_s": round(time.perf_counter() - t0, 3)}})
        self.lin.step = t + 1

    def consistency_check(self, seqs: list[Sequence], sampler_lps: list[list[float]], rollouts: int) -> dict:
        """Re-score part of the batch with the trainer; fail loudly unless it matches the sampler within the noise.

        A wrong, stale or swapped adapter shifts every token (0.12 per token against a noise of 0.006), so it fails
        on any sample and again on the whole batch; a failure caused by a few near-tie tokens is diluted when the
        check is repeated on all rollouts of the step, which happens once before raising."""
        cfg = self.run.cfg.trainer
        sample = check_sample([len(s.completion) for s in seqs], rollouts, cfg.check_tokens, cfg.check_rollouts)
        stats = self._check_stats(seqs, sampler_lps, sample)
        if not _check_passes(stats) and len(sample) < len(seqs):
            log.warning("%s step %d: consistency check failed on a sample %s; repeating on all %d rollouts",
                        self.lin.id, self.lin.step, stats, len(seqs))
            stats = {**self._check_stats(seqs, sampler_lps, list(range(len(seqs)))), "first_attempt": stats}
            log.warning("%s step %d: consistency check on all rollouts %s", self.lin.id, self.lin.step,
                        {k: v for k, v in stats.items() if k != "first_attempt"})
        if not _check_passes(stats):
            raise ConsistencyError(f"{self.lin.id} step {self.lin.step} ({self.version}): trainer and sampler "
                                   f"log-probabilities disagree beyond the noise: {stats}")
        return stats

    def _check_stats(self, seqs: list[Sequence], sampler_lps: list[list[float]], idx: list[int]) -> dict:
        ours = self.run.policy.logprobs(self.weights, [seqs[i] for i in idx])
        token_abs, seq_abs, tokens = 0.0, [], 0
        for i, lp in zip(idx, ours):
            theirs = torch.tensor(sampler_lps[i], dtype=torch.float32)
            token_abs += float((lp - theirs).abs().sum())
            seq_abs.append(abs(float(lp.sum() - theirs.sum())))
            tokens += len(lp)
        return {"rollouts": len(idx), "tokens": tokens, "token_mean_abs": round(token_abs / tokens, 5),
                "seq_mean_abs": round(sum(seq_abs) / len(seq_abs), 4),
                "seq_abs_per_token": round(sum(seq_abs) / tokens, 5)}

    # -- probes, culling, checkpoints ------------------------------------------------------------------------------
    def probe_due(self) -> bool:
        return self.lin.step % self.run.cfg.probes.every == 0 or self.lin.step >= self.amp.steps

    def checkpoint_due(self, every: int) -> bool:
        """A light checkpoint (no probe) every `every` steps and at every clone point of this lineage."""
        return self.lin.step % every == 0 or self.lin.step in self.clone_points

    def probe_requests(self) -> list[Request]:
        return probe_requests(self.version, self.run.probe_tasks, self.run.prompt, self.run.cfg.probes.max_tokens,
                              self.lin.seed, self.lin.step)

    def finish_probe(self, samples: list[list[Sample]]) -> None:
        """Record the probe at the current step, decide culling, write the checkpoint and commit the state."""
        s = self.lin.step
        records = [{"step": s, **r} for r in
                   response_records([(t, g[0]) for t, g in zip(self.run.probe_tasks, samples)], token_ids=True)]
        write_jsonl_gz(self.dir / "probe_responses" / f"step_{s:06d}.jsonl.gz", records)
        summary = {"step": s, "time": now(), **summarize_probe(records)}
        summary["cull"] = self.cull_reason(summary)
        summary["culled"] = summary["cull"] is not None and self.culling.enabled
        append_jsonl(self.dir / "probes.jsonl", summary)
        self.lin.probes.append(summary)
        self.write_checkpoint()
        if summary["culled"]:
            self.cull(summary["cull"])
        elif s >= self.amp.steps:
            self.lin.status = "done"
        else:
            self.lin.status = "active"
        write_json(self.dir / "state.json", dataclasses.asdict(self.lin))  # commit point
        self.prune_checkpoints()
        if summary["cull"] is not None and not summary["culled"]:
            log.warning("%s would be culled at step %d (%s); culling is disabled", self.lin.id, s, summary["cull"])
        log.info("%s probe at step %d: accuracy %.3f %s, cull=%s", self.lin.id, s, summary["accuracy"],
                 {k: round(v, 3) for k, v in summary["by_domain"].items()}, summary["cull"])

    def cull_reason(self, summary: dict) -> str | None:
        """Collapse criteria, applied only from the end of the grace period on."""
        grace = self.culling.grace
        if summary["step"] < grace or summary["step"] == 0:
            return None
        base = self.run.base_probe["accuracy"]
        if summary["complied"] < 0.5:
            return "fewer than half of the probe answers are in the wrapper asked for"
        if summary["truncated"] > 0.5:
            return "more than half of the probe answers are truncated"
        if summary["accuracy"] < 0.5 * base:
            return "probe accuracy below half the base model's"
        previous = [p for p in self.lin.probes if p["step"] >= grace]
        if summary["accuracy"] < base and previous and previous[-1]["accuracy"] < base:
            return "probe accuracy below the base model's on two consecutive probes"
        return None

    def light_checkpoint(self) -> None:
        """What resume needs (adapter, optimizer, state) at the current step, without a probe."""
        self.write_checkpoint()
        write_json(self.dir / "state.json", dataclasses.asdict(self.lin))  # commit point
        self.prune_checkpoints()
        log.info("%s light checkpoint at step %d", self.lin.id, self.lin.step)

    def prune_checkpoints(self) -> None:
        """After a commit: keep the current checkpoint, the probe-step ones (final evaluation may name them) and the
        clone points; drop older light checkpoints, and the optimizer state of every checkpoint but the current one
        and the clone points."""
        probe_steps = {p["step"] for p in self.lin.probes}
        for path in (self.dir / "checkpoints").glob("step_*"):
            if path.name.endswith(".tmp") or (step := int(path.name[5:11])) == self.lin.step:
                continue
            if step in self.clone_points:
                continue
            if step not in probe_steps:
                shutil.rmtree(path)
            else:
                (path / "optimizer.pt").unlink(missing_ok=True)

    def write_checkpoint(self) -> None:
        final = self.checkpoint_dir(self.lin.step)
        tmp = final.with_name(final.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        self.run.policy.save_adapter(self.weights, tmp)  # fp32: exact resume
        if self.weights.optimizer is not None:
            torch.save(self.weights.optimizer, tmp / "optimizer.pt")
        shutil.rmtree(final, ignore_errors=True)  # a redone step after a crash
        tmp.rename(final)

    def cull(self, reason: str) -> None:
        lin = self.lin
        lin.status, lin.cull_reason = "culled", reason
        generation = lin.generation + 1
        if generation > self.culling.max_replacements:
            log.warning("%s culled (%s); slot %s exhausted", lin.id, reason, lin.slot)
            return
        new = first_lineage(self.run.cfg, lin.slot, generation, stable_seed("lineage", self.spec.seed, generation))
        write_json(lineage_dir(self.run, new.id) / "state.json", dataclasses.asdict(new))
        lin.replaced_by = new.id
        log.warning("%s culled (%s); replaced by %s", lin.id, reason, new.id)
