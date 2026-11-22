"""Member life cycle: seeding, amplification steps, probes, culling, checkpoints and resume.

A configured member id names a *slot*. The slot's first lineage has the slot id; after a cull, the k-th replacement
lineage is "<slot>~<k>" with a new seed and the same arm and scaffold. Each lineage lives in
runs/<run>/members/<lineage>/; its state.json is rewritten only at checkpoints and is the commit point for resume.
"""

from __future__ import annotations

import dataclasses
import logging
import random
import shutil
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch

from evotaste import config
from evotaste.records import (append_jsonl, filter_jsonl, now, read_json, read_jsonl_gz, write_json,
                              write_jsonl_gz)
from evotaste.sampler import Request, Sample
from evotaste.tasks import (SYSTEM_PROMPT, Streams, Suites, Task, load_scaffolds, prompt_ids, score_many, stable_seed,
                            system_prompt)
from evotaste.trainer import MemberWeights, Sequence, group_weights, load_optimizer

log = logging.getLogger(__name__)

# Policy-consistency check (docs/dev/INFRA_NOTES.md, "Trainer-sampler consistency"): with the right adapter the
# per-token mean |trainer - sampler| log-probability is about 0.013 and the summed |difference| about 0.003 per token;
# serving the base model instead of a random adapter gave 0.11 and 0.065. The limits sit between the two.
CHECK_TOKEN_MEAN_ABS_MAX = 0.04
CHECK_SEQ_ABS_PER_TOKEN_MAX = 0.015
PROBE_SUITES = ("train", "selection", "foundation")


class ConsistencyError(RuntimeError):
    """The trainer does not reproduce the sampler's log-probabilities: wrong or stale adapter, or a prompt mismatch."""


class Run:
    """Shared context of one run directory. The worker attaches `policy` and `sampler`."""

    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.cfg = config.load(self.dir / "config.yaml")
        self.suites = Suites(self.cfg.suites)
        self.streams = Streams(self.suites, self.cfg.seed, self.cfg.probes)
        self.probe_tasks = [(suite, t) for suite in PROBE_SUITES for t in self.streams.probe_tasks[suite]]
        self.scaffolds = load_scaffolds(self.cfg.scaffolds)
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model)
        self.policy = None
        self.sampler = None
        self.prompt = lru_cache(maxsize=8192)(self._prompt)

    def _prompt(self, question: str, teacher: str = "plain") -> list[int]:
        system = SYSTEM_PROMPT if teacher == "plain" else system_prompt(self.scaffolds[teacher])
        return prompt_ids(self.tokenizer, question, system)

    def check_probe_set(self) -> None:
        """Members' probes are compared with the base model's: the probe set rebuilt by this code must be the one the
        base model answered at prepare time (a code change that alters instances would otherwise go unnoticed)."""
        base = [(r["suite"], r["uid"], r["question"]) for r in
                read_jsonl_gz(self.dir / "base" / "probe_responses.jsonl.gz")]
        if base != [(suite, t.uid, t.question) for suite, t in self.probe_tasks]:
            raise RuntimeError(f"{self.dir}: the probe set differs from the one the base model answered at prepare "
                               "time; the run was prepared by different code and cannot be continued with this one")

    @property
    def base_probe(self) -> dict:
        return read_json(self.dir / "base" / "probe_summary.json")

    def seed_pool(self, teacher: str) -> list[tuple[list[int], list[list[int]]]]:
        """Prompts with at least one correct teacher sample: (plain prompt ids, correct completions), by uid."""
        by_uid: dict[str, tuple[list[int], list]] = {}
        for r in read_jsonl_gz(self.dir / "seeding" / f"{teacher}.jsonl.gz"):
            by_uid.setdefault(r["uid"], (r["prompt_ids"], []))[1].append(r["completion_ids"])
        if not by_uid:
            raise RuntimeError(f"seeding pool {teacher!r} has no correct sample")
        return [by_uid[u] for u in sorted(by_uid)]


def summarize_probe(records: list[dict]) -> dict:
    """Accuracy per probe suite (macro over families), per family, and format statistics of one probe."""
    by_family: dict[str, list[float]] = {}
    suite_of = {}
    for r in records:
        by_family.setdefault(r["family"], []).append(r["reward"])
        suite_of[r["family"]] = r["suite"]
    family_acc = {f: sum(v) / len(v) for f, v in sorted(by_family.items())}
    accuracy = {}
    for suite in PROBE_SUITES:
        accs = [a for f, a in family_acc.items() if suite_of[f] == suite]
        accuracy[suite] = sum(accs) / len(accs)
    n = len(records)
    return {"n": n, "accuracy": accuracy, "by_family": family_acc,
            "parsed": sum(r["parsed"] for r in records) / n, "truncated": sum(r["truncated"] for r in records) / n,
            "completion_tokens": sum(r["completion_tokens"] for r in records) / n}


def response_records(pairs: list[tuple[Task, Sample]]) -> list[dict]:
    """One record per (task, sample), all scored in one batch by the scorer processes."""
    scores = score_many([(task, sample.text, sample.finish_reason) for task, sample in pairs])
    return [_record(task, sample, s) for (task, sample), s in zip(pairs, scores)]


def _record(task: Task, sample: Sample, s) -> dict:
    return {"uid": task.uid, "family": task.family, "question": task.question, "answer": task.answer,
            "text": sample.text, "completion_tokens": len(sample.token_ids), "finish_reason": sample.finish_reason,
            "extracted": s.extracted, "score": s.score, "reward": s.reward, "parsed": s.parsed,
            "truncated": s.truncated, "score_error": s.error, "logprob": sample.logprob}


@dataclass
class Lineage:
    id: str
    slot: str
    generation: int  # 0 for the slot's first lineage, k for its k-th replacement
    arm: str
    scaffold: str | None
    seed: int
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


def current_lineage(run: Run, slot: str) -> Lineage | None:
    """The slot's living (or finished) lineage, following replacements; None if the slot is exhausted."""
    lin = load_lineage(run, slot)
    if lin is None:
        spec = run.cfg.member(slot)
        lin = Lineage(slot, slot, 0, spec.arm, spec.scaffold, spec.seed)
        write_json(lineage_dir(run, slot) / "state.json", dataclasses.asdict(lin))
    while lin.status == "culled":
        if lin.replaced_by is None:
            return None
        lin = load_lineage(run, lin.replaced_by)
        if lin is None:
            raise RuntimeError(f"slot {slot}: replacement lineage state missing")
    return lin


class Member:
    def __init__(self, run: Run, lin: Lineage):
        self.run, self.lin = run, lin
        self.seeding, self.amp, self.culling = run.cfg.settings(run.cfg.member(lin.slot))
        self.dir = lineage_dir(run, lin.id)
        self.teacher = lin.scaffold or "plain"
        self.weights: MemberWeights | None = None
        self.version: str | None = None  # adapter version currently served for this member

    def __repr__(self):
        return f"Member({self.lin.id}, step {self.lin.step}, {self.lin.status})"

    @property
    def finished(self) -> bool:
        return self.lin.status in ("done", "culled")

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
        self.weights = MemberWeights(self.run.policy.load_adapter(ckpt), load_optimizer(ckpt / "optimizer.pt"))
        log.info("%s resumed from %s", self.lin.id, ckpt)

    def checkpoint_dir(self, step: int) -> Path:
        return self.dir / "checkpoints" / f"step_{step:06d}"

    # -- seeding -------------------------------------------------------------------------------------------------
    def seed(self) -> None:
        """Warm-up: positive updates on correct teacher samples, trained without the scaffold in the prompt."""
        lin, policy = self.lin, self.run.policy
        b_std = self.seeding.perturb_std if lin.arm == "perturb" else 0.0
        self.weights = policy.new_member(lin.seed, b_std)
        pool = self.run.seed_pool(self.teacher)
        for i in range(self.seeding.steps):
            rng = random.Random(stable_seed("warmup", lin.seed, i))
            seqs = []
            for _ in range(self.seeding.batch):
                prompt, completions = rng.choice(pool)
                seqs.append(Sequence(prompt, rng.choice(completions), 1.0))
            stats = policy.update(self.weights, seqs, self.seeding.lr)
            append_jsonl(self.dir / "history.jsonl", {"phase": "seed", "step": i, "time": now(), **stats})
        self.weights.optimizer = None  # amplification starts from a fresh AdamW state
        log.info("%s seeded (%d positive updates, teacher %s)", lin.id, self.seeding.steps, self.teacher)

    def publish(self) -> float:
        """Hand the current weights to the sampler as a new adapter version (bf16, as vLLM serves them)."""
        t0 = time.perf_counter()
        self.version = self.run.sampler.publish(
            self.lin.id, lambda path: self.run.policy.save_adapter(self.weights, path, torch.bfloat16))
        return time.perf_counter() - t0

    def retire(self) -> None:
        self.run.sampler.retire(self.lin.id)
        self.version = None

    # -- amplification --------------------------------------------------------------------------------------------
    def step_tasks(self) -> list[Task]:
        return self.run.streams.train_batch(self.lin.step, self.amp.prompts)

    def rollout_requests(self, tasks: list[Task]) -> list[Request]:
        return [Request(self.version, self.run.prompt(t.question), self.amp.rollouts, self.amp.max_tokens)
                for t in tasks]

    def amplify(self, tasks: list[Task], samples: list[list[Sample]], generate_s: float, batch_members: int):
        """One group update on this step's rollouts; then publish, record, and advance the step counter."""
        t, policy = self.lin.step, self.run.policy
        t0 = time.perf_counter()
        seqs, sampler_lps, records = [], [], []
        groups = 0
        flat = response_records([(task, s) for task, group in zip(tasks, samples) for s in group])
        for j, (task, group) in enumerate(zip(tasks, samples)):
            recs, flat = flat[:len(group)], flat[len(group):]
            weights = group_weights([r["reward"] for r in recs])
            groups += any(w != 0.0 for w in weights)
            prompt = self.run.prompt(task.question)
            for k, (s, r, w) in enumerate(zip(group, recs, weights)):
                seqs.append(Sequence(prompt, s.token_ids, w))
                sampler_lps.append(s.token_logprobs)
                records.append({"step": t, "prompt": j, "k": k, **r, "weight": w})
        score_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        check = self.consistency_check(seqs, sampler_lps)
        check_s = time.perf_counter() - t0
        stats = policy.update(self.weights, seqs, self.amp.lr)
        publish_s = self.publish()
        t0 = time.perf_counter()
        write_jsonl_gz(self.dir / "rollouts" / f"step_{t:06d}.jsonl.gz", records)
        n = len(records)
        by_family: dict[str, list[float]] = {}
        for r in records:
            by_family.setdefault(r["family"], []).append(r["reward"])
        append_jsonl(self.dir / "history.jsonl", {
            "phase": "amp", "step": t, "time": now(), "version": self.version,
            "reward": sum(r["reward"] for r in records) / n,
            "reward_by_family": {f: sum(v) / len(v) for f, v in sorted(by_family.items())},
            "completion_tokens": sum(r["completion_tokens"] for r in records) / n,
            "parsed": sum(r["parsed"] for r in records) / n, "truncated": sum(r["truncated"] for r in records) / n,
            "groups": groups, "prompts": len(tasks), **stats, "check": check,
            "timing": {"generate_s": round(generate_s, 3), "batch_members": batch_members,
                       "score_s": round(score_s, 3), "check_s": round(check_s, 3), "update_s": stats["update_s"],
                       "publish_s": round(publish_s, 3), "write_s": round(time.perf_counter() - t0, 3)}})
        self.lin.step = t + 1

    def consistency_check(self, seqs: list[Sequence], sampler_lps: list[list[float]]) -> dict:
        """Re-score a few rollouts with the trainer; fail loudly unless they match the sampler within the noise."""
        idx = list(range(0, len(seqs), self.amp.rollouts))[:self.run.cfg.trainer.check_rollouts]
        ours = self.run.policy.logprobs(self.weights, [seqs[i] for i in idx])
        token_abs, seq_abs, tokens = 0.0, [], 0
        for i, lp in zip(idx, ours):
            theirs = torch.tensor(sampler_lps[i], dtype=torch.float32)
            token_abs += float((lp - theirs).abs().sum())
            seq_abs.append(abs(float(lp.sum() - theirs.sum())))
            tokens += len(lp)
        stats = {"rollouts": len(idx), "tokens": tokens, "token_mean_abs": round(token_abs / tokens, 5),
                 "seq_mean_abs": round(sum(seq_abs) / len(seq_abs), 4),
                 "seq_abs_per_token": round(sum(seq_abs) / tokens, 5)}
        if stats["token_mean_abs"] > CHECK_TOKEN_MEAN_ABS_MAX or stats["seq_abs_per_token"] > CHECK_SEQ_ABS_PER_TOKEN_MAX:
            raise ConsistencyError(f"{self.lin.id} step {self.lin.step} ({self.version}): trainer and sampler "
                                   f"log-probabilities disagree beyond the noise: {stats}")
        return stats

    # -- probes, culling, checkpoints ------------------------------------------------------------------------------
    def probe_due(self) -> bool:
        return self.lin.step % self.run.cfg.probes.every == 0 or self.lin.step >= self.amp.steps

    def probe_requests(self) -> list[Request]:
        return [Request(self.version, self.run.prompt(t.question), 1, self.run.cfg.probes.max_tokens)
                for _, t in self.run.probe_tasks]

    def finish_probe(self, samples: list[list[Sample]]) -> None:
        """Record the probe at the current step, decide culling, write the checkpoint and commit the state."""
        s = self.lin.step
        records = [{"step": s, "suite": suite, **r} for (suite, _), r in
                   zip(self.run.probe_tasks, response_records([(t, g[0]) for (_, t), g in zip(self.run.probe_tasks, samples)]))]
        write_jsonl_gz(self.dir / "probe_responses" / f"step_{s:06d}.jsonl.gz", records)
        summary = {"step": s, "time": now(), **summarize_probe(records)}
        summary["cull"] = self.cull_reason(summary)
        append_jsonl(self.dir / "probes.jsonl", summary)
        self.lin.probes.append(summary)
        self.write_checkpoint()
        if summary["cull"] is not None:
            self.cull(summary["cull"])
        elif s >= self.amp.steps:
            self.lin.status = "done"
        else:
            self.lin.status = "active"
        write_json(self.dir / "state.json", dataclasses.asdict(self.lin))  # commit point
        for path in (self.dir / "checkpoints").glob("step_*/optimizer.pt"):
            if path.parent != self.checkpoint_dir(s):
                path.unlink()
        log.info("%s probe at step %d: %s, cull=%s", self.lin.id, s,
                 {k: round(v, 3) for k, v in summary["accuracy"].items()}, summary["cull"])

    def cull_reason(self, summary: dict) -> str | None:
        """Collapse criteria, applied only from the end of the grace period on."""
        grace = self.culling.grace
        if summary["step"] < grace or summary["step"] == 0:
            return None
        base = self.run.base_probe["accuracy"]
        if summary["parsed"] < 0.5:
            return "fewer than half of the probe answers are parseable"
        if summary["truncated"] > 0.5:
            return "more than half of the probe answers are truncated"
        if summary["accuracy"]["foundation"] < 0.5 * base["foundation"]:
            return "foundation accuracy below half the base model's"
        previous = [p for p in self.lin.probes if p["step"] >= grace]
        if summary["accuracy"]["train"] < base["train"] and previous and previous[-1]["accuracy"]["train"] < base["train"]:
            return "train-test accuracy below the base model's on two consecutive probes"
        return None

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
        spec = self.run.cfg.member(lin.slot)
        new = Lineage(f"{lin.slot}~{generation}", lin.slot, generation, lin.arm, lin.scaffold,
                      stable_seed("lineage", spec.seed, generation))
        write_json(lineage_dir(self.run, new.id) / "state.json", dataclasses.asdict(new))
        lin.replaced_by = new.id
        log.warning("%s culled (%s); replaced by %s", lin.id, reason, new.id)
