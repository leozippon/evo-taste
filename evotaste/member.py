"""Member life cycle: seeding, clone starts, amplification steps, probes, culling, checkpoints and resume.

A configured member id names a *slot*. The slot's first lineage has the slot id; after a cull, the k-th replacement
lineage is "<slot>~<k>" with a new seed and the same context (or parent). Each lineage lives in
runs/<run>/members/<lineage>/; its state.json is rewritten only at checkpoints and is the commit point for resume.
Every sampling request carries a seed derived from the lineage seed and the request's place (phase, step, prompt).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import random
import shutil
import statistics
import time
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path

import torch

from evotaste import config
from evotaste.records import (append_jsonl, filter_jsonl, now, read_json, read_jsonl, read_jsonl_gz, write_json,
                              write_jsonl_gz)
from evotaste.sampler import Request, Sample
from evotaste.tasks import (Streams, Suites, Task, load_pass_rates, load_strata, probe_set, prompt_ids, rewrap,
                            score_many, stable_seed, train_cell)
from evotaste.trainer import (MemberWeights, Sequence, advantages, gpu_processes, length_residuals, length_returns,
                              load_optimizer, taste_factors)

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
# Self-chosen practice: every source's moving average before the member's first reward on it. Near the base model's
# expected training reward per source (0.40-0.49 by the census), so that a source's average does not depend on how
# often it has been observed; relative to the base model (reference `base`) the expected gain at the start, 0.
PRACTICE_START = {None: 0.5, "base": 0.0}


def practice_shares(defaults: dict[str, float], averages: dict[str, float], settings: config.Practice) -> dict:
    """Source s's share of a member's batch: default_s exp(eta (r_s - mean_s r)), clipped to floor..ceiling x
    default_s and renormalised. Clipping and renormalisation are solved jointly (the shares are clip(c x_s, lo_s,
    hi_s) with c such that they sum to 1), so that the final shares respect both bounds; a clip followed by a plain
    renormalisation could leave them."""
    names = list(defaults)
    mean = sum(averages[s] for s in names) / len(names)
    x = [defaults[s] * math.exp(settings.eta * (averages[s] - mean)) for s in names]
    lo = [settings.floor * defaults[s] for s in names]
    hi = [settings.ceiling * defaults[s] for s in names]

    def total(c):
        return sum(min(max(c * v, a), b) for v, a, b in zip(x, lo, hi))
    # total(c) is continuous, nondecreasing and linear between the knots; total(0) = sum(lo) <= 1 <= sum(hi)
    prev = 0.0
    for knot in sorted({a / v for v, a in zip(x, lo)} | {b / v for v, b in zip(x, hi)}):
        if total(knot) >= 1:
            break
        prev = knot
    low, high = total(prev), total(knot)
    c = knot if high == low else prev + (1 - low) * (knot - prev) / (high - low)
    return {s: min(max(c * v, a), b) for s, v, a, b in zip(names, x, lo, hi)}


@dataclass
class PracticeRecord:
    """A practising member's own record, saved with every checkpoint (practice.json) and inherited by its clones."""
    averages: dict[str, float]  # source -> moving average of the member's training reward on it
    drawn: dict[str, int]  # training-stream cell -> items the member has drawn from it in its own batches


# Consolidating one's own frontier (amplification.revisit). A member with revisits keeps an item record, saved with
# every checkpoint as ITEMS_FILE. Its frontier is the items it has solved at least once but in fewer than 3/4 of at
# least `min_seen` attempts; `own` revisits draw from it with probability proportional to 1 - the item's census pass
# rate + REVISIT_EPSILON, so that items the base model rarely solves come first and none has probability 0.
REVISIT_EPSILON = 0.05
ITEMS_FILE = "items.jsonl.gz"
YOKED_FILE = "yoked.json"  # a yoked member's fixed revisit set, saved with every checkpoint


@dataclass
class Item:
    """A member's record of one training item it has been shown."""
    attempts: int = 0  # rollouts on it
    successes: int = 0  # rollouts with the scorer's (unshaped) reward 1
    first: int | None = None  # the step of the first success
    wrapper: str | None = None  # the first successful answer's wrapper and sampled ids: kept only if it was recorded
    answer: list[int] | None = None  # live and had at most length.free completion tokens


def record_attempts(items: dict[str, Item], step: int, attempts, free: int) -> None:
    """Count one step's attempts, in batch order (prompt-major), each (uid, wrapper, reward, completion tokens, sampled
    ids or None). The first success of an item keeps its wrapper and ids if they are given and it is within `free`."""
    for uid, wrapper, reward, tokens, ids in attempts:
        item = items.setdefault(uid, Item())
        item.attempts += 1
        if reward == 1.0:
            item.successes += 1
            if item.first is None:
                item.first = step
                if ids is not None and tokens <= free:
                    item.wrapper, item.answer = wrapper, list(ids)


def in_frontier(item: Item, min_seen: int) -> bool:
    """Solved at least once, in fewer than 3/4 of its attempts, over at least `min_seen` attempts."""
    return item.successes >= 1 and item.attempts >= min_seen and 4 * item.successes < 3 * item.attempts


def write_items(path: Path, items: dict[str, Item]) -> None:
    write_jsonl_gz(path, [{"uid": uid, **dataclasses.asdict(item)} for uid, item in items.items()])


def read_items(path: Path) -> dict[str, Item]:
    return {r["uid"]: Item(**{k: v for k, v in r.items() if k != "uid"}) for r in read_jsonl_gz(path)}


def revisit_slots(share: float, prompts: int) -> int:
    """Revisits per batch: share x prompts rounded half up, at least one when share > 0."""
    n = math.floor(share * prompts + 0.5)
    return max(n, 1) if share > 0 else 0


def weighted_draws(uids: list[str], weights: list[float], n: int, rng: random.Random) -> list[str]:
    """n distinct uids (all of them if there are fewer), by successive sampling: each draw takes one of the uids not yet
    drawn with probability proportional to its weight."""
    uids, weights, out = list(uids), list(weights), []
    while uids and len(out) < n:
        x = rng.random() * sum(weights)
        i, acc = 0, weights[0]
        while acc <= x and i < len(weights) - 1:
            i += 1
            acc += weights[i]
        out.append(uids.pop(i))
        weights.pop(i)
    return out


def revisit_plan(stream: list[Task], items: dict[str, Item], settings: config.Revisit, rates: dict[str, float],
                 train_task, seeds: tuple, yoked: list[str] | None = None) -> tuple[list[Task], dict]:
    """A revisiting member's batch: the step's shared batch with its last revisit_slots(...) prompts replaced by
    revisits, and the history entry that describes them.

    Candidates exclude every item of the shared batch, so that no item is asked twice in a step. `own` draws from the
    frontier with weights 1 - census pass rate + REVISIT_EPSILON (`weighted_draws`, seeded by `seeds`). `matched` makes
    the same draw on its own record and replaces each drawn item, in draw order, by the item of the same source with the
    nearest census pass rate among those shown at least `min_seen` times, outside the frontier and not taken yet (ties
    in a seeded order); so the two arms revisit equally hard, equally familiar items and differ only in whose successes
    they follow. `yoked` draws like `own`, from the member's fixed set `yoked` (`yoked_set`) instead of its frontier.
    Slots left empty keep the shared batch's prompts, so the non-revisit part of the batch is always a prefix of the
    shared batch. Every revisit is the stream's own task of a train-part pool item (`train_task`), with the default
    wrapper; with no slot the batch is the shared batch itself."""
    n = revisit_slots(settings.share, len(stream))
    shown = {t.uid for t in stream}
    frontier = {uid for uid, item in items.items() if in_frontier(item, settings.min_seen)}
    if settings.source == "yoked":
        pool = sorted(set(yoked) - shown)
        tasks = {uid: train_task(uid) for uid in pool}
        chosen = weighted_draws(pool, [1 - rates[uid] + REVISIT_EPSILON for uid in pool], n,
                                random.Random(stable_seed("revisit", *seeds)))
        return stream[:len(stream) - len(chosen)] + [tasks[uid] for uid in chosen], {
            "source": "yoked", "slots": n, "frontier": len(frontier), "candidates": len(pool), "revisits": chosen,
            "shortfall": n - len(chosen)}
    pool = sorted(frontier - shown)
    tasks = {uid: train_task(uid) for uid in pool}
    own = weighted_draws(pool, [1 - rates[uid] + REVISIT_EPSILON for uid in pool], n,
                         random.Random(stable_seed("revisit", *seeds)))
    chosen: list[str | None] = own
    if settings.source == "matched":
        candidates = sorted(uid for uid, item in items.items()
                            if uid not in frontier and uid not in shown and item.attempts >= settings.min_seen)
        tasks.update((uid, train_task(uid)) for uid in candidates)
        random.Random(stable_seed("revisit-match", *seeds)).shuffle(candidates)
        rank = {uid: i for i, uid in enumerate(candidates)}
        by_source: dict[str, list[str]] = {}
        for uid in candidates:
            by_source.setdefault(tasks[uid].family, []).append(uid)
        chosen, taken = [], set()
        for uid in own:
            options = [c for c in by_source.get(tasks[uid].family, []) if c not in taken]
            best = min(options, key=lambda c: (abs(rates[c] - rates[uid]), rank[c])) if options else None
            chosen.append(best)
            taken.add(best)
    revisits = [tasks[uid] for uid in chosen if uid is not None]
    batch = stream[:len(stream) - len(revisits)] + revisits
    return batch, {"source": settings.source, "slots": n, "frontier": len(frontier), "own": own, "revisits": chosen,
                   "shortfall": n - len(revisits)}


def rebuild_items(run: "Run", lineage_id: str, before: int) -> dict[str, Item]:
    """The item record of a lineage's state at step `before`, from the rollouts recorded by it and, before its start,
    by its chain of parents (committed steps only): attempts, successes and first successes. The rollout records do not
    keep sampled ids, so no answer is known."""
    files, lid, end = [], lineage_id, before
    while True:
        lin = load_lineage(run, lid)
        if lin is None or lin.step < end:
            raise RuntimeError(f"cannot rebuild the item record of {lineage_id}: {lid} has not committed step {end}")
        start = lin.parent["step"] if lin.parent else 0
        files[:0] = [(t, lineage_dir(run, lid) / "rollouts" / f"step_{t:06d}.jsonl.gz") for t in range(start, end)]
        if lin.parent is None:
            break
        lid, end = lin.parent["lineage"], start
    items: dict[str, Item] = {}
    for t, path in files:
        if not path.exists():
            raise RuntimeError(f"cannot rebuild the item record of {lineage_id}: {path} is missing")
        record_attempts(items, t, [(r["uid"], r["wrapper"], r["reward"], r["completion_tokens"], None)
                                   for r in read_jsonl_gz(path)], 0)
    return items


def yoked_set(run: "Run", donor: config.Donor, own: dict[str, Item], min_seen: int) -> list[str]:
    """A yoked member's revisit set, fixed at its clone point: the items of the donor's frontier at donor.step (its
    record rebuilt from the recorded rollouts) that are not in the member's own frontier (`own`, its record at the
    clone point), sorted."""
    mine = {uid for uid, item in own.items() if in_frontier(item, min_seen)}
    theirs = rebuild_items(run, donor.member, donor.step)
    return sorted(uid for uid, item in theirs.items() if in_frontier(item, min_seen) and uid not in mine)


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


def stream_version(run_dir: Path) -> int:
    """The training-stream version a run was prepared with (prepared.json `stream_version`); a run prepared before
    versions were recorded has version 1. A run that is not prepared has no stream yet."""
    path = Path(run_dir) / "prepared.json"
    if not path.exists():
        raise RuntimeError(f"{run_dir} is not prepared: its stream version is recorded by `worker prepare`")
    return read_json(path).get("stream_version", 1)


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
        return Streams(self.suites, self.cfg.seed, strata=strata, version=stream_version(self.dir))

    @cached_property
    def base_pass_rates(self) -> dict[str, float]:
        """uid -> the base model's census pass rate on a training item (the pinned strata table)."""
        return load_pass_rates(self.cfg.strata.path, self.cfg.strata.sha256)

    @cached_property
    def probe_tasks(self) -> list[Task]:
        return probe_set(self.suites)  # needs no stream: prepare and fork check it before prepared.json exists

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
        self.practice: PracticeRecord | None = None  # with amplification.practice only
        self.items: dict[str, Item] | None = None  # the item record, with amplification.revisit only
        self.yoked: list[str] | None = None  # the fixed revisit set, with revisit source yoked only
        self.plan: dict | None = None  # this step's revisits, from step_tasks to amplify
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
        self.practice = self.load_practice(ckpt, s)
        self.items = self.load_items(ckpt, s)
        self.yoked = self.load_yoked(ckpt)
        log.info("%s resumed from %s", self.lin.id, ckpt)

    def checkpoint_dir(self, step: int) -> Path:
        return self.dir / "checkpoints" / f"step_{step:06d}"

    # -- self-chosen practice ------------------------------------------------------------------------------------
    def new_practice(self) -> PracticeRecord | None:
        """Equal averages for every training source, nothing drawn; None for a member that does not practise."""
        if self.amp.practice is None:
            return None
        start = PRACTICE_START[self.amp.practice.reference]
        return PracticeRecord({s: start for s in self.run.streams.source_shares}, {})

    def load_practice(self, ckpt: Path, step: int) -> PracticeRecord | None:
        """The practice record saved with a checkpoint (this lineage's, or a clone's parent's). A checkpoint without
        one is the start of practice only at step 0 (after seeding); later it is refused, like a lost optimizer
        state."""
        if self.amp.practice is None:
            return None
        if (ckpt / "practice.json").exists():
            return PracticeRecord(**read_json(ckpt / "practice.json"))
        if step > 0:
            raise RuntimeError(f"{ckpt}: practice.json is missing, but {self.lin.id} practises and starts at step "
                               f"{step}")
        return self.new_practice()

    def practice_shares(self) -> dict[str, float]:
        return practice_shares(self.run.streams.source_shares, self.practice.averages, self.amp.practice)

    # -- the item record (revisits) -------------------------------------------------------------------------------
    def load_items(self, ckpt: Path, step: int) -> dict[str, Item] | None:
        """The item record saved with one of this lineage's checkpoints; a checkpoint without it is the start of the
        record only at step 0, where nothing has been shown yet (a forked member's step-0 checkpoint)."""
        if self.amp.revisit is None:
            return None
        if (ckpt / ITEMS_FILE).exists():
            return read_items(ckpt / ITEMS_FILE)
        if step > 0:
            raise RuntimeError(f"{ckpt}: {ITEMS_FILE} is missing, but {self.lin.id} revisits and starts at step {step}")
        return {}

    def inherit_items(self, ckpt: Path) -> dict[str, Item] | None:
        """A clone's item record: the one its parent saved at the clone point, or, for a parent that keeps none, the
        record rebuilt from the recorded rollouts of the parent and its own parents up to the clone point."""
        if self.amp.revisit is None:
            return None
        p = self.lin.parent
        if (ckpt / ITEMS_FILE).exists():
            return read_items(ckpt / ITEMS_FILE)
        parent = self.run.cfg.member(load_lineage(self.run, p["lineage"]).slot)
        if p["step"] > 0 and self.run.cfg.settings(parent)[1].revisit is not None:
            raise RuntimeError(f"{ckpt}: {ITEMS_FILE} is missing, but {p['lineage']} keeps an item record")
        items = rebuild_items(self.run, p["lineage"], p["step"])
        log.info("%s: item record rebuilt from the rollouts of %s up to step %d (%d items)", self.lin.id,
                 p["lineage"], p["step"], len(items))
        return items

    def load_yoked(self, ckpt: Path) -> list[str] | None:
        """A yoked member's fixed set, saved with each of its checkpoints since its clone start."""
        rv = self.amp.revisit
        if rv is None or rv.source != "yoked":
            return None
        if not (ckpt / YOKED_FILE).exists():
            raise RuntimeError(f"{ckpt}: {YOKED_FILE} is missing, but {self.lin.id} revisits a donor's frontier")
        saved = read_json(ckpt / YOKED_FILE)
        if saved["donor"] != dataclasses.asdict(rv.donor):
            raise RuntimeError(f"{ckpt}: {YOKED_FILE} was drawn from donor {saved['donor']}, not {rv.donor}")
        return saved["items"]

    # -- seeding -------------------------------------------------------------------------------------------------
    def begin_seeding(self) -> None:
        """Fresh LoRA tensors from the lineage seed (B = 0: the member starts as the base model)."""
        self.weights = self.run.policy.new_member(self.lin.seed)
        self.practice = self.new_practice()
        self.items = None if self.amp.revisit is None else {}

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
        """The parent's weights, optimizer state, practice record and item record at the clone point, and its step
        counter. The clone commits its own copy of that checkpoint with its first probe."""
        p = self.lin.parent
        history = ancestry_history(self.run, p["lineage"], p["step"])
        self.weights = load_checkpoint(self.run.policy, ckpt, history, p["step"], p["lineage"])
        self.practice = self.load_practice(ckpt, p["step"])
        self.items = self.inherit_items(ckpt)
        rv = self.amp.revisit
        if rv is not None and rv.source == "yoked":  # fixed now, for the whole course
            self.yoked = yoked_set(self.run, rv.donor, self.items, rv.min_seen)
            log.info("%s: %d items in the frontier of %s at step %d and not in its own", self.lin.id,
                     len(self.yoked), rv.donor.member, rv.donor.step)
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
        """The run's shared batch of this step; with practice (eta > 0) the member's own batch: proportions from its
        averages, items continuing its own walk through the stream's pools; with revisits the shared batch with its
        last prompts replaced by revisits (`revisit_plan`, which `amplify` then records)."""
        p, rv = self.amp.practice, self.amp.revisit
        if p is not None and p.eta != 0:
            return self.run.streams.member_batch(self.lin.step, self.amp.prompts, self.practice_shares(),
                                                 self.practice.drawn)
        batch = self.run.streams.train_batch(self.lin.step, self.amp.prompts)
        if rv is None:
            return batch
        batch, self.plan = revisit_plan(batch, self.items, rv, self.run.base_pass_rates, self.run.streams.train_task,
                                        (self.run.cfg.seed, self.lin.seed, self.lin.step), self.yoked)
        return batch

    def rollout_requests(self, tasks: list[Task]) -> list[Request]:
        return [Request(self.version, self.run.prompt(t.question), self.amp.rollouts, self.amp.max_tokens,
                        stable_seed("rollout", self.lin.seed, self.lin.step, j)) for j, t in enumerate(tasks)]

    def amplify(self, tasks: list[Task], samples: list[list[Sample]], generate_s: float, batch_members: int):
        """One group update on this step's rollouts; then publish, record, and advance the step counter."""
        t, policy, amp = self.lin.step, self.run.policy, self.amp
        rv, plan, first = amp.revisit, None, len(tasks)  # first: the index of the first revisit prompt
        if rv is not None:  # the revisits step_tasks drew for this step are the last prompts of the batch
            plan, self.plan = self.plan, None
            revisited = [uid for uid in plan["revisits"] if uid is not None]
            first = len(tasks) - len(revisited)
            if [task.uid for task in tasks[first:]] != revisited:
                raise RuntimeError(f"{self.lin.id} step {t}: the batch is not the one its revisits were drawn for")
        t0 = time.perf_counter()
        seqs, sampler_lps, groups_records = [], [], []
        flat = response_records([(task, s) for task, group in zip(tasks, samples) for s in group])
        for task, group in zip(tasks, samples):
            recs, flat = flat[:len(group)], flat[len(group):]
            groups_records.append(recs)
            prompt = self.run.prompt(task.question)
            for s, r in zip(group, recs):  # eligible for the imitation terms and the taste-shaped reward
                eligible = r["reward"] == 1.0 and (not amp.imitate_within_free
                                                   or r["completion_tokens"] <= amp.length.free)
                seqs.append(Sequence(prompt, s.token_ids, 0.0, imitate=eligible))  # weights follow below
                sampler_lps.append(s.token_logprobs)
        score_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        check = retry_on_oom(policy, f"{self.lin.id} step {t} check",
                             lambda: self.consistency_check(seqs, sampler_lps, amp.rollouts))
        check_s = time.perf_counter() - t0
        rewards = [[r["reward"] for r in recs] for recs in groups_records]
        taste, taste_of, factors = {}, None, None
        if amp.taste is not None and amp.taste.strength != 0:  # strength 0: nothing is computed
            taste_of, factors, taste_stats = self.taste_shaping(seqs, groups_records, sampler_lps)
            taste = {"taste": taste_stats}
        records, groups, i = [], 0, 0
        for j, recs in enumerate(groups_records):
            tokens = [r["completion_tokens"] for r in recs]
            shaped = rewards[j] if factors is None else [x * f for x, f in zip(rewards[j], factors[j])]
            returns = length_returns(shaped, tokens, amp.max_tokens, amp.length.free, amp.length.slope)
            weights = advantages(returns, amp.advantage)
            groups += any(w != 0.0 for w in weights)
            if factors is not None:  # the advantage shaping adds, against the one the scored rewards give
                plain = advantages(length_returns(rewards[j], tokens, amp.max_tokens, amp.length.free,
                                                  amp.length.slope), amp.advantage)
                taste_stats["adv_shaping"] += sum(abs(w - p) for w, p in zip(weights, plain)) / len(seqs)
                taste_stats["adv_plain"] += sum(abs(p) for p in plain) / len(seqs)
            for k, (r, g, w) in enumerate(zip(recs, returns, weights)):
                seqs[i].weight = w
                records.append({"step": t, "prompt": j, "k": k, **r, "return": g, "weight": w, **(
                    {} if factors is None else {"taste_score": taste_of.get(i), "taste_factor": factors[j][k]}), **(
                    {} if rv is None else {"revisit": j >= first})})
                i += 1
        replay = self.replay_sequences(tasks[first:]) if rv is not None and rv.imitate > 0 else []
        stats = retry_on_oom(policy, f"{self.lin.id} step {t} update",
                             lambda: policy.update(self.weights, seqs + replay, amp.lr, amp.self_imitation,
                                                   amp.exaggeration, 0.0 if rv is None else rv.imitate))
        publish = self.publish()
        t0 = time.perf_counter()
        write_jsonl_gz(self.dir / "rollouts" / f"step_{t:06d}.jsonl.gz", records)
        revisit = {}
        if rv is not None:  # this step's revisits, then this step's attempts into the record
            on, off = ([r["reward"] for r in records if r["revisit"] == x] for x in (True, False))
            revisit = {"revisit": {**plan, "reward": _mean(on) if on else None,
                                   "stream_reward": _mean(off) if off else None}}
            record_attempts(self.items, t, [(task.uid, task.wrapper, r["reward"], r["completion_tokens"], s.token_ids)
                                            for task, group, recs in zip(tasks, samples, groups_records)
                                            for s, r in zip(group, recs)], amp.length.free)
        by_family: dict[str, list[float]] = {}
        for r in records:
            by_family.setdefault(r["family"], []).append(r["reward"])
        reward_by_family = {f: _mean(v) for f, v in sorted(by_family.items())}  # training families are sources
        practice = {}
        if self.practice is not None:  # the shares this batch was drawn with (default ones at eta 0), then update
            practice = {"practice": {"shares": self.practice_shares(), "averages": dict(self.practice.averages)}}
            tracked = reward_by_family
            if amp.practice.reference == "base":  # the gain over the base model's pass rate on the same items
                rates: dict[str, list[float]] = {}
                for task in tasks:
                    rates.setdefault(task.family, []).append(self.run.base_pass_rates[task.uid])
                base = {f: _mean(v) for f, v in sorted(rates.items())}
                practice["practice"]["base"] = base
                tracked = {f: reward - base[f] for f, reward in reward_by_family.items()}
            decay = amp.practice.decay
            for source, value in tracked.items():
                self.practice.averages[source] = decay * self.practice.averages[source] + (1 - decay) * value
            if amp.practice.eta != 0:
                for task in tasks:
                    self.practice.drawn[train_cell(task)] = self.practice.drawn.get(train_cell(task), 0) + 1
        append_jsonl(self.dir / "history.jsonl", {
            "phase": "amp", "step": t, "time": now(), "version": self.version,
            "reward": _mean(r["reward"] for r in records), "return": _mean(r["return"] for r in records),
            "reward_by_family": reward_by_family,
            "completion_tokens": _mean(r["completion_tokens"] for r in records),
            "complied": _mean(r["complied"] for r in records), "truncated": _mean(r["truncated"] for r in records),
            "groups": groups, "prompts": len(tasks), **stats, "check": check, **practice, **taste, **revisit,
            "timing": {"generate_s": round(generate_s, 3), "batch_members": batch_members,
                       "score_s": round(score_s, 3), "check_s": round(check_s, 3), "update_s": stats["update_s"],
                       "publish_s": round(sum(publish.values()), 3), "publish": publish,
                       "write_s": round(time.perf_counter() - t0, 3)}})
        self.lin.step = t + 1

    def replay_sequences(self, revisited: list[Task]) -> list[Sequence]:
        """Replay imitation: for each revisited item with a stored first successful answer (recorded live, within
        length.free), that answer's own sampled ids behind the prompt it was given under: the item under the answer's
        wrapper."""
        out = []
        for task in revisited:
            item = self.items[task.uid]
            if item.answer is not None:
                asked = task if item.wrapper == task.wrapper else rewrap(task, item.wrapper)
                out.append(Sequence(self.run.prompt(asked.question), item.answer, 0.0, replay=True))
        return out

    def taste_shaping(self, seqs: list[Sequence], groups_records: list[list[dict]], sampler_lps: list[list[float]]):
        """The taste-shaped reward's factors for this step's rollouts (prompt-major, as `seqs`). Each eligible rollout
        (flagged `imitate`) gets its characteristic score: the mean over its completion tokens of log p_member -
        log p_base, the member being the weights that sampled it (this step's, before the update) and the base model
        the same network with the adapter disabled, both by the trainer on the very same prompt and completion ids.
        Returns {rollout index: score}, the factors per group (`trainer.taste_factors`) and the step's statistics,
        among them `member_vs_sampler`, the consistency check's per-token measure (summed |trainer - sampler| per
        sequence over all tokens) on every scored rollout."""
        policy, lin, taste = self.run.policy, self.lin, self.amp.taste
        t0 = time.perf_counter()
        idx = [i for i, s in enumerate(seqs) if s.imitate]
        chosen = [seqs[i] for i in idx]

        def score():
            return policy.logprobs(self.weights, chosen), policy.base_logprobs(chosen)
        mine, base = retry_on_oom(policy, f"{lin.id} step {lin.step} taste score", score) if chosen else ([], [])
        scores = [float((m - b).sum()) / len(m) for m, b in zip(mine, base)]
        tokens = sum(len(m) for m in mine)
        vs_sampler = (sum(abs(float(m.sum()) - float(torch.tensor(sampler_lps[i], dtype=torch.float32).sum()))
                          for i, m in zip(idx, mine)) / tokens) if tokens else None
        score_of = dict(zip(idx, scores))
        per_group, i = [], 0
        for recs in groups_records:
            per_group.append([score_of.get(i + k) for k in range(len(recs))])
            i += len(recs)
        fit = {}
        if taste.residualize:  # shape on what the score says beyond the answer's length
            per_group, fit = length_residuals(per_group, [[r["completion_tokens"] for r in recs]
                                                          for recs in groups_records])
        factors, stats = taste_factors(per_group, taste.strength, taste.clip)
        lengths = [len(s.completion) for s in chosen]
        corr = (statistics.correlation(scores, lengths)
                if len(scores) > 1 and len(set(scores)) > 1 and len(set(lengths)) > 1 else None)
        if stats["skipped"]:
            log.info("%s step %d: no taste shaping (%s)", lin.id, lin.step, stats["skipped"])
        return score_of, factors, {
            "eligible": len(chosen), **stats, "score_mean": statistics.fmean(scores) if scores else None,
            "score_sd": statistics.pstdev(scores) if scores else None, "length_corr": corr, **fit,
            "adv_shaping": 0.0, "adv_plain": 0.0, "member_vs_sampler": vs_sampler,
            "taste_s": round(time.perf_counter() - t0, 3)}

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
        if self.practice is not None:  # JSON floats round-trip exactly
            write_json(tmp / "practice.json", dataclasses.asdict(self.practice))
        if self.items is not None:
            write_items(tmp / ITEMS_FILE, self.items)
        if self.yoked is not None:
            write_json(tmp / YOKED_FILE, {"donor": dataclasses.asdict(self.amp.revisit.donor), "items": self.yoked})
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
