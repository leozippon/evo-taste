"""Read-only view of a run directory for the analysis tools (docs/dev/ENGINE_NOTES.md, "On-disk layout").

Everything here tolerates a run in progress: slots whose lineage has no state yet, histories still being appended,
probes and seeding pools not yet written. Records after a lineage's last checkpoint are uncommitted (a resume
deletes and redoes them): amplification history after it is kept only while the lineage is active and its worker
runs, probe files after it are never used. Whether a worker runs is read from /proc/locks (the worker holds an
flock on runs/<name>/locks/<slot>.lock); no lock is ever taken. Final evaluations are read by `load_evaluation`,
the only reader of runs/<name>/eval: it opens a blind vault only when the caller passes blind=True (the tools' explicit
--blind flag) and the run's harvest manifest exists. Commands run from the repository root, like the engine's: the
frozen config and the harvest manifest are found by relative path.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from evotaste import config
from evotaste.records import read_json, read_jsonl, read_jsonl_gz

# The probe suites, in display order: evotaste.member.PROBE_SUITES (not imported because member pulls in torch).
SUITES = ("train", "selection", "foundation")
VAULTS = ("blind1", "blind2")  # evotaste.evaluate.VAULTS
PROCEDURAL_UID = re.compile(r"^[^/]+/\d+/\d+$")  # <family>/<dataset seed>/<index>; external uids are <source>/<id>


def manifest_path(run_name: str) -> Path:
    """The harvest manifest, as evotaste.evaluate.manifest_path (not imported: evaluate pulls in torch)."""
    return Path("docs") / "harvest" / f"{run_name}.json"


def member_specs(run_dir: str | Path) -> tuple[str, dict[str, dict]]:
    """Run name and the configured slots of the frozen config: id -> {arm, scaffold, role}. Read without the
    engine's schema so that the optional member field `role` works for configs with and without it; `role: baseline`
    marks a single-model baseline (a comparator), a missing role a population member."""
    data = yaml.safe_load((Path(run_dir) / "config.yaml").read_text())
    specs = {}
    for m in data["members"]:
        role = m.get("role", "member")
        if role not in ("member", "baseline"):
            raise ValueError(f"member {m['id']}: unknown role {role!r}")
        specs[m["id"]] = {"arm": m["arm"], "scaffold": m.get("scaffold"), "role": role}
    return data["name"], specs


def held_locks() -> set[tuple[int, int, int]]:
    """(device major, device minor, inode) of every file currently locked on this host."""
    held = set()
    for line in Path("/proc/locks").read_text().splitlines():
        fields = [f for f in line.split() if f != "->"]  # "->" marks a waiter blocked on the lock above
        major, minor, inode = fields[5].split(":")  # e.g. "1: FLOCK  ADVISORY  WRITE 3071670 08:02:54788294 0 EOF"
        held.add((int(major, 16), int(minor, 16), int(inode)))
    return held


@dataclass(frozen=True)
class Lineage:
    id: str
    slot: str
    generation: int
    arm: str
    scaffold: str | None
    status: str  # new | active | done | culled from state.json; "pending" while the slot has no state yet
    checkpoint: int  # updates done at the last checkpoint
    cull_reason: str | None
    replaced_by: str | None
    seeding_lr: float
    amp_lr: float
    amp_steps: int
    dir: Path
    running: bool  # a worker holds the slot's lock and this is the slot's current lineage (status new or active)

    @property
    def bias(self) -> str:
        """The induced bias: the scaffold name, or the arm (plain, perturb) of a member without a scaffold."""
        return self.scaffold or self.arm

    @property
    def teacher(self) -> str:
        """The seeding pool the lineage imitated."""
        return self.scaffold or "plain"

    def history(self) -> list[dict]:
        """Amplification lines of history.jsonl in step order, one per step (a later line replaces an earlier one).
        Steps at or after the last checkpoint are included only while the lineage is active and its worker runs."""
        by_step = {r["step"]: r for r in read_jsonl(self.dir / "history.jsonl") if r["phase"] == "amp"}
        in_progress = self.status == "active" and self.running
        return [by_step[t] for t in sorted(by_step) if t < self.checkpoint or in_progress]

    def probe_steps(self) -> list[int]:
        """Committed probe steps: at most the last checkpoint, and none before the first checkpoint exists."""
        if self.status in ("new", "pending"):
            return []
        steps = (int(p.name[5:11]) for p in (self.dir / "probe_responses").glob("step_*.jsonl.gz"))
        return sorted(s for s in steps if s <= self.checkpoint)

    def probe_responses(self, step: int) -> list[dict]:
        return read_jsonl_gz(self.dir / "probe_responses" / f"step_{step:06d}.jsonl.gz")


class RunData:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.cfg = config.load(self.dir / "config.yaml")
        self.name = self.cfg.name
        self.lineages = self._lineages()

    def _lineages(self) -> list[Lineage]:
        """Every lineage of every configured slot, in config order, replacements after the lineage they replace."""
        held = held_locks()
        out = []
        for spec in self.cfg.members:
            seeding, amp, _ = self.cfg.settings(spec)
            lock = self.dir / "locks" / f"{spec.id}.lock"
            locked = lock.exists() and (os.major((st := lock.stat()).st_dev), os.minor(st.st_dev), st.st_ino) in held
            lineage_id = spec.id
            while lineage_id is not None:
                path = self.dir / "members" / lineage_id / "state.json"
                s = read_json(path) if path.exists() else {
                    "id": spec.id, "slot": spec.id, "generation": 0, "arm": spec.arm, "scaffold": spec.scaffold,
                    "status": "pending", "step": 0, "cull_reason": None, "replaced_by": None}
                out.append(Lineage(s["id"], s["slot"], s["generation"], s["arm"], s["scaffold"], s["status"],
                                   s["step"], s["cull_reason"], s["replaced_by"], seeding.lr, amp.lr, amp.steps,
                                   path.parent, locked and s["status"] in ("new", "active")))
                lineage_id = s["replaced_by"]
        return out

    def base_summary(self) -> dict | None:
        path = self.dir / "base" / "probe_summary.json"  # written after the responses
        return read_json(path) if path.exists() else None

    def base_responses(self) -> list[dict] | None:
        if self.base_summary() is None:
            return None
        return read_jsonl_gz(self.dir / "base" / "probe_responses.jsonl.gz")

    def teachers(self) -> list[str]:
        """Teachers whose seeding pool is complete (its summary is written after the samples)."""
        return sorted(p.stem for p in (self.dir / "seeding").glob("*.json"))

    def pool_summary(self, teacher: str) -> dict:
        return read_json(self.dir / "seeding" / f"{teacher}.json")

    def pool_records(self, teacher: str) -> list[dict]:
        return read_jsonl_gz(self.dir / "seeding" / f"{teacher}.jsonl.gz")


@dataclass(frozen=True)
class Evaluation:
    """One final evaluation (runs/<name>/eval/<suite>/<tag>/) as arrays over adapters, instances and samples."""
    suite: str
    tag: str
    per_family: int
    samples: int
    adapters: tuple[str, ...]
    uids: tuple[str, ...]  # instances, ordered by family, then uid
    families: np.ndarray  # family or external source of each instance
    procedural: np.ndarray  # per instance: a procedural family (True) or an external source (False)
    reward: np.ndarray  # adapter x instance x sample, 0 or 1
    tokens: np.ndarray  # completion tokens, same shape
    dropped: tuple[str, ...]  # uids whose question text differs between adapters, left out


def load_evaluation(run_dir: str | Path, suite: str, tag: str, blind: bool = False) -> Evaluation:
    """Every record file of the evaluation. A blind vault is read only with blind=True and an existing harvest
    manifest. Instances whose question text differs between adapters are dropped; an adapter missing any record of
    the remaining instances and samples, or a duplicate record, is an error."""
    run_dir = Path(run_dir)
    if suite in VAULTS:
        name, _ = member_specs(run_dir)
        if not blind:
            raise PermissionError(f"{suite} is a blind vault: it is read only when --blind {suite} is given")
        if not manifest_path(name).exists():
            raise PermissionError(f"refusing to read {suite}: no harvest manifest {manifest_path(name)}")
    out = run_dir / "eval" / suite / tag
    params = read_json(out / "params.json")
    records = [r for path in sorted(out.glob("records-*.jsonl.gz")) for r in read_jsonl_gz(path)]
    if not records:
        raise FileNotFoundError(f"{out}: no record files")
    keys = [(r["adapter"], r["uid"], r["sample"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{out}: duplicate (adapter, uid, sample) records")
    questions: dict[str, set[str]] = {}
    for r in records:
        questions.setdefault(r["uid"], set()).add(hashlib.sha1(r["question"].encode()).hexdigest())
    dropped = sorted(u for u, q in questions.items() if len(q) > 1)
    family = {r["uid"]: r["family"] for r in records}
    uids = sorted((u for u in questions if u not in set(dropped)), key=lambda u: (family[u], u))
    adapters = sorted({r["adapter"] for r in records})
    a_index, u_index = {a: i for i, a in enumerate(adapters)}, {u: i for i, u in enumerate(uids)}
    s = params["samples"]
    reward = np.full((len(adapters), len(uids), s), np.nan)
    tokens = np.full_like(reward, np.nan)
    for r in records:
        if r["uid"] in u_index:
            reward[a_index[r["adapter"]], u_index[r["uid"]], r["sample"]] = r["reward"]
            tokens[a_index[r["adapter"]], u_index[r["uid"]], r["sample"]] = r["completion_tokens"]
    missing = np.isnan(reward).any(axis=(1, 2))
    if missing.any():
        raise ValueError(f"{out}: incomplete evaluation; adapters without every instance x {s} samples: "
                         f"{[a for a, m in zip(adapters, missing) if m]}")
    ev = Evaluation(suite, tag, params["per_family"], s, tuple(adapters), tuple(uids),
                    np.array([family[u] for u in uids]), np.array([bool(PROCEDURAL_UID.match(u)) for u in uids]),
                    reward, tokens, tuple(dropped))
    if not dropped:  # the procedural/external split must reproduce the engine's macro accuracy
        engine = read_json(out / "summary.json")["adapters"]
        fams = sorted(set(ev.families[ev.procedural]))
        for i, a in enumerate(adapters):
            macro = np.mean([reward[i, ev.families == f].mean() for f in fams])
            if abs(macro - engine[a]["macro"]) > 1e-9:
                raise ValueError(f"{out}: macro of {a} is {macro}, the engine's summary says {engine[a]['macro']}")
    return ev
