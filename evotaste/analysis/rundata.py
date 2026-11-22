"""Read-only view of a run directory for the analysis tools (docs/dev/ENGINE_NOTES.md, "On-disk layout").

Everything here tolerates a run in progress: slots whose lineage has no state yet, histories still being appended,
probes and seeding pools not yet written. Nothing under runs/<name>/eval is read (blind evaluations live there).
Commands run from the repository root, like the engine's: the frozen config refers to its files by relative path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evotaste import config
from evotaste.records import read_json, read_jsonl, read_jsonl_gz

# The probe suites, in display order: evotaste.member.PROBE_SUITES (not imported because member pulls in torch).
SUITES = ("train", "selection", "foundation")


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

    @property
    def bias(self) -> str:
        """The induced bias: the scaffold name, or the arm (plain, perturb) of a member without a scaffold."""
        return self.scaffold or self.arm

    @property
    def teacher(self) -> str:
        """The seeding pool the lineage imitated."""
        return self.scaffold or "plain"

    def history(self) -> list[dict]:
        """Amplification lines of history.jsonl, including steps after the last checkpoint."""
        return [r for r in read_jsonl(self.dir / "history.jsonl") if r["phase"] == "amp"]

    def probe_steps(self) -> list[int]:
        return sorted(int(p.name[5:11]) for p in (self.dir / "probe_responses").glob("step_*.jsonl.gz"))

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
        out = []
        for spec in self.cfg.members:
            seeding, amp, _ = self.cfg.settings(spec)
            lineage_id = spec.id
            while lineage_id is not None:
                path = self.dir / "members" / lineage_id / "state.json"
                s = read_json(path) if path.exists() else {
                    "id": spec.id, "slot": spec.id, "generation": 0, "arm": spec.arm, "scaffold": spec.scaffold,
                    "status": "pending", "step": 0, "cull_reason": None, "replaced_by": None}
                out.append(Lineage(s["id"], s["slot"], s["generation"], s["arm"], s["scaffold"], s["status"],
                                   s["step"], s["cull_reason"], s["replaced_by"], seeding.lr, amp.lr, amp.steps,
                                   path.parent))
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
