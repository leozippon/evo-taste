"""Read-only access to a Round 2 run directory, its selection evaluations and the census, for the analysis commands.

The files and their fields are the engine's (docs/dev/ENGINE_NOTES.md, "On-disk layout" and "Record schemas"). Only
committed records count: probe files up to a lineage's checkpoint step (`state.json` `step`) and amplification history
lines below it; seeding history only once the lineage has a checkpoint. Evaluations are read from the selection part
only: the vault is never loaded here. Every check that would let a format change pass silently raises instead.
Commands run from the repository root, like the engine's.
"""

from __future__ import annotations

import collections
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evotaste import config
from evotaste.records import read_json, read_jsonl, read_jsonl_gz

SUITE = "selection"  # the only evaluation part the pilots read
UNIT_KINDS = ("near", "far", "procedural")  # held-out items of training sources, unseen units, format-probe families
FIELDS = ("reward", "correct_any", "complied", "truncated", "tokens", "program")


@dataclass(frozen=True)
class Contexts:
    """The frozen contexts file: each context's level (placebo, stance, tactic, reference) and theme, and the jargon
    list."""
    level: dict[str, str]
    theme: dict[str, str | None]
    jargon: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> Contexts:
        data = yaml.safe_load(Path(path).read_text())
        level, theme = {"placebo": "placebo"}, {"placebo": None}
        for section, name in (("stances", "stance"), ("tactics", "tactic")):
            for c, v in data[section].items():
                theme[c] = v["theme"]
                level[c] = "reference" if v["theme"] == "reference" else name
        return cls(level, theme, tuple(data["jargon"]))

    @property
    def names(self) -> list[str]:
        return sorted(self.level)


@dataclass(frozen=True)
class Lineage:
    id: str
    context: str | None
    level: str  # placebo | stance | tactic | reference | unseeded
    theme: str | None
    seeding: str  # opcd, opcd[math], imitation, none
    arm: str  # the update rule: advantage, plus +si<beta> with self-imitation
    parent: tuple[str, int] | None  # (parent lineage, clone step) of a clone
    steps: int  # amplification steps configured
    status: str  # pending (no state yet) | new | active | done | culled
    step: int  # amplification updates at the last checkpoint
    dir: Path

    @property
    def control(self) -> bool:
        """The washout controls: the placebo context and the unseeded member."""
        return self.level in ("placebo", "unseeded")


def rule_name(amp: config.Amplification) -> str:
    return amp.advantage + (f"+si{amp.self_imitation:g}" if amp.self_imitation > 0 else "")


class Run:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.cfg = config.load(self.dir / "config.yaml")
        self.name = self.cfg.name
        self.contexts = Contexts.load(self.cfg.contexts)
        self.units = unit_kinds(self.cfg.suites)
        self.lineages = self._lineages()

    def _lineages(self) -> dict[str, Lineage]:
        out = {}
        for spec in self.cfg.members:
            seeding, amp, _ = self.cfg.settings(spec)
            context = self.cfg.context_of(spec)
            level = "unseeded" if context is None else self.contexts.level[context]
            seed = seeding.method + (f"[{','.join(seeding.domains)}]" if seeding.domains and seeding.method != "none"
                                     else "")
            lineage_id = spec.id
            while lineage_id is not None:  # the slot's lineage and its replacements after culls
                path = self.dir / "members" / lineage_id / "state.json"
                s = read_json(path) if path.exists() else {"status": "pending", "step": 0, "replaced_by": None}
                parent = (spec.parent.member, spec.parent.step) if spec.parent else None
                out[lineage_id] = Lineage(lineage_id, context, level, self.contexts.theme.get(context), seed,
                                          rule_name(amp), parent, amp.steps, s["status"], s["step"], path.parent)
                lineage_id = s["replaced_by"]
        return out

    def root(self, lin: Lineage) -> Lineage:
        """The seeded member a clone (or a chain of clones) continues; a member that is not a clone is its own."""
        while lin.parent:
            lin = self.lineages[lin.parent[0]]
        return lin

    @staticmethod
    def restart(lin: Lineage, step: int) -> bool:
        """Whether a probe at `step` is a clone's first, of the checkpoint it copied: a second sample of its parent's
        state, used for the noise of one probe and otherwise left out so that no state is counted twice."""
        return lin.parent is not None and step == lin.parent[1]

    def committed(self, lin: Lineage) -> bool:
        return lin.status not in ("pending", "new")

    def probe_steps(self, lin: Lineage) -> list[int]:
        if not self.committed(lin):
            return []
        steps = (int(p.name[5:11]) for p in (lin.dir / "probe_responses").glob("step_*.jsonl.gz"))
        return sorted(s for s in steps if s <= lin.step)

    def probe_records(self, lin: Lineage, step: int) -> list[dict]:
        return read_jsonl_gz(lin.dir / "probe_responses" / f"step_{step:06d}.jsonl.gz")

    def probe_summaries(self, lin: Lineage) -> dict[int, dict]:
        steps = set(self.probe_steps(lin))
        return {r["step"]: r for r in read_jsonl(lin.dir / "probes.jsonl") if r["step"] in steps}

    def base_records(self, context: str | None = None) -> list[dict]:
        """The base model's plain probe answers, or its answers under a context."""
        path = self.dir / "base" / ("probe_responses.jsonl.gz" if context is None else f"contexts/{context}.jsonl.gz")
        return read_jsonl_gz(path)

    def history(self, lin: Lineage) -> tuple[list[dict], list[dict]]:
        """Committed (seeding, amplification) lines of history.jsonl, in step order."""
        if not self.committed(lin):
            return [], []
        lines = read_jsonl(lin.dir / "history.jsonl")
        seed = sorted((r for r in lines if r["phase"] == "seed"), key=lambda r: r["step"])
        amp = sorted((r for r in lines if r["phase"] == "amp" and r["step"] < lin.step), key=lambda r: r["step"])
        if len({r["step"] for r in amp}) != len(amp) or len({r["step"] for r in seed}) != len(seed):
            raise ValueError(f"{lin.dir}/history.jsonl: a step appears twice")
        return seed, amp

    def resolve(self, label: str) -> tuple[Lineage | None, int | None]:
        """(lineage, step) of an evaluation label: base, <lineage> (its last checkpoint) or <lineage>@<step>."""
        if label == "base":
            return None, None
        lineage_id, _, step = label.partition("@")
        if lineage_id not in self.lineages:
            raise ValueError(f"label {label!r}: no lineage {lineage_id!r} in run {self.name}")
        lin = self.lineages[lineage_id]
        return lin, int(step) if step else lin.step


def unit_kinds(suites_path: str | Path) -> dict[str, str]:
    """Selection family -> near (a training source's held-out items), far (an unseen unit) or procedural (a
    reasoning-gym format-probe family), from the suites file."""
    data = yaml.safe_load(Path(suites_path).read_text())
    train = set(data["suites"]["train"])
    out = {f: "near" if f in train else "far" for f in data["suites"]["selection"]}
    for f in data["units"]:
        if f in out and f in train:
            raise ValueError(f"{suites_path}: {f} is both a training source and an unseen unit")
    out.update({f: "procedural" for f in data["procedural"]["selection"]})
    return out


# -- response grids ------------------------------------------------------------------------------------------------

@dataclass
class Grid:
    """Responses of several subjects (adapters, or lineage probes) to the same items, as arrays subject x item x
    sample. `items` holds uid, family, domain, kind, wrapper, unit (near/far/procedural) and stratum (from the census,
    when given). `x` maps a field (FIELDS, and response traits when computed) to a float array; `program` is NaN outside
    kind open or when the response has no channel. `texts` holds sample 0's text per subject and item when kept."""
    labels: list[str]
    items: pd.DataFrame
    x: dict[str, np.ndarray]
    texts: np.ndarray | None = None
    sha: dict[str, str | None] | None = None
    aliases: dict[str, str] | None = None  # evaluation labels dropped because another label is the same adapter

    @property
    def samples(self) -> int:
        return self.x["reward"].shape[2]

    def take(self, labels: list[str]) -> Grid:
        idx = [self.labels.index(a) for a in labels]
        return Grid(list(labels), self.items, {k: v[idx] for k, v in self.x.items()},
                    None if self.texts is None else self.texts[idx], self.sha, self.aliases)


def record_values(r: dict) -> dict[str, float]:
    channel = r["channel"]
    return {"reward": float(r["reward"]), "correct_any": float(r["correct_any"]), "complied": float(r["complied"]),
            "truncated": float(r["truncated"]), "tokens": float(r["completion_tokens"]),
            "program": float(channel == "program") if r["kind"] == "open" and channel is not None else np.nan}


def build_grid(groups: dict[str, list[dict]], samples: int, units: dict[str, str], traits=None,
               keep_texts: bool = False) -> Grid:
    """A grid from records grouped by subject; every subject must answer the same items (same uid and question) with
    samples 0..samples-1 exactly once each (records without `sample` are sample 0). `traits`, a function of a record
    returning a dict of floats, adds response traits."""
    first = next(iter(groups.values()))
    meta = {}
    for r in first:
        meta.setdefault(r["uid"], (r["family"], r["domain"], r["kind"], r["wrapper"],
                                   hashlib.sha1(r["question"].encode()).hexdigest()))
    uids = sorted(meta, key=lambda u: (meta[u][0], u))
    unknown = sorted({meta[u][0] for u in uids} - set(units))
    if unknown:
        raise ValueError(f"families not in the selection part of the suites file: {unknown}")
    pos = {u: i for i, u in enumerate(uids)}
    labels = list(groups)
    x = collections.defaultdict(lambda: np.full((len(labels), len(uids), samples), np.nan))
    seen = np.zeros((len(labels), len(uids), samples), dtype=int)
    texts = np.full((len(labels), len(uids)), None, dtype=object) if keep_texts else None
    for s, label in enumerate(labels):
        for r in groups[label]:
            i = pos.get(r["uid"])
            m = (r["family"], r["domain"], r["kind"], r["wrapper"], hashlib.sha1(r["question"].encode()).hexdigest())
            if i is None or meta[r["uid"]] != m:
                raise ValueError(f"{label}: item {r['uid']} is not the same item for every subject")
            k = r.get("sample", 0)
            seen[s, i, k] += 1
            values = record_values(r) | (traits(r) if traits else {})
            for name, v in values.items():
                x[name][s, i, k] = v
            if keep_texts and k == 0:
                texts[s, i] = r["text"]
    if (seen != 1).any():
        bad = [labels[s] for s in sorted(set(np.argwhere(seen != 1)[:, 0]))]
        raise ValueError(f"incomplete or duplicated responses (every item x {samples} samples once) for {bad}")
    items = pd.DataFrame({"uid": uids, "family": [meta[u][0] for u in uids], "domain": [meta[u][1] for u in uids],
                          "kind": [meta[u][2] for u in uids], "wrapper": [meta[u][3] for u in uids],
                          "unit": [units[meta[u][0]] for u in uids]})
    return Grid(labels, items, dict(x), texts)


def probe_grid(run: Run, subjects: dict[str, tuple[Lineage | None, int | None]], traits=None) -> Grid:
    """Probe answers of subjects label -> (lineage, step), or (None, None) for the base model's plain answers; checks
    that the per-family accuracy reproduces each probe summary the engine wrote."""
    groups = {}
    for label, (lin, step) in subjects.items():
        groups[label] = run.base_records() if lin is None else run.probe_records(lin, step)
        summary = (read_json(run.dir / "base" / "probe_summary.json") if lin is None
                   else run.probe_summaries(lin)[step])
        check_families(groups[label], summary["by_family"], f"{label} probe")
    return build_grid(groups, 1, run.units, traits)


def check_families(records: list[dict], by_family: dict[str, float], what: str) -> None:
    fams = collections.defaultdict(list)
    for r in records:
        fams[r["family"]].append(r["reward"])
    mine = {f: sum(v) / len(v) for f, v in fams.items()}
    if set(mine) != set(by_family) or any(abs(mine[f] - by_family[f]) > 1e-9 for f in mine):
        raise ValueError(f"{what}: per-family accuracy differs from the engine's summary (a record format change?)")


def load_eval(run: Run, tag: str, traits=None, keep_texts: bool = False) -> Grid:
    """A complete selection evaluation runs/<name>/eval/selection/<tag>/. Labels that name the same adapter (equal
    content hash) are the same model (a clone at its clone point is its parent's checkpoint): one is kept, a label of
    the original lineage before a clone's, and the others are listed in `aliases`."""
    out = run.dir / "eval" / SUITE / tag
    params = read_json(out / "params.json")
    summary = read_json(out / "summary.json")["adapters"]  # written only once the grid is complete
    records = [r for p in sorted(out.glob("records-*.jsonl.gz")) for r in read_jsonl_gz(p)]
    groups = collections.defaultdict(list)
    for r in records:
        if r["suite"] != SUITE:
            raise ValueError(f"{out}: a record of suite {r['suite']!r}")
        groups[r["adapter"]].append(r)
    if set(groups) != set(summary):
        raise ValueError(f"{out}: the records' labels differ from summary.json's")
    sha, aliases, kept = {}, {}, {}
    for label in sorted(groups, key=lambda a: (a != "base" and run.resolve(a)[0].parent is not None, a)):
        hashes = {r["adapter_sha256"] for r in groups[label]}
        if len(hashes) != 1:
            raise ValueError(f"{out}: {label} has records of {len(hashes)} adapters")
        h = hashes.pop()
        check_families(groups[label], summary[label]["by_family"], f"{out} {label}")
        if h is not None and h in kept:
            aliases[label] = kept[h]
            continue
        kept[h] = label
        sha[label] = h
    grid = build_grid({a: groups[a] for a in sha}, params["samples"], run.units, traits, keep_texts)
    grid.sha, grid.aliases = sha, aliases
    return grid


# -- census and context score ---------------------------------------------------------------------------------------

def census_strata(census_dir: str | Path) -> dict[str, str]:
    """uid -> the base model's stratum on a selection item from the census's selection part: frontier (no sample
    correct), mastered (every sample correct), sweet otherwise."""
    passes, samples = collections.Counter(), collections.Counter()
    paths = sorted(Path(census_dir, "selection").glob("chunk-*.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"{census_dir}/selection: no census chunks")
    for p in paths:
        for r in read_jsonl_gz(p):
            if r["part"] != SUITE or r["context"] is not None or r["variant"] != "default":
                raise ValueError(f"{p}: not a default selection record")
            passes[r["uid"]] += r["reward"]
            samples[r["uid"]] += 1
    return {u: "frontier" if passes[u] == 0 else "mastered" if passes[u] == n else "sweet" for u, n in samples.items()}


def add_strata(grid: Grid, strata: dict[str, str] | None) -> None:
    grid.items["stratum"] = [strata.get(u) for u in grid.items["uid"]] if strata else None


def context_scores(run: Run, table: Path | None = None) -> pd.DataFrame:
    """A context-score table of the engine (runs/<name>/context_score/<selection>/table.csv; `table`, or the run's
    only selection): subject, lineage, step, context, domain, responses, tokens, delta_sum, delta_mean, one row per
    subject, context and domain."""
    if table is None:
        found = sorted((run.dir / "context_score").glob("*/table.csv"))
        if len(found) != 1:
            raise FileNotFoundError(f"{run.dir}/context_score: {len(found)} selections with a table "
                                    f"({', '.join(p.parent.name for p in found) or 'none'}); name one with --table")
        table = found[0]
    t = pd.read_csv(table, dtype={"subject": str, "lineage": str, "context": str, "domain": str})
    expected = ["subject", "lineage", "step", "context", "domain", "responses", "tokens", "delta_sum", "delta_mean"]
    if list(t.columns) != expected:
        raise ValueError(f"{table}: columns {list(t.columns)}, expected {expected}")
    if not np.allclose(t["delta_sum"] / t["tokens"], t["delta_mean"]):
        raise ValueError(f"{table}: delta_mean is not delta_sum / tokens")
    return t
