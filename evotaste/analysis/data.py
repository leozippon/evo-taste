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
    arm: str  # the update rule (rule_name)
    parent: tuple[str, int] | None  # (parent lineage, clone step) of a clone
    steps: int  # amplification steps configured
    status: str  # pending (no state yet) | new | active | done | culled
    step: int  # amplification updates at the last checkpoint
    dir: Path
    taste: float | None = None  # strength of the taste-shaped reward when the member sets one (pilot E), else None

    @property
    def control(self) -> bool:
        """The washout controls: the placebo context and the unseeded member."""
        return self.level in ("placebo", "unseeded")


def taste_of(amp: config.Amplification):
    """The taste-shaped reward's settings (WP-E3), or None; read by name because older engines have no such key."""
    return getattr(amp, "taste", None)


def rule_name(amp: config.Amplification) -> str:
    """The update rule, naming every term the update actually computes: the advantage, then +si<beta>
    (self-imitation), +ex<gamma>x<weight> (exaggeration), ~free when the imitation terms take only successes within
    the free length, +pr<eta> (self-chosen practice; +prb<eta> relative to the base model's pass rate) and
    +taste<strength>. A term that the engine skips (weight, gamma, eta or strength 0) is not named, so two members
    with the same name receive the same update."""
    name = amp.advantage
    imitation = False
    if amp.self_imitation > 0:
        name += f"+si{amp.self_imitation:g}"
        imitation = True
    ex = amp.exaggeration
    if ex is not None and ex.gamma > 0 and ex.weight > 0:
        name += f"+ex{ex.gamma:g}x{ex.weight:g}"
        imitation = True
    if imitation and amp.imitate_within_free:
        name += "~free"
    if amp.practice is not None and amp.practice.eta > 0:
        name += f"+pr{'b' if getattr(amp.practice, 'reference', None) == 'base' else ''}{amp.practice.eta:g}"
    taste = taste_of(amp)
    if taste is not None and taste.strength != 0:
        name += f"+taste{taste.strength:g}"
    return name


class Run:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.cfg = config.load(self.dir / "config.yaml")
        self.name = self.cfg.name
        self.contexts = Contexts.load(self.cfg.contexts)
        self.units = unit_kinds(self.cfg.suites)
        self.lineages = self._lineages()
        # finished scaled probes (the engine writes probe_summary.json last): label -> summary
        self.scaled = {p.parent.name: read_json(p) for p in sorted((self.dir / "scaled").glob("*/probe_summary.json"))}
        for label, s in self.scaled.items():
            if label != s["label"] or s["lineage"] not in self.lineages:
                raise ValueError(f"{self.dir}/scaled/{label}: summary of {s['label']} (lineage {s['lineage']})")

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
                taste = taste_of(amp)
                out[lineage_id] = Lineage(lineage_id, context, level, self.contexts.theme.get(context), seed,
                                          rule_name(amp), parent, amp.steps, s["status"], s["step"], path.parent,
                                          None if taste is None else float(taste.strength))
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

    def state(self, lin: Lineage, step: int) -> tuple[str, int]:
        """The checkpoint (lineage, step) a probe at `step` samples: a clone's first probe samples its parent's."""
        while self.restart(lin, step):
            lin = self.lineages[lin.parent[0]]
        return lin.id, step

    def base_weights(self, lin: Lineage | None, step: int | None) -> bool:
        """Whether a probe (or evaluated label) samples the base model's weights: the base model itself, or an
        unseeded member (its adapter starts with B = 0) before any update, its clones' first probes included."""
        if lin is None:
            return True
        lin_id, at = self.state(lin, step)
        return at == 0 and self.lineages[lin_id].seeding == "none"

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

    def items_where(self, keep) -> Grid:
        """The same subjects on the items where `keep` (a mask over items) holds."""
        keep = np.asarray(keep, bool)
        return Grid(self.labels, self.items[keep].reset_index(drop=True), {k: v[:, keep] for k, v in self.x.items()},
                    None if self.texts is None else self.texts[:, keep], self.sha, self.aliases)

    def real(self) -> Grid:
        """The dataset units: every unit but the procedural format probe."""
        return self.items_where(self.items["unit"] != "procedural")

    def procedural(self) -> Grid:
        """The procedural format-probe units (reasoning-gym families, strict scorers)."""
        return self.items_where(self.items["unit"] == "procedural")


def strata(items: pd.DataFrame) -> np.ndarray:
    """The bootstrap's strata: unit kind (near, far, procedural) within domain. Units themselves are too small in the
    probe set, where every procedural family has a single instance: resampling within a one-item unit has no variance,
    and intervals would ignore the noise of those draws."""
    return (items["unit"] + "/" + items["domain"]).to_numpy()


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


def probe_grid(run: Run, subjects: dict[str, tuple[Lineage | None, int | None]], traits=None,
               keep_texts: bool = False) -> Grid:
    """Probe answers of subjects label -> (lineage, step), or (None, None) for the base model's plain answers, and
    scaled probes (label <lineage>@<step>x<factor>, given as (lineage, step) like the probe they scale); checks that
    the per-family accuracy reproduces each probe summary the engine wrote."""
    groups = {}
    for label, (lin, step) in subjects.items():
        if lin is None:
            groups[label] = run.base_records()
            summary = read_json(run.dir / "base" / "probe_summary.json")
        elif label in run.scaled:
            groups[label] = read_jsonl_gz(run.dir / "scaled" / label / "probe_responses.jsonl.gz")
            summary = run.scaled[label]
        else:
            groups[label] = run.probe_records(lin, step)
            summary = run.probe_summaries(lin)[step]
        check_families(groups[label], summary["by_family"], f"{label} probe")
    return build_grid(groups, 1, run.units, traits, keep_texts)


def check_families(records: list[dict], by_family: dict[str, float], what: str) -> None:
    fams = collections.defaultdict(list)
    for r in records:
        fams[r["family"]].append(r["reward"])
    mine = {f: sum(v) / len(v) for f, v in fams.items()}
    if set(mine) != set(by_family) or any(abs(mine[f] - by_family[f]) > 1e-9 for f in mine):
        raise ValueError(f"{what}: per-family accuracy differs from the engine's summary (a record format change?)")


def load_eval(run: Run, tag: str, traits=None, keep_texts: bool = False) -> Grid:
    """A complete selection evaluation runs/<name>/eval/selection/<tag>/, or several (tags separated by commas) of
    the same items and samples, merged. Labels that name the same adapter (equal content hash) are the same model (a
    clone at its clone point is its parent's checkpoint): one is kept, a label of the original lineage before a
    clone's, and the others are listed in `aliases`."""
    groups, samples = collections.defaultdict(list), set()
    for t in tag.split(","):
        out = run.dir / "eval" / SUITE / t
        if not (out / "params.json").exists():
            raise SystemExit(f"{out}: no evaluation under this tag")
        if not (out / "summary.json").exists():  # the engine writes it only once every shard has finished
            raise SystemExit(f"{out}: the evaluation is incomplete (no summary.json yet)")
        samples.add(read_json(out / "params.json")["samples"])
        summary = read_json(out / "summary.json")["adapters"]
        mine = collections.defaultdict(list)
        for p in sorted(out.glob("records-*.jsonl.gz")):
            for r in read_jsonl_gz(p):
                if r["suite"] != SUITE:
                    raise ValueError(f"{out}: a record of suite {r['suite']!r}")
                mine[r["adapter"]].append(r)
        if set(mine) != set(summary):
            raise ValueError(f"{out}: the records' labels differ from summary.json's")
        for label, recs in mine.items():
            check_families(recs, summary[label]["by_family"], f"{out} {label}")
            if label in groups:  # in two tags: must be the same adapter, whose answers are then the same draws
                if {r["adapter_sha256"] for r in recs} != {r["adapter_sha256"] for r in groups[label]}:
                    raise ValueError(f"{label} is a different adapter in two of the tags {tag}")
                continue
            groups[label] = recs
    if len(samples) != 1:
        raise SystemExit(f"tags {tag} have different numbers of samples per item: {sorted(samples)}")
    sha, aliases, kept = {}, {}, {}
    for label in sorted(groups, key=lambda a: (a != "base" and run.resolve(a)[0].parent is not None, a)):
        hashes = {r["adapter_sha256"] for r in groups[label]}
        if len(hashes) != 1:
            raise ValueError(f"{label} has records of {len(hashes)} adapters")
        h = hashes.pop()
        if h is not None and h in kept:
            aliases[label] = kept[h]
            continue
        kept[h] = label
        sha[label] = h
    grid = build_grid({a: groups[a] for a in sha}, samples.pop(), run.units, traits, keep_texts)
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


# -- deviation and cross-likelihood (evaluate deviation [--cross]) --------------------------------------------------

DEVIATION_COLUMNS = ["subject", "lineage", "step", "factor", "uid", "domain", "tokens", "delta_sum"]
CROSS_COLUMNS = ["scorer", "author", "step", "uid", "domain", "tokens", "logprob_sum"]


def deviation_selection(run: Run, selection: str | Path | None = None) -> Path:
    """A selection directory of `evaluate deviation` (runs/<name>/deviation/<N>items-<hash>/): the one named, or the
    run's only one."""
    if selection is not None:
        return Path(selection)
    found = sorted(p.parent for p in (run.dir / "deviation").glob("*/selection.json"))
    if len(found) != 1:
        raise SystemExit(f"{run.dir}/deviation: {len(found)} selections ({', '.join(p.name for p in found) or 'none'}); "
                         "run python -m evotaste.evaluate deviation, or name one with --selection")
    return found[0]


def author_file(run: Run, label: str) -> Path:
    """The probe answers that an author label of the likelihood tables stands for: a committed member probe
    <lineage>@<step> or a finished scaled probe <lineage>@<step>x<factor>."""
    if label in run.scaled:
        return run.dir / "scaled" / label / "probe_responses.jsonl.gz"
    lin, step = run.resolve(label)
    if step not in run.probe_steps(lin):
        raise ValueError(f"{label}: not a committed probe of run {run.name}")
    return lin.dir / "probe_responses" / f"step_{step:06d}.jsonl.gz"


def _items_of(run: Run, sel: Path, df: pd.DataFrame, who: str, what: str) -> pd.DataFrame:
    """Checks that every `who` has every item of selection.json exactly once, in the base model's domains, and adds
    each item's family and unit kind."""
    uids = read_json(sel / "selection.json")["uids"]
    meta = {r["uid"]: (r["family"], r["domain"]) for r in run.base_records()}
    counts = df.groupby(who)["uid"].agg(["size", "nunique"])
    if set(df["uid"]) != set(uids) or (counts["size"] != len(uids)).any() or (counts["nunique"] != len(uids)).any():
        raise ValueError(f"{sel}/{what}: not every {who} has every item of the selection exactly once")
    if any(meta[u][1] != d for u, d in zip(df["uid"], df["domain"])):
        raise ValueError(f"{sel}/{what}: an item's domain differs from the base model's probe answers")
    df["family"] = [meta[u][0] for u in df["uid"]]
    df["unit"] = [run.units[f] for f in df["family"]]
    return df


def deviation_table(run: Run, selection: str | Path | None = None, partial: bool = False) -> tuple[pd.DataFrame, Path]:
    """Per-response rows of a deviation selection (columns DEVIATION_COLUMNS plus family and unit): its responses.csv,
    which the engine writes when the command finishes, or with `partial` its per-author cache files (<label>.json), of
    a selection still being scored. A cache counts only if it was computed on the author's current committed answers
    (the SHA-256 of the answers file, the key the engine itself checks besides the adapter's hash: a checkpoint and the
    probe it answered are committed together, so new answers come with a new file)."""
    sel = deviation_selection(run, selection)
    if not partial:
        if not (sel / "responses.csv").exists():
            raise SystemExit(f"{sel}: no responses.csv (the deviation command has not finished); --partial reads the "
                             "per-author results finished so far")
        df = pd.read_csv(sel / "responses.csv", dtype={"subject": str, "lineage": str, "uid": str, "domain": str})
        if list(df.columns) != DEVIATION_COLUMNS:
            raise ValueError(f"{sel}/responses.csv: columns {list(df.columns)}, expected {DEVIATION_COLUMNS}")
    else:
        rows, stale = [], []
        for cache in sorted(sel.glob("*@*.json")):
            c = read_json(cache)
            label = cache.name[:-len(".json")]
            try:
                current = hashlib.sha256(author_file(run, label).read_bytes()).hexdigest() == c["sha256"]
            except ValueError:
                current = False
            if not current:
                stale.append(label)
                continue
            if {r["subject"] for r in c["responses"]} != {label}:
                raise ValueError(f"{cache}: rows of another subject")
            rows += c["responses"]
        if stale:
            print(f"{sel}: left out {len(stale)} cache files of answers no longer committed: {', '.join(stale)}")
        if not rows:
            raise SystemExit(f"{sel}: no finished per-author results yet")
        df = pd.DataFrame(rows)
        if list(df.columns) != DEVIATION_COLUMNS:
            raise ValueError(f"{sel}: cache columns {list(df.columns)}, expected {DEVIATION_COLUMNS}")
    df["factor"] = pd.to_numeric(df["factor"])
    return _items_of(run, sel, df, "subject", "responses"), sel


def cross_table(run: Run, step: int, selection: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    """cross-<step>.csv of a deviation selection: the summed log-probability of every author's answer to every item
    under every scorer (base: the base model, as author its plain probe answers). Checks that every scorer has every
    author's answer to every item once, with the author's completion tokens the same under every scorer."""
    sel = deviation_selection(run, selection)
    path = sel / f"cross-{step}.csv"
    if not path.exists():
        raise SystemExit(f"{path}: no cross-likelihood table at step {step} (python -m evotaste.evaluate deviation "
                         f"runs/{run.name} --steps {step} --cross)")
    df = pd.read_csv(path, dtype={"scorer": str, "author": str, "uid": str, "domain": str})
    if list(df.columns) != CROSS_COLUMNS:
        raise ValueError(f"{path}: columns {list(df.columns)}, expected {CROSS_COLUMNS}")
    if (df["step"] != step).any():
        raise ValueError(f"{path}: rows of another step")
    subjects = set(df["scorer"])
    if set(df["author"]) != subjects or df.duplicated(["scorer", "author", "uid"]).any():
        raise ValueError(f"{path}: scorers and authors differ, or a (scorer, author, item) appears twice")
    if (df.groupby(["author", "uid"])["tokens"].nunique() != 1).any() or (
            df.groupby(["author", "uid"]).size() != len(subjects)).any():
        raise ValueError(f"{path}: an answer is not scored once by every scorer with the same tokens")
    _items_of(run, sel, df[df["scorer"] == "base"].copy(), "author", path.name)
    meta = {r["uid"]: r["family"] for r in run.base_records()}
    df["family"] = [meta[u] for u in df["uid"]]
    df["unit"] = [run.units[f] for f in df["family"]]
    return df, sel


@dataclass
class Scores:
    """A context-score selection as arrays: per subject (base, base:<context>, <lineage>@<step>), context and item,
    the response's summed log-ratio log p_base(token | context prompt) - log p_base(token | plain prompt), and per
    subject and item its completion tokens."""
    dir: Path
    subjects: list[str]
    lineage: list[str | None]
    step: list[int | None]
    contexts: list[str]
    uids: list[str]
    domains: np.ndarray  # per item
    delta: np.ndarray  # subject x context x item
    tokens: np.ndarray  # subject x item


def selections(run: Run) -> list[Path]:
    """The run's context-score selections with per-response scores."""
    return sorted(p.parent for p in (run.dir / "context_score").glob("*/responses.csv"))


def context_scores(run: Run, selection: Path | None = None) -> Scores:
    """The engine's per-response context scores (runs/<name>/context_score/<selection>/responses.csv; `selection`, or
    the run's only selection). Every subject must have every context and item exactly once, with the same completion
    tokens under every context, and the sums must reproduce the selection's table.csv."""
    if selection is None:
        found = selections(run)
        if len(found) != 1:
            raise SystemExit(f"{run.dir}/context_score: {len(found)} selections with responses.csv "
                             f"({', '.join(p.name for p in found) or 'none'}); run python -m evotaste.evaluate "
                             "context-score, or name one with --selection")
        selection = found[0]
    selection = Path(selection)
    r = pd.read_csv(selection / "responses.csv", dtype={"subject": str, "lineage": str, "uid": str, "domain": str,
                                                        "context": str})
    expected = ["subject", "lineage", "step", "uid", "domain", "context", "tokens", "delta_sum"]
    if list(r.columns) != expected:
        raise ValueError(f"{selection}/responses.csv: columns {list(r.columns)}, expected {expected}")
    subjects = list(dict.fromkeys(r["subject"]))
    contexts = sorted(r["context"].unique())
    uids = sorted(r["uid"].unique())
    si, ci, ui = ({k: i for i, k in enumerate(v)} for v in (subjects, contexts, uids))
    shape = (len(subjects), len(contexts), len(uids))
    if len(r) != np.prod(shape) or r.duplicated(["subject", "context", "uid"]).any():
        raise ValueError(f"{selection}/responses.csv: not every subject has every context and item exactly once")
    idx = (r["subject"].map(si).to_numpy(), r["context"].map(ci).to_numpy(), r["uid"].map(ui).to_numpy())
    delta, tokens = np.zeros(shape), np.zeros(shape)
    delta[idx], tokens[idx] = r["delta_sum"].to_numpy(), r["tokens"].to_numpy()
    if (tokens != tokens[:, :1, :]).any():
        raise ValueError(f"{selection}/responses.csv: a response's tokens differ between contexts")
    domain = r.drop_duplicates("uid").set_index("uid")["domain"]
    if r.drop_duplicates(["uid", "domain"]).shape[0] != len(uids):
        raise ValueError(f"{selection}/responses.csv: an item has two domains")
    table = pd.read_csv(selection / "table.csv", dtype={"subject": str, "context": str, "domain": str})
    sums = r.groupby(["subject", "context", "domain"])[["tokens", "delta_sum"]].sum()
    joined = table[["subject", "context", "domain", "tokens", "delta_sum"]].merge(
        sums.reset_index(), on=["subject", "context", "domain"], suffixes=("", "_r"))
    if len(joined) != len(sums) or not (np.allclose(joined["delta_sum"], joined["delta_sum_r"], rtol=1e-6, atol=1e-6)
                                        and (joined["tokens"] == joined["tokens_r"]).all()):
        raise ValueError(f"{selection}: responses.csv does not reproduce table.csv")
    first = r.drop_duplicates("subject").set_index("subject")
    return Scores(selection, subjects, [None if pd.isna(x) else x for x in first.loc[subjects, "lineage"]],
                  [None if pd.isna(x) else int(x) for x in first.loc[subjects, "step"]], contexts, uids,
                  domain.loc[uids].to_numpy(), delta, tokens[:, 0, :])
