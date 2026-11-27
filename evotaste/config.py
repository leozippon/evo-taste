"""Run configuration schema (configs/run/*.yaml). Unknown keys and invalid values are errors, never ignored.

`worker prepare` freezes the file into runs/<name>/config.yaml; every later command reads only that copy.
"""

from __future__ import annotations

import dataclasses
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROLES = ("member", "baseline")
SEEDING_METHODS = ("opcd", "imitation", "none")
ADVANTAGES = ("centered", "standardized")
PRACTICE_REFERENCES = (None, "base")
REVISIT_SOURCES = ("own", "matched")
_ID = re.compile(r"^[A-Za-z0-9_-]+$")  # '~' and '@' are reserved for lineage and adapter-version names


@dataclass(frozen=True)
class Server:
    kv_cache: str = "20G"  # --kv-cache-memory-bytes
    # With a fixed KV budget vLLM uses this only for a startup check (free memory >= fraction x card); its default
    # 0.92 fails as soon as the trainer or anyone else holds memory. KV 20G needs about 0.61 of an L20.
    gpu_memory_utilization: float = 0.65
    max_num_seqs: int = 512
    max_num_batched_tokens: int = 8192
    max_loras: int = 8  # adapters that can share one batch; raise it with more members per worker


@dataclass(frozen=True)
class Lora:
    rank: int = 32
    alpha: int = 64


@dataclass(frozen=True)
class Trainer:
    token_budget: int = 8192  # padded tokens per micro-batch (rows x longest row), and the length of a packed row
    shared_prefix: bool = True  # pack a prompt's completions behind one copy of the prompt, up to token_budget
    grad_checkpointing: bool = True
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    # policy-consistency check before every update: re-score at least check_tokens completion tokens and at least
    # check_rollouts rollouts of the step's batch with the trainer (all of them if the batch is smaller)
    check_rollouts: int = 8
    check_tokens: int = 3000


@dataclass(frozen=True)
class Seeding:
    method: str = "opcd"  # opcd (on-policy context distillation) | imitation (of correct teacher samples) | none
    steps: int = 40  # updates
    batch: int = 64  # prompts per update (Streams.seed_batch)
    lr: float = 1e-4
    max_tokens: int = 4096  # the member's samples (opcd) or the teacher's (imitation)
    domains: tuple[str, ...] | None = None  # domains of the seeding prompts; None: all
    samples: int = 4  # imitation: teacher samples per prompt


@dataclass(frozen=True)
class Length:
    """return = reward - slope * clip((tokens - free) / (max_tokens - free), 0, 1)"""
    free: int = 2048
    slope: float = 0.25


@dataclass(frozen=True)
class Exaggeration:
    """Exaggerated self-imitation: weight x the token-mean KL(q || member) over the eligible rollouts, with the fixed
    target q = softmax(l_m + gamma (l_m - l_b)) from the member's and the base model's logits."""
    gamma: float
    weight: float


@dataclass(frozen=True)
class Practice:
    """Self-chosen practice: source s gets default_s exp(eta (r_s - mean r)) of the batch, within floor..ceiling x
    default_s, where r_s is the member's moving average (decay `decay`) of its training reward on s; with reference
    `base`, of that reward minus the base model's census pass rate on the same items."""
    eta: float
    decay: float
    floor: float
    ceiling: float
    reference: str | None = None  # None: the member's own reward; "base": its gain over the base model's pass rate


@dataclass(frozen=True)
class Taste:
    """Taste-shaped reward: the reward of a correct, eligible rollout times 1 + strength x z, where z is its
    characteristic score (log p_member - log p_base per completion token) relative to the other eligible rollouts of
    its group, standardised over the step and clipped to +-clip. strength may be negative (toward the base model).
    With `residualize`, the score's least-squares fit on log completion length over the step's shaped rollouts is
    removed first, so that shaping does not follow length."""
    strength: float
    clip: float = 2.0
    residualize: bool = False


@dataclass(frozen=True)
class Revisit:
    """Consolidating one's own frontier: `share` of a step's prompts (rounded half up, at least one when share > 0)
    revisit items the member was shown before, in place of the last prompts of the shared batch. `own`: items of its
    frontier (solved at least once, in fewer than 3/4 of at least `min_seen` attempts), drawn with probability
    proportional to 1 - census pass rate + a small constant (member.REVISIT_EPSILON); `matched`: for each item `own`
    would draw, the shown item
    outside the frontier of the same source with the nearest census pass rate. `imitate` > 0 adds that weight times the
    token-mean NLL of each revisited item's stored first successful answer."""
    share: float
    source: str  # own | matched
    min_seen: int = 4  # attempts (one visit is `rollouts` attempts)
    imitate: float = 0.0


@dataclass(frozen=True)
class Amplification:
    steps: int = 60
    prompts: int = 64
    rollouts: int = 4
    max_tokens: int = 4096
    lr: float = 1e-4
    length: Length = field(default_factory=Length)
    advantage: str = "centered"  # centered: (return - mean) / 0.5; standardized: (return - mean) / std (Round 1)
    self_imitation: float = 0.0  # beta: weight of the token-mean NLL of the eligible rollouts
    exaggeration: Exaggeration | None = None
    # eligible for the imitation terms: reward 1 and, if true, at most length.free completion tokens (pilot C: any
    # rollout with reward 1)
    imitate_within_free: bool = False
    practice: Practice | None = None  # None: the run's shared stream
    taste: Taste | None = None  # None (or strength 0): rewards as scored
    revisit: Revisit | None = None  # None: no item record, the shared batch


@dataclass(frozen=True)
class Probes:
    every: int = 15
    max_tokens: int = 4096  # also used by `evaluate`


@dataclass(frozen=True)
class Culling:
    enabled: bool = True  # false: would-be culls are recorded and logged, nothing is culled or replaced
    grace: int = 40
    max_replacements: int = 2  # new lineages per member slot before the slot is given up


@dataclass(frozen=True)
class Overrides:
    """Per-member settings that replace the run's (for example a baseline's longer course)."""
    seeding: dict = field(default_factory=dict)
    amplification: dict = field(default_factory=dict)
    culling: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Parent:
    member: str  # a non-clone member of the same run; the clone starts from its first lineage's checkpoint
    step: int


@dataclass(frozen=True)
class Strata:
    """The census strata table the training stream is drawn from (outside git), pinned by its SHA-256."""
    path: str
    sha256: str


@dataclass(frozen=True)
class MemberSpec:
    id: str
    seed: int  # lineage seed: LoRA initialisation, every sampling seed, imitation draws
    context: str | None = None  # a name in the contexts file; required by seeding opcd and imitation
    parent: Parent | None = None  # a clone: starts from the parent's checkpoint at that step, optimizer included;
    # the parent may itself be a clone
    overrides: Overrides = field(default_factory=Overrides)
    role: str = "member"  # member: population, ranked by the harvest; baseline: listed but never ranked


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: str
    seed: int  # training stream and seeding prompts
    members: tuple[MemberSpec, ...]
    suites: str = "configs/suites-r2.yaml"
    contexts: str = "configs/contexts.yaml"
    strata: Strata | None = None  # required as soon as any member has amplification steps
    # Steps at which every member keeps its checkpoint with optimizer state, so that clones can be added to the run
    # after their parents have passed them (worker prepare on the config extended by those clones).
    clonable_steps: tuple[int, ...] = ()
    server: Server = field(default_factory=Server)
    lora: Lora = field(default_factory=Lora)
    trainer: Trainer = field(default_factory=Trainer)
    seeding: Seeding = field(default_factory=Seeding)
    amplification: Amplification = field(default_factory=Amplification)
    probes: Probes = field(default_factory=Probes)
    culling: Culling = field(default_factory=Culling)

    def member(self, member_id: str) -> MemberSpec:
        for m in self.members:
            if m.id == member_id:
                return m
        raise KeyError(f"member {member_id!r} not in run {self.name}")

    def settings(self, spec: MemberSpec) -> tuple[Seeding, Amplification, Culling]:
        """The member's effective seeding, amplification and culling settings. A clone continues its parent: it
        takes the parent's effective settings, with its own amplification and culling overrides on top."""
        base = self.settings(self.member(spec.parent.member)) if spec.parent else (
            self.seeding, self.amplification, self.culling)
        return tuple(_override(cls, value, getattr(spec.overrides, name), name)
                     for cls, value, name in zip((Seeding, Amplification, Culling), base,
                                                 ("seeding", "amplification", "culling")))

    def context_of(self, spec: MemberSpec) -> str | None:
        """The member's seeding context; a clone's is its parent's."""
        return self.context_of(self.member(spec.parent.member)) if spec.parent else spec.context

    def start_step(self, spec: MemberSpec) -> int:
        """The amplification step a member starts at: 0, or a clone's clone point."""
        return spec.parent.step if spec.parent else 0

    def clone_points(self, member_id: str) -> set[int]:
        """Steps at which the lineage `member_id` keeps a checkpoint with optimizer state: those its clones name, and
        the run's clonable steps."""
        return {m.parent.step for m in self.members if m.parent and m.parent.member == member_id} | set(
            self.clonable_steps)


def _override(cls, value, changes: dict, where: str):
    """`value` (a dataclass) with the keys of `changes` replaced; nested sections are merged key by key."""
    data = dataclasses.asdict(value)
    for k, v in changes.items():
        data[k] = {**data[k], **v} if isinstance(data.get(k), dict) and isinstance(v, dict) else v
    return _build(cls, data, where)


def _convert(tp, value, where: str):
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    if origin in (typing.Union, types.UnionType):  # X | None
        if value is None:
            return None
        tp = next(a for a in typing.get_args(tp) if a is not type(None))
        origin = typing.get_origin(tp)
    if origin is tuple:  # tuple[X, ...]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where}: expected a list, got {value!r}")
        return tuple(_convert(typing.get_args(tp)[0], v, f"{where}[{i}]") for i, v in enumerate(value))
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    if tp is float and isinstance(value, (int, float, str)) and not isinstance(value, bool):
        try:
            return float(value)  # YAML reads 1e-5 as a string
        except ValueError:
            raise ValueError(f"{where}: expected a number, got {value!r}") from None
    if tp is dict and isinstance(value, dict):
        return value
    if not isinstance(value, tp) or (tp is int and isinstance(value, bool)):
        raise ValueError(f"{where}: expected {tp.__name__}, got {value!r}")
    return value


def _build(cls, data, where: str):
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping, got {data!r}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)} (allowed: {sorted(names)})")
    missing = [f.name for f in dataclasses.fields(cls) if f.name not in data and f.default is dataclasses.MISSING
               and f.default_factory is dataclasses.MISSING]
    if missing:
        raise ValueError(f"{where}: missing required keys {missing}")
    return cls(**{k: _convert(hints[k], v, f"{where}.{k}") for k, v in data.items()})


def _check_settings(where: str, seeding: Seeding, amp: Amplification, culling: Culling) -> None:
    if seeding.method not in SEEDING_METHODS:
        raise ValueError(f"{where}: seeding.method must be one of {SEEDING_METHODS}")
    if min(seeding.steps, amp.steps, culling.grace, culling.max_replacements) < 0:
        raise ValueError(f"{where}: seeding.steps, amplification.steps and culling settings must be >= 0")
    if min(seeding.batch, seeding.max_tokens, seeding.samples, amp.prompts, amp.max_tokens) < 1:
        raise ValueError(f"{where}: seeding batch, max_tokens, samples and amplification prompts, max_tokens "
                         "must be >= 1")
    if seeding.domains is not None and not seeding.domains:
        raise ValueError(f"{where}: seeding.domains must be null (all domains) or a non-empty list")
    if amp.rollouts < 2:
        raise ValueError(f"{where}: amplification.rollouts must be >= 2")
    if not 0 <= amp.length.free < amp.max_tokens or amp.length.slope < 0:
        raise ValueError(f"{where}: amplification.length needs 0 <= free < max_tokens and slope >= 0")
    if amp.advantage not in ADVANTAGES:
        raise ValueError(f"{where}: amplification.advantage must be one of {ADVANTAGES}")
    if amp.self_imitation < 0:
        raise ValueError(f"{where}: amplification.self_imitation must be >= 0")
    ex, pr, ta, rv = amp.exaggeration, amp.practice, amp.taste, amp.revisit
    if ex is not None and min(ex.gamma, ex.weight) < 0:
        raise ValueError(f"{where}: amplification.exaggeration needs gamma >= 0 and weight >= 0")
    if pr is not None and not (pr.eta >= 0 and 0 <= pr.decay < 1 and 0 < pr.floor <= 1 <= pr.ceiling):
        raise ValueError(f"{where}: amplification.practice needs eta >= 0, 0 <= decay < 1 and "
                         "0 < floor <= 1 <= ceiling")
    if pr is not None and pr.reference not in PRACTICE_REFERENCES:
        raise ValueError(f"{where}: amplification.practice.reference must be null (the member's own reward) or base")
    # a correct answer must keep a positive reward, above a wrong one's: factors 1 +- strength x clip stay above 0
    if ta is not None and not (ta.clip > 0 and abs(ta.strength) * ta.clip < 1):
        raise ValueError(f"{where}: amplification.taste needs clip > 0 and |strength| x clip < 1")
    if rv is not None and not (0 <= rv.share <= 1 and rv.source in REVISIT_SOURCES and rv.min_seen >= 1
                               and rv.imitate >= 0):
        raise ValueError(f"{where}: amplification.revisit needs 0 <= share <= 1, source one of {REVISIT_SOURCES}, "
                         "min_seen >= 1 and imitate >= 0")
    if rv is not None and pr is not None:  # practice draws its own batch; revisits replace part of the shared one
        raise ValueError(f"{where}: amplification.revisit and amplification.practice cannot be combined")


def parse(data: dict, context_names) -> RunConfig:
    cfg = _build(RunConfig, data, "config")
    ids = [m.id for m in cfg.members]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("config.members: member ids must be present and unique")
    seeds = [m.seed for m in cfg.members]
    repeated = sorted({s for s in seeds if seeds.count(s) > 1})
    if repeated:  # a shared seed means a shared LoRA initialisation and common random numbers in every draw
        raise ValueError(f"config.members: seeds must be distinct; repeated: {repeated}")
    if any(s < 0 for s in cfg.clonable_steps):
        raise ValueError("config.clonable_steps: steps must be >= 0")
    for m in cfg.members:
        where = f"member {m.id}"
        if not _ID.match(m.id):
            raise ValueError(f"{where}: use letters, digits, '_' and '-' only")
        if m.role not in ROLES:
            raise ValueError(f"{where}: role must be one of {ROLES}")
        if m.parent is not None:
            _check_clone(cfg, m)
            continue
        seeding, amp, culling = cfg.settings(m)  # also validates the overrides
        _check_settings(where, seeding, amp, culling)
        if (seeding.method == "none") != (m.context is None):
            raise ValueError(f"{where}: a context is required for, and only for, seeding methods opcd and imitation")
        if m.context is not None and m.context not in context_names:
            raise ValueError(f"{where}: unknown context {m.context!r}")
    if cfg.probes.every < 1 or cfg.probes.max_tokens < 1:
        raise ValueError("probes.every and probes.max_tokens must be >= 1")
    if cfg.strata is None and any(cfg.settings(m)[1].steps > 0 for m in cfg.members):
        raise ValueError("config.strata: the training stream needs the census strata table ({path, sha256})")
    return cfg


def _check_clone(cfg: RunConfig, m: MemberSpec) -> None:
    where = f"member {m.id} (a clone)"
    chain, spec = [m.id], m
    while spec.parent is not None:  # the chain of parents must end at a seeded member
        try:
            spec = cfg.member(spec.parent.member)
        except KeyError:
            raise ValueError(f"{where}: parent {spec.parent.member!r} is not a member of this run") from None
        if spec.id in chain:
            raise ValueError(f"{where}: its chain of parents is a cycle: {' <- '.join(chain + [spec.id])}")
        chain.append(spec.id)
    if m.context is not None or m.overrides.seeding:
        raise ValueError(f"{where}: a clone has its parent's context and seeding; do not set them")
    seeding, amp, culling = cfg.settings(m)
    _check_settings(where, seeding, amp, culling)
    parent = cfg.member(m.parent.member)
    start, end = cfg.start_step(parent), cfg.settings(parent)[1].steps
    if not start <= m.parent.step <= end or m.parent.step >= amp.steps:
        raise ValueError(f"{where}: parent.step {m.parent.step} must be within the parent's course ({start}..{end}) "
                         f"and below the clone's amplification.steps ({amp.steps})")
    if m.parent.step > 0 and amp.practice != cfg.settings(parent)[1].practice:
        raise ValueError(f"{where}: a clone after step 0 continues its parent's practice record and must keep its "
                         "practice settings")


def load_contexts(path: str | Path) -> dict[str, str]:
    """Context name -> the text the teacher reads: preamble, the context's text and closing (configs/contexts.yaml).
    Names are unique across the placebo, stance and tactic sections."""
    data = yaml.safe_load(Path(path).read_text())
    named = [("placebo", data["placebo"]), *data["stances"].items(), *data["tactics"].items()]
    names = [n for n, _ in named]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: context names are not unique: {names}")
    return {name: f"{data['preamble']} {v['text']} {data['closing']}" for name, v in named}


def load(path: str | Path) -> RunConfig:
    data = yaml.safe_load(Path(path).read_text())
    return parse(data, load_contexts(data.get("contexts", RunConfig.contexts)))
