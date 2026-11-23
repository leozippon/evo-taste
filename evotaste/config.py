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

ARMS = ("plain", "scaffold", "perturb")
ROLES = ("member", "baseline")
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
    token_budget: int = 8192  # padded tokens per micro-batch (rows x longest row)
    shared_prefix: bool = True  # run each prompt once for its K completions
    grad_checkpointing: bool = True
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    check_rollouts: int = 8  # rollouts re-scored by the trainer before every update (policy-consistency check)


@dataclass(frozen=True)
class Seeding:
    pool_prompts: int = 1536
    samples: int = 4  # teacher samples per pool prompt
    max_tokens: int = 640
    steps: int = 30
    batch: int = 64  # sequences per positive update
    lr: float = 1e-5
    perturb_std: float = 0.01  # perturbation arm: LoRA B ~ N(0, perturb_std)


@dataclass(frozen=True)
class Amplification:
    steps: int = 300
    prompts: int = 64
    rollouts: int = 4
    max_tokens: int = 640
    lr: float = 1e-5
    # lambda >= 0: group weights come from reward - lambda * truncated (cut off at max_tokens); 0 keeps the reward
    truncation_penalty: float = 0.0


@dataclass(frozen=True)
class Probes:
    every: int = 20
    train_per_family: int = 8
    selection_per_family: int = 12
    foundation_per_source: int = 150
    max_tokens: int = 640


@dataclass(frozen=True)
class Culling:
    grace: int = 40
    max_replacements: int = 2  # new lineages per member slot before the slot is given up


@dataclass(frozen=True)
class Overrides:
    """Per-member settings that replace the run's (this is how the long and wide baselines are expressed)."""
    seeding: dict = field(default_factory=dict)
    amplification: dict = field(default_factory=dict)
    culling: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MemberSpec:
    id: str
    arm: str  # plain | scaffold | perturb
    seed: int
    scaffold: str | None = None  # a name in configs/scaffolds.yaml (control or scaffolds); scaffold arm only
    overrides: Overrides = field(default_factory=Overrides)
    role: str = "member"  # member: population, ranked by the harvest; baseline: single-model RL, listed but never ranked


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: str
    seed: int  # training stream and seeding pool
    members: tuple[MemberSpec, ...]
    suites: str = "configs/suites.yaml"
    scaffolds: str = "configs/scaffolds.yaml"
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
        """The member's effective seeding, amplification and culling settings."""
        return (_build(Seeding, {**dataclasses.asdict(self.seeding), **spec.overrides.seeding}, "seeding"),
                _build(Amplification, {**dataclasses.asdict(self.amplification), **spec.overrides.amplification},
                       "amplification"),
                _build(Culling, {**dataclasses.asdict(self.culling), **spec.overrides.culling}, "culling"))


def _convert(tp, value, where: str):
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    if origin is tuple:  # tuple[MemberSpec, ...]
        if not isinstance(value, list):
            raise ValueError(f"{where}: expected a list")
        return tuple(_convert(typing.get_args(tp)[0], v, f"{where}[{i}]") for i, v in enumerate(value))
    if origin in (typing.Union, types.UnionType):  # X | None
        if value is None:
            return None
        tp = next(a for a in typing.get_args(tp) if a is not type(None))
    if tp is float and isinstance(value, (int, float, str)) and not isinstance(value, bool):
        return float(value)  # YAML reads 1e-5 as a string
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
    return cls(**{k: _convert(hints[k], v, f"{where}.{k}") for k, v in data.items()})


def parse(data: dict, scaffold_names) -> RunConfig:
    cfg = _build(RunConfig, data, "config")
    ids = [m.id for m in cfg.members]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("config.members: member ids must be present and unique")
    for m in cfg.members:
        if not _ID.match(m.id):
            raise ValueError(f"member id {m.id!r}: use letters, digits, '_' and '-' only")
        if m.arm not in ARMS:
            raise ValueError(f"member {m.id}: arm must be one of {ARMS}")
        if m.role not in ROLES:
            raise ValueError(f"member {m.id}: role must be one of {ROLES}")
        if (m.arm == "scaffold") != (m.scaffold is not None):
            raise ValueError(f"member {m.id}: a scaffold is required for, and only for, the scaffold arm")
        if m.scaffold is not None and m.scaffold not in scaffold_names:
            raise ValueError(f"member {m.id}: unknown scaffold {m.scaffold!r}")
        shared_pool = set(m.overrides.seeding) & {"pool_prompts", "samples", "max_tokens"}
        if shared_pool:
            raise ValueError(f"member {m.id}: the seeding pool is shared per teacher; cannot override {shared_pool}")
        if cfg.settings(m)[1].truncation_penalty < 0:  # also validates the overrides
            raise ValueError(f"member {m.id}: amplification.truncation_penalty must be >= 0")
    if cfg.probes.every < 1 or cfg.amplification.rollouts < 2:
        raise ValueError("probes.every must be >= 1 and amplification.rollouts >= 2")
    return cfg


def load(path: str | Path) -> RunConfig:
    from evotaste.tasks import load_scaffolds

    data = yaml.safe_load(Path(path).read_text())
    return parse(data, load_scaffolds(data.get("scaffolds", RunConfig.scaffolds)))
