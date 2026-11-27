"""Stand-ins for the task layer, the tokenizer and the vLLM sampler, so that the engine's whole member life cycle runs
on the CPU with the tiny model of conftest.py. Task and Score carry exactly the contract's fields
(docs/dev/ROUND2_SPEC.md, "Contract between packages"); prompts are token ids derived from the question and context."""

from __future__ import annotations

import hashlib
import random
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from evotaste import config
from evotaste.member import Run
from evotaste.sampler import Sample
from evotaste.tasks import stable_seed
from evotaste.trainer import MemberWeights, Sequence
from tests.conftest import VOCAB

ROOT = Path(__file__).resolve().parents[1]
MODEL = "tiny"
EOT = VOCAB - 1  # the fake end-of-turn token


@dataclass(frozen=True)
class Task:
    uid: str
    family: str
    domain: str
    kind: str
    question: str
    gold: str
    wrapper: str
    stratum: str | None


@dataclass(frozen=True)
class Score:
    reward: float
    correct_any: bool
    complied: bool
    channel: str | None
    truncated: bool
    extracted: str | None
    error: str | None


def score_many(jobs, timeout=None):
    """Reward 1 iff the completion stopped and its first token is even; complied iff it has at least two tokens."""
    out = []
    for _task, text, finish in jobs:
        ids = [int(x) for x in text.split()]
        correct = ids[0] % 2 == 0
        out.append(Score(float(correct and finish == "stop"), correct, len(ids) > 1, "direct", finish != "stop",
                         text.split()[0], None))
    return out


def task(prefix: str, j: int, domain: str = "math", wrapper: str = "box") -> Task:
    return Task(f"{prefix}/{j}", f"{domain}_{j % 2}", domain, "open", f"{prefix} question {j}", "0", wrapper, "sweet")


class Streams:
    def __init__(self):
        self.probe_tasks = [task("probe", j, ("math", "code", "logic")[j % 3], ("box", "answer_tag")[j % 2])
                            for j in range(6)]

    def train_batch(self, step, prompts):
        return [task(f"train/{step}", j, ("math", "code")[j % 2]) for j in range(prompts)]

    def seed_batch(self, i, n, domains=None):
        return [task(f"seed/{i}", j, (domains or ("math", "logic"))[j % len(domains or ("math", "logic"))])
                for j in range(n)]


class FakeTokenizer:
    """Token ids that are multiples of 3 have no letter or digit in their text (layout), the end-of-turn id none at
    all; every other id reads as a word."""

    def __len__(self):
        return VOCAB

    def decode(self, ids, skip_special_tokens=False):
        return "".join("" if i == EOT and skip_special_tokens else "<|eot|>" if i == EOT else
                       " **\n" if i % 3 == 0 else f" w{i}" for i in ids)


class FakeRun(Run):
    """Run with the stand-in task layer: prompts are 4-8 token ids of the question, the context's ids before them."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.cfg = config.load(self.dir / "config.yaml")
        self.contexts = config.load_contexts(self.cfg.contexts)
        self.tokenizer = FakeTokenizer()
        self.streams = Streams()
        self.probe_tasks = self.streams.probe_tasks
        self.policy = None
        self.sampler = None
        self.prompt = lru_cache(maxsize=None)(self._prompt)

    def _prompt(self, question, context=None):
        rng = random.Random(stable_seed("prompt", question))
        ids = [rng.randrange(VOCAB - 1) for _ in range(rng.randint(4, 8))]
        if context is not None:
            ids = [rng.randrange(VOCAB - 1) for _ in range(3)] + [stable_seed(context) % (VOCAB - 1)] + ids
        return ids


class FakeSampler:
    """Draws token ids from the request's seed (sample k from seed + k, as vLLM) and reports the log-probabilities
    of the version it serves, which it loads from the bf16 adapter directory the member saved, as vLLM does."""

    def __init__(self, policy, root: Path):
        self.policy, self.root = policy, root
        self.versions: dict[str, MemberWeights] = {}
        self.current: dict[str, str] = {}
        self.counter = 0
        self.requests = []

    def publish(self, member, save):
        self.counter += 1
        name = f"{member}@{self.counter}"
        save(self.root / name)
        self.versions[name] = MemberWeights(self.policy.load_adapter(self.root / name))
        shutil.rmtree(self.root / name)
        self.retire(member)
        self.current[member] = name
        return name, {"save_s": 0.0, "load_s": 0.0, "unload_s": 0.0, "cleanup_s": 0.0}

    def retire(self, member):
        if member in self.current:
            del self.versions[self.current.pop(member)]

    def load_existing(self, label, path):
        self.counter += 1
        name = f"{label}@{self.counter}"
        self.versions[name] = MemberWeights(self.policy.load_adapter(path))
        return name

    def close(self):
        pass

    def generate(self, requests):
        self.requests += requests
        out = []
        for r in requests:
            group = []
            for k in range(r.n):
                rng = random.Random(r.seed + k)
                n = rng.randint(1, r.max_tokens)
                ids = [rng.randrange(VOCAB - 1) for _ in range(n)] + ([EOT] if n < r.max_tokens else [])
                seq = [Sequence(r.prompt, ids)]
                lps = (self.policy.base_logprobs(seq) if r.model == MODEL else
                       self.policy.logprobs(self.versions[r.model], seq))[0].tolist()
                group.append(Sample(" ".join(map(str, ids)), ids, "stop" if ids[-1] == EOT else "length", lps))
            out.append(group)
        return out


def frozen_run(tmp_path: Path, data: dict, name: str = "config.yaml") -> Path:
    """runs/<name>/ as `worker.freeze` makes it from a config file in tmp_path, which must be the working directory,
    with stand-in suites and strata files and a copy of the real contexts file."""
    from evotaste import worker

    if not (tmp_path / "contexts.yaml").exists():
        shutil.copyfile(ROOT / "configs" / "contexts.yaml", tmp_path / "contexts.yaml")
        (tmp_path / "suites.yaml").write_text("suites: {}\n")
        (tmp_path / "strata.tsv").write_text("uid\tpasses\tsamples\n")
    sha = hashlib.sha256((tmp_path / "strata.tsv").read_bytes()).hexdigest()
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"model": MODEL, "seed": 0, "suites": "suites.yaml", "contexts": "contexts.yaml",
                                    "strata": {"path": "strata.tsv", "sha256": sha}, **data}, sort_keys=False))
    return worker.freeze(path)


def write_run(tmp_path: Path, data: dict) -> Path:
    """runs/<name>/ with the given config and a copy of the real contexts file."""
    run_dir = tmp_path / "runs" / data["name"]
    run_dir.mkdir(parents=True)
    shutil.copyfile(ROOT / "configs" / "contexts.yaml", run_dir / "contexts.yaml")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"model": MODEL, "seed": 0, **data,
                                                         "contexts": str(run_dir / "contexts.yaml")}))
    return run_dir
