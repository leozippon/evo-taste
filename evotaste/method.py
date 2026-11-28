"""Method census (docs/design/scale.md, questions 1 and 2; docs/dev/SCALE_SPEC.md, WP-M): inference only.

  python -m evotaste.method items configs/method.yaml [--limit N]
  python -m evotaste.method attempts configs/method.yaml --model L [--shard i/n] [--limit N] [server overrides]
  python -m evotaste.method alternatives configs/method.yaml --model L [--shard i/n] [--limit N] [server overrides]
  python -m evotaste.method pairs configs/method.yaml [--limit N]
  python -m evotaste.method judge configs/method.yaml [--shard i/n] [--limit N] [server overrides]

Stages, in this order, each writing under results/method/<name>/ (<name>-limitN with --limit: the first N drawn items):
  items         a seeded draw of training items, stratified by domain and the census (1.7B) stratum: items.tsv
  attempts      every subject model answers every item with the training prompt, `attempts.samples` at temperature 1
  alternatives  shown its own first correct attempt, the model is asked for a solution by a different method
  pairs         the pairs of correct solutions to judge, drawn once: pairs.jsonl.gz
  judge         the judge model names the approach of each solution and relates them (same / variant / different),
                greedy and constrained to a JSON object; then the reader sample for independent readers

Every stage records the settings its records depend on, with the hashes of the stages it reads, in a manifest, and
refuses to continue or to read records made under other settings. Generation stages run in chunks, written atomically
once scored (census.run_chunks), so a restart skips finished chunks and shard i of n takes the chunks whose index is
i mod n. A process uses the GPUs in CUDA_VISIBLE_DEVICES (more than one only with tensor parallelism in the model's
server arguments) and refuses to start unless each has the model's `need_free_mib` free.

Only items of the train part are read (Suites.items(source, "train")); every uid is checked against that part.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import itertools
import json
import logging
import os
import random
import resource
import string
import subprocess
from pathlib import Path
from types import SimpleNamespace

import yaml

from evotaste.census import CensusJob, run_chunks, sample_records
from evotaste.records import atomic_write, now, read_json, read_jsonl_gz, write_json, write_jsonl_gz
from evotaste.tasks import (Suites, Task, default_wrapper, instruction, prompt_ids, score_many, stable_seed,
                            strata_rows, stratum_of)

log = logging.getLogger("evotaste.method")
STAGES = ("items", "attempts", "alternatives", "pairs", "judge")
DOMAINS = ("math", "code", "logic")
STRATA = ("mastered", "sweet", "frontier")
KEYS = {"name", "suites", "census", "items", "models", "attempts", "alternatives", "pairs", "judge"}
THINK_OFF = "<think>\n\n</think>\n\n"  # the end of a prompt whose template honoured enable_thinking=False
THINK_ON = "<think>\n"  # the end of a Qwen3.5-family prompt with thinking enabled
EFFORTS = ("low", "medium", "xhigh")  # the reasoning efforts of the Qwen3.5-family template
RELATIONS = ("same", "variant", "different")
VERDICT = ("approach_a", "approach_b", "relation")  # the judge's keys, in the order it must give them
APPROACH_CHARS = 160  # hard cap on one approach (the prompt asks for at most 15 words)
SCHEMA = {"type": "object",
          "properties": {"approach_a": {"type": "string", "minLength": 1, "maxLength": APPROACH_CHARS},
                         "approach_b": {"type": "string", "minLength": 1, "maxLength": APPROACH_CHARS},
                         "relation": {"enum": list(RELATIONS)}},
          "required": list(VERDICT), "additionalProperties": False}
HEAD = 2 / 3  # share of a truncated solution's token budget kept from its start (the rest from its end)


# ---------------------------------------------------------------------------------------------------------------
# Configuration and manifests


def _fields(template: str, names: set[str], where: str) -> None:
    found = {f for _, f, _, _ in string.Formatter().parse(template) if f is not None}
    if found != names:
        raise ValueError(f"{where}: template fields {sorted(found)}, expected {sorted(names)}")


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != KEYS:
        raise ValueError(f"{path}: missing keys {sorted(KEYS - set(cfg))}, unknown keys {sorted(set(cfg) - KEYS)}")
    targets = cfg["items"]["targets"]
    if not set(targets) <= set(DOMAINS) or any(set(t) != set(STRATA) for t in targets.values()):
        raise ValueError(f"{path}: items.targets must map domains of {DOMAINS} to counts for {STRATA}")
    j = cfg["judge"]
    if j["model"] not in cfg["models"]:
        raise ValueError(f"{path}: judge.model {j['model']!r} is not a label of models")
    if j["thinking"] not in (False, *EFFORTS):
        raise ValueError(f"{path}: judge.thinking must be false or one of {EFFORTS}")
    if j["thinking"] and "--reasoning-parser" not in cfg["models"][j["model"]]["server"]["args"]:
        raise ValueError(f"{path}: a thinking judge needs '--reasoning-parser qwen3' in its server args, so that the "
                         "JSON constraint starts after the reasoning")
    if set(j["method"]) != set(DOMAINS):
        raise ValueError(f"{path}: judge.method must define the method for each of {DOMAINS}")
    _fields(cfg["alternatives"]["prompt"], {"problem", "solution", "instruction"}, f"{path}: alternatives.prompt")
    _fields(j["prompt"], {"method", "relations", "problem", "a", "b"}, f"{path}: judge.prompt")
    return cfg


def out_dir(cfg: dict, limit: int | None) -> Path:
    return Path("results/method") / (cfg["name"] + (f"-limit{limit}" if limit else ""))


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _key(stage: str, label: str | None) -> str:
    return stage if label is None else f"{stage}/{label}"


def manifest(cfg: dict, stage: str, label: str | None = None, limit: int | None = None) -> dict:
    """What a stage's records depend on: its own settings and, through their hashes, those of the stages it reads.
    Server settings that change only memory and speed are left out; the model's weights and server arguments are in."""
    def model(lbl):
        m = cfg["models"][lbl]
        return {"label": lbl, "weights": m["model"], "args": m["server"]["args"]}

    if stage == "items":
        own = {"suites": cfg["suites"], "suites_sha256": hashlib.sha256(Path(cfg["suites"]).read_bytes()).hexdigest(),
               "census": cfg["census"], "items": cfg["items"], "limit": limit}
        up = []
    elif stage in ("attempts", "alternatives"):  # scored here: records under other verifier rules are not comparable
        from evotaste.verify import RULES_VERSION

        own = {"model": model(label), stage: cfg[stage], "verifier_rules": RULES_VERSION}
        up = [("items", None) if stage == "attempts" else ("attempts", label)]
    elif stage == "pairs":
        own = {"subjects": list(cfg["models"]), "pairs": cfg["pairs"]}
        up = [(s, lbl) for lbl in cfg["models"] for s in ("attempts", "alternatives")]
    elif stage == "judge":
        own = {"model": model(cfg["judge"]["model"]), "judge": cfg["judge"]}
        up = [("pairs", None)]
    else:
        raise ValueError(f"unknown stage {stage!r}")
    body = {"stage": stage, "model": label, "config": own,
            "upstream": {_key(s, lbl): manifest(cfg, s, lbl, limit)["sha256"] for s, lbl in up}}
    return body | {"sha256": _sha(body)}


def stage_dir(base: Path, stage: str, label: str | None) -> Path:
    return base / stage / label if label else base / stage


def manifest_path(base: Path, stage: str, label: str | None) -> Path:
    if stage in ("items", "pairs"):
        return base / f"{stage}.json"
    return stage_dir(base, stage, label) / "manifest.json"


def chunk_path(directory: Path, c: int) -> Path:
    return directory / f"chunk-{c:05d}.jsonl.gz"


def open_stage(base: Path, expected: dict, jobs: int, chunks: int) -> None:
    """Start or continue a stage: an existing manifest must be the expected one; a new stage writes its manifest
    before any record."""
    path = manifest_path(base, expected["stage"], expected["model"])
    if path.exists():
        stored = read_json(path)
        if stored["sha256"] != expected["sha256"]:
            raise SystemExit(f"{path}: these records were made under other settings than the current config's; they "
                             "cannot be continued or mixed (use a new name)")
        if (stored["jobs"], stored["chunks"]) != (jobs, chunks):
            raise SystemExit(f"{path}: {stored['jobs']} jobs in {stored['chunks']} chunks recorded, {jobs} in "
                             f"{chunks} now")
        return
    if any(path.parent.glob("chunk-*.jsonl.gz")):
        raise SystemExit(f"{path.parent}: records without a manifest")
    write_json(path, expected | {"jobs": jobs, "chunks": chunks, "time": now()})


def stored_manifest(base: Path, stage: str, label: str | None = None, expected: dict | None = None) -> dict:
    """A stage's manifest on disk, checked against the expected one if given."""
    path = manifest_path(base, stage, label)
    if not path.exists():
        raise SystemExit(f"stage {_key(stage, label)} has not been run here ({path} is missing)")
    stored = read_json(path)
    if expected is not None and stored["sha256"] != expected["sha256"]:
        raise SystemExit(f"{path}: made under other settings than the current config's")
    return stored


def read_stage(base: Path, stage: str, label: str | None = None, expected: dict | None = None) -> list[dict]:
    """Every record of a finished generation stage, in chunk order; refused while chunks are missing."""
    n = stored_manifest(base, stage, label, expected)["chunks"]
    d = stage_dir(base, stage, label)
    missing = [c for c in range(n) if not chunk_path(d, c).exists()]
    if missing:
        raise SystemExit(f"{d}: {len(missing)} of {n} chunks missing (e.g. {missing[:5]}); finish the stage first")
    return [r for c in range(n) for r in read_jsonl_gz(chunk_path(d, c))]


# ---------------------------------------------------------------------------------------------------------------
# Items


def train_index(suites: Suites) -> dict[str, Task]:
    return {t.uid: t for s in suites.train for t in suites.items(s, "train")}


def train_tasks(cfg: dict, uids: list[str]) -> list[Task]:
    """The train-part tasks of these uids (default wrappers); refuses any other uid."""
    index = train_index(Suites(cfg["suites"]))
    stray = sum(u not in index for u in uids)
    if stray:  # a count only: a stray uid could be a vault uid
        raise RuntimeError(f"{stray} items are not in the train part")
    return [index[u] for u in uids]


def proportional(n: int, sizes: dict[str, int]) -> dict[str, int]:
    """n split over pools in proportion to their sizes, by largest remainder (ties by name); never more than a pool
    holds."""
    total = sum(sizes.values())
    if total < n:
        raise ValueError(f"{n} items asked of pools holding {total}")
    alloc = {s: n * k // total for s, k in sizes.items()}
    rest = sorted(sizes, key=lambda s: (-(n * sizes[s] % total), s))[:n - sum(alloc.values())]
    return {s: alloc[s] + (s in rest) for s in sizes}


def draw_items(rows: list[tuple[str, int, int]], train: dict[str, Task], targets: dict, seed: int) -> list[tuple]:
    """(task, passes, samples) of a stratified draw from census strata rows (uid, passes, samples): targets[domain]
    [stratum] items per cell, spread over the cell's sources in proportion to their items, each source's share drawn
    with a seeded sample; the result in a seeded order. Every row must be a train-part item under its default wrapper
    (`train`: uid -> task of the train part)."""
    stray = sum(uid not in train for uid, _, _ in rows)
    if stray:  # a count only
        raise ValueError(f"the census strata table names {stray} items outside the train part")
    cells: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for uid, passes, samples in rows:
        t = train[uid]
        if t.wrapper != default_wrapper(t.kind):
            raise ValueError(f"{uid}: not under its default wrapper")
        cells[(t.domain, stratum_of(passes, samples))][t.family].append((t, passes, samples))
    out = []
    for domain in sorted(targets):
        for stratum in STRATA:
            pools = cells[(domain, stratum)]
            try:
                alloc = proportional(targets[domain][stratum], {s: len(p) for s, p in pools.items()})
            except ValueError as e:
                raise ValueError(f"{domain}/{stratum}: {e}") from None
            for s in sorted(alloc):
                pool = sorted(pools[s], key=lambda x: x[0].uid)
                out += random.Random(stable_seed("method-items", seed, domain, stratum, s)).sample(pool, alloc[s])
    random.Random(stable_seed("method-items", seed, "order")).shuffle(out)
    return out


ITEM_COLUMNS = ("uid", "domain", "source", "passes", "samples")  # passes of samples in the census


def read_items(base: Path) -> list[dict]:
    path = base / "items.tsv"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run the items stage first")
    lines = path.read_text().splitlines()
    if tuple(lines[0].split("\t")) != ITEM_COLUMNS:
        raise ValueError(f"{path}: header {lines[0]!r}")
    rows = [dict(zip(ITEM_COLUMNS, line.split("\t"))) for line in lines[1:]]
    return [r | {"passes": int(r["passes"]), "samples": int(r["samples"])} for r in rows]


def run_items(config: Path, limit: int | None) -> None:
    cfg = load_config(config)
    base = out_dir(cfg, limit)
    expected = manifest(cfg, "items", limit=limit)
    path = base / "items.tsv"
    if path.exists():
        stored_manifest(base, "items", expected=expected)
        log.info("%s exists; the draw is never redone", path)
        return
    rows = strata_rows(cfg["census"]["strata"], cfg["census"]["sha256"])
    drawn = draw_items(rows, train_index(Suites(cfg["suites"])), cfg["items"]["targets"], cfg["items"]["seed"])
    drawn = drawn[:limit] if limit else drawn
    open_stage(base, expected, len(drawn), 1)
    atomic_write(path, ("\t".join(ITEM_COLUMNS) + "\n" + "".join(
        f"{t.uid}\t{t.domain}\t{t.family}\t{p}\t{n}\n" for t, p, n in drawn)).encode())
    cells = collections.Counter((t.domain, stratum_of(p, n)) for t, p, n in drawn)
    log.info("drew %d items: %s", len(drawn), dict(sorted(cells.items())))


def _tasks_of(cfg: dict, items: list[dict]) -> list[Task]:
    tasks = train_tasks(cfg, [r["uid"] for r in items])
    if any((t.domain, t.family) != (r["domain"], r["source"]) for t, r in zip(tasks, items)):
        raise RuntimeError("items.tsv does not match the train part (domain or source differs)")
    return tasks


# ---------------------------------------------------------------------------------------------------------------
# Prompts


def check_thinking_off(tok, model: str) -> None:
    """The subject prompt (tasks.prompt_ids) disables thinking; refuse a template that ignores the flag."""
    text = tok.apply_chat_template([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not text.endswith(THINK_OFF):
        raise SystemExit(f"{model}: the chat template does not honour enable_thinking=False (prompt ends "
                         f"{text[-40:]!r})")


def alternative_question(template: str, task: Task, solution: str) -> str:
    """The user turn of the alternatives stage: the problem, the model's own correct solution, the request for another
    method, and the item's own answer instruction."""
    return template.format(problem=task.problem, solution=solution.strip(),
                           instruction=instruction(task.kind, task.gold, task.wrapper))


def truncate(tok, text: str, budget: int) -> tuple[str, bool]:
    """The text cut to `budget` tokens (HEAD of them from its start, the rest from its end) with a marker saying how
    many were left out; whether it was cut."""
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) <= budget:
        return text, False
    head = round(budget * HEAD)
    return (tok.decode(ids[:head]) + f"\n\n[... {len(ids) - budget} tokens omitted ...]\n\n"
            + tok.decode(ids[len(ids) - (budget - head):])), True


def judge_text(j: dict, problem: str, a: str, b: str, domain: str) -> str:
    return j["prompt"].format(method=j["method"][domain].strip(), relations=j["relations"].strip(), problem=problem,
                              a=a, b=b)


def judge_ids(tok, j: dict, user: str) -> list[int]:
    """Chat-template ids of a judge prompt, thinking disabled or at the configured effort; refuses a template that
    does not do as asked."""
    messages = [{"role": "system", "content": j["system"].strip()}, {"role": "user", "content": user}]
    flags = ({"enable_thinking": True, "reasoning_effort": j["thinking"]} if j["thinking"]
             else {"enable_thinking": False})
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **flags)
    if not text.endswith(THINK_ON if j["thinking"] else THINK_OFF):
        raise SystemExit(f"the judge's chat template does not honour thinking={j['thinking']!r} (prompt ends "
                         f"{text[-40:]!r})")
    return tok(text, add_special_tokens=False).input_ids


def parse_verdict(text: str, thinking: bool) -> tuple[dict | None, bool]:
    """(verdict, thought) of a judge reply. The verdict is the reply's JSON object with exactly the keys of VERDICT in
    that order, non-empty approaches and a known relation, else None: nothing is repaired or guessed. A thinking judge's
    reply is its reasoning, one `</think>`, then the object; `thought` says whether the reasoning was non-empty."""
    thought = False
    if thinking:
        if text.count("</think>") != 1:
            return None, bool(text.strip())
        reasoning, text = text.split("</think>")
        thought = bool(reasoning.strip())
    try:
        obj = json.loads(text)
    except ValueError:
        return None, thought
    if not (isinstance(obj, dict) and list(obj) == list(VERDICT) and obj["relation"] in RELATIONS
            and all(isinstance(obj[k], str) and obj[k].strip() for k in VERDICT[:2])):
        return None, thought
    return obj, thought


def check_lengths(prompts: list[list[int]], max_tokens: int, max_model_len: int, what: str) -> None:
    longest = max(map(len, prompts), default=0)
    if longest + max_tokens > max_model_len:
        raise SystemExit(f"{what}: the longest prompt has {longest} tokens, which with {max_tokens} new tokens needs "
                         f"max_model_len >= {longest + max_tokens}; the server has {max_model_len}")


# ---------------------------------------------------------------------------------------------------------------
# Serving


def free_mib() -> list[int]:
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpus:
        raise SystemExit("pin the GPU(s) with CUDA_VISIBLE_DEVICES")
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", gpus],
                         capture_output=True, text=True, check=True).stdout
    return [int(x) for x in out.split()]


@contextlib.contextmanager
def serving(m: dict, what: str, base: Path, attach: str | None, overrides: dict):
    """A sampler of model entry `m`: a vLLM server of its own on the visible GPU(s), or the server at `attach`."""
    from evotaste.sampler import Sampler, Server

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)  # two chunks in flight hold up to 1,024 connections
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))
    srv = sampler = None
    if not attach:
        free = free_mib()
        if min(free) < m["need_free_mib"]:
            raise SystemExit(f"GPU(s) {os.environ['CUDA_VISIBLE_DEVICES']} have {free} MiB free; {m['model']} needs "
                             f"{m['need_free_mib']} on each")
        settings = {**m["server"], **overrides}
        srv = Server(m["model"], SimpleNamespace(**settings), None, base / "logs" / f"vllm-{what}-{os.getpid()}.log",
                     max_model_len=settings["max_model_len"], extra_args=tuple(settings["args"]))
        srv.start()
    try:
        if srv:
            srv.wait_ready()
        sampler = Sampler(srv.url if srv else attach, m["model"], f"method-{what}")
        yield sampler
    finally:
        if sampler:
            sampler.close()
        if srv:
            srv.stop()


def _todo(directory: Path, n_chunks: int, shard: str) -> list[int]:
    index, count = (int(x) for x in shard.split("/"))
    return [c for c in range(index, n_chunks, count) if not chunk_path(directory, c).exists()]


def _chunked(n: int, size: int) -> list[list[int]]:
    return [list(range(i, min(i + size, n))) for i in range(0, n, size)]


# ---------------------------------------------------------------------------------------------------------------
# Attempts and alternatives


def references(records: list[dict]) -> dict[str, dict]:
    """uid -> the model's first correct attempt (lowest sample index with reward 1), for items with one."""
    out = {}
    for r in sorted(records, key=lambda r: (r["uid"], r["sample"])):
        if r["reward"] == 1.0 and r["uid"] not in out:
            out[r["uid"]] = r
    return out


def run_generation(config: Path, stage: str, label: str, shard: str, limit: int | None, attach: str | None,
                   overrides: dict) -> None:
    from transformers import AutoTokenizer

    from evotaste.sampler import Request

    cfg = load_config(config)
    if label not in cfg["models"]:
        raise SystemExit(f"--model must be one of {sorted(cfg['models'])}")
    base, m, settings = out_dir(cfg, limit), cfg["models"][label], cfg[stage]
    expected = manifest(cfg, stage, label, limit)
    stored_manifest(base, "items", expected=manifest(cfg, "items", limit=limit))
    tasks = _tasks_of(cfg, read_items(base))
    tok = AutoTokenizer.from_pretrained(m["model"])
    check_thinking_off(tok, m["model"])
    if stage == "attempts":
        jobs = [(t, None) for t in tasks]
        questions = [t.question for t in tasks]
    else:
        refs = references(read_stage(base, "attempts", label, manifest(cfg, "attempts", label, limit)))
        jobs = [(t, refs[t.uid]) for t in tasks if t.uid in refs]
        questions = [alternative_question(settings["prompt"], t, ref["text"]) for t, ref in jobs]
    prompts = [prompt_ids(tok, q) for q in questions]
    check_lengths(prompts, settings["max_tokens"], m["server"]["max_model_len"], f"{stage} of {label}")
    chunked = _chunked(len(jobs), settings["chunk_items"])
    d = stage_dir(base, stage, label)
    open_stage(base, expected, len(jobs), len(chunked))
    todo = _todo(d, len(chunked), shard)
    log.info("%s %s of %s shard %s: %d jobs in %d chunks, %d to do here", base, stage, label, shard, len(jobs),
             len(chunked), len(todo))
    if not todo:
        return
    variant = "default" if stage == "attempts" else "alternative"
    with serving(m, f"{stage}-{label}", base, attach, overrides) as sampler:
        def generate(c: int):
            return sampler.generate([Request(m["model"], prompts[i], settings["samples"], settings["max_tokens"],
                                             stable_seed("method", stage, label, jobs[i][0].uid)) for i in chunked[c]])

        def score(c: int, results) -> list[dict]:
            thinking = sum("<think>" in s.text or "</think>" in s.text for group in results for s in group)
            if thinking:
                raise RuntimeError(f"chunk {c}: {thinking} samples contain a thinking block although the prompt "
                                   "disabled thinking; not writing it")
            flat = [(jobs[i][0], s.text, s.finish_reason) for i, group in zip(chunked[c], results) for s in group]
            scores = iter(score_many(flat))
            out = []
            for i, group in zip(chunked[c], results):
                task, ref = jobs[i]
                records = sample_records(stage, CensusJob(task, len(group), None, variant), group,
                                         [next(scores) for _ in group])
                for r, s in zip(records, group):
                    r |= {"model": label, "logprob_sum": s.logprob, "logprob_mean": s.logprob / len(s.token_ids)}
                    if ref is not None:
                        r["reference"] = ref["sample"]
                out += records
            return out

        run_chunks(lambda c: chunk_path(d, c), todo, generate, score, d / "timing.jsonl",
                   {"stage": stage, "model": label, "shard": shard})


# ---------------------------------------------------------------------------------------------------------------
# Pairs


def _side(model: str, stage: str, sample: int) -> dict:
    return {"model": model, "stage": stage, "sample": sample}


def build_pairs(subjects: list[str], items: list[tuple[str, str]], correct: dict, alternatives: dict, seed: int,
                per_item: int, repeats: int) -> list[dict]:
    """The pairs to judge. items: (uid, domain) in order; correct[label][uid]: sorted sample indices of the label's
    correct attempts; alternatives[label][uid]: (reference sample, first correct alternative sample) for items whose
    alternatives were asked for, the second None when none was correct.

      spontaneous  per item and model, up to `per_item` disjoint pairs of correct attempts, drawn seeded
      requested    per item and model with a correct alternative, the reference attempt and that alternative
      cross        per item and pair of models with correct attempts, the first correct attempt of each
      repeat       `repeats` spontaneous pairs drawn seeded, repeated with the two sides swapped

    The side shown first (a) is a seeded coin flip per pair; a repeat shows the other order of its pair."""
    pairs = []

    def add(kind, pid, uid, domain, x, y):
        if random.Random(stable_seed("method-pairs", seed, "order", pid)).random() < 0.5:
            x, y = y, x
        pairs.append({"id": pid, "kind": kind, "uid": uid, "domain": domain, "a": x, "b": y, "of": None})

    for label in subjects:
        for uid, domain in items:
            ok = list(correct[label].get(uid, []))
            random.Random(stable_seed("method-pairs", seed, "spontaneous", label, uid)).shuffle(ok)
            for k in range(min(per_item, len(ok) // 2)):
                add("spontaneous", f"spontaneous:{label}:{uid}:{k}", uid, domain,
                    _side(label, "attempts", ok[2 * k]), _side(label, "attempts", ok[2 * k + 1]))
            ref, alt = alternatives[label].get(uid, (None, None))
            if alt is not None:
                add("requested", f"requested:{label}:{uid}", uid, domain, _side(label, "attempts", ref),
                    _side(label, "alternatives", alt))
    for l1, l2 in itertools.combinations(subjects, 2):
        for uid, domain in items:
            if correct[l1].get(uid) and correct[l2].get(uid):
                add("cross", f"cross:{l1}+{l2}:{uid}", uid, domain, _side(l1, "attempts", correct[l1][uid][0]),
                    _side(l2, "attempts", correct[l2][uid][0]))
    spontaneous = [p for p in pairs if p["kind"] == "spontaneous"]
    for p in random.Random(stable_seed("method-pairs", seed, "repeat")).sample(spontaneous,
                                                                             min(repeats, len(spontaneous))):
        pairs.append(p | {"id": f"repeat:{p['id']}", "kind": "repeat", "a": p["b"], "b": p["a"], "of": p["id"]})
    return pairs


def run_pairs(config: Path, limit: int | None) -> None:
    cfg = load_config(config)
    base = out_dir(cfg, limit)
    expected = manifest(cfg, "pairs", limit=limit)
    path = base / "pairs.jsonl.gz"
    if path.exists():
        stored_manifest(base, "pairs", expected=expected)
        log.info("%s exists; the pairs are never redrawn", path)
        return
    items = [(r["uid"], r["domain"]) for r in read_items(base)]
    subjects = list(cfg["models"])
    correct, alternatives = {}, {}
    for label in subjects:
        attempts = read_stage(base, "attempts", label, manifest(cfg, "attempts", label, limit))
        alts = read_stage(base, "alternatives", label, manifest(cfg, "alternatives", label, limit))
        correct[label] = {}
        for r in sorted(attempts, key=lambda r: r["sample"]):
            if r["reward"] == 1.0:
                correct[label].setdefault(r["uid"], []).append(r["sample"])
        by_uid = collections.defaultdict(list)
        for r in alts:
            by_uid[r["uid"]].append(r)
        alternatives[label] = {}
        for uid, rs in by_uid.items():
            if rs[0]["reference"] != correct[label][uid][0]:
                raise RuntimeError(f"{label} {uid}: the alternatives' reference is not the first correct attempt")
            good = sorted(r["sample"] for r in rs if r["reward"] == 1.0)
            alternatives[label][uid] = (rs[0]["reference"], good[0] if good else None)
    pairs = build_pairs(subjects, items, correct, alternatives, cfg["pairs"]["seed"], cfg["pairs"]["spontaneous"],
                        cfg["pairs"]["repeats"])
    open_stage(base, expected, len(pairs), 1)
    write_jsonl_gz(path, pairs)
    log.info("wrote %d pairs: %s", len(pairs), dict(collections.Counter(p["kind"] for p in pairs)))


# ---------------------------------------------------------------------------------------------------------------
# Judge and the reader sample


def verdict_record(pair: dict, judge: str, prompt_tokens: int, truncated: tuple[bool, bool], replies,
                   thinking: bool) -> dict:
    verdict, thought = parse_verdict(replies[-1].text, thinking)
    return pair | {"judge": judge, "prompt_tokens": prompt_tokens, "truncated_a": truncated[0],
                   "truncated_b": truncated[1],
                   "replies": [{"text": s.text, "tokens": len(s.token_ids), "finish_reason": s.finish_reason}
                               for s in replies],
                   "completion_tokens": sum(len(s.token_ids) for s in replies), "parsed": verdict is not None,
                   "thought": thought} | (verdict or dict.fromkeys(VERDICT))


def reader_sample(records: list[dict], size: int, seed: int) -> list[dict]:
    """Up to `size` judged pairs (parsed, repeats left out), spread equally over the cells of kind, subject model
    ("cross" for cross pairs), domain and verdict, as far as each cell reaches; seeded, in a seeded order."""
    cells = collections.defaultdict(list)
    for r in records:
        if r["parsed"] and r["kind"] != "repeat":
            model = "cross" if r["kind"] == "cross" else r["a"]["model"]
            cells[(r["kind"], model, r["domain"], r["relation"])].append(r)
    queues = []
    for key in sorted(cells):
        rs = sorted(cells[key], key=lambda r: r["id"])
        random.Random(stable_seed("method-reader", seed, *key)).shuffle(rs)
        queues.append(rs)
    chosen = []
    while len(chosen) < size and any(queues):
        for q in queues:
            if q and len(chosen) < size:
                chosen.append(q.pop())
    random.Random(stable_seed("method-reader", seed, "order")).shuffle(chosen)
    return chosen


def reader_instructions(j: dict) -> str:
    """What independent readers are told: the judge's own definitions, and how to label."""
    methods = "\n\n".join(f"{d}: {j['method'][d].strip()}" for d in DOMAINS)
    return (f"Each line of reader-sample.jsonl gives a problem and two correct solutions to it. Decide whether the two "
            f"solve it by the same method.\n\nWhat counts as the method, by domain:\n\n{methods}\n\nThe relation is "
            f"one of:\n{j['relations'].strip()}\n\nWrite one JSON line per pair to a labels file: "
            '{"reader_id": "r001", "relation": "same" | "variant" | "different", "reader": "<your name>"}\n')


def write_reader_sample(base: Path, cfg: dict, records: list[dict], tasks: dict[str, Task], texts: dict) -> None:
    j = cfg["judge"]
    chosen = reader_sample(records, j["reader_sample"], cfg["pairs"]["seed"])
    sample, key = [], []
    for n, r in enumerate(chosen, 1):
        rid = f"r{n:03d}"
        sample.append({"reader_id": rid, "domain": r["domain"], "problem": tasks[r["uid"]].question,
                       "solution_a": texts[_text_key(r["a"], r["uid"])],
                       "solution_b": texts[_text_key(r["b"], r["uid"])]})
        key.append({"reader_id": rid, "pair": r["id"], "kind": r["kind"], "model_a": r["a"]["model"],
                    "model_b": r["b"]["model"], "relation": r["relation"]})
    atomic_write(base / "reader-key.jsonl", "".join(json.dumps(k) + "\n" for k in key).encode())
    atomic_write(base / "reader-instructions.txt", reader_instructions(j).encode())
    atomic_write(base / "reader-sample.jsonl", "".join(json.dumps(s) + "\n" for s in sample).encode())
    log.info("wrote the reader sample: %d pairs", len(sample))


def _text_key(side: dict, uid: str) -> tuple:
    return side["model"], side["stage"], uid, side["sample"]


def run_judge(config: Path, shard: str, limit: int | None, attach: str | None, overrides: dict) -> None:
    from transformers import AutoTokenizer

    from evotaste.sampler import Request

    cfg = load_config(config)
    base, j = out_dir(cfg, limit), cfg["judge"]
    m = cfg["models"][j["model"]]
    expected = manifest(cfg, "judge", limit=limit)
    stored_manifest(base, "pairs", expected=manifest(cfg, "pairs", limit=limit))
    pairs = read_jsonl_gz(base / "pairs.jsonl.gz")
    items = read_items(base)
    tasks = dict(zip((r["uid"] for r in items), _tasks_of(cfg, items)))
    texts = {}
    for label in cfg["models"]:
        for stage in ("attempts", "alternatives"):
            for r in read_stage(base, stage, label, manifest(cfg, stage, label, limit)):
                texts[(label, stage, r["uid"], r["sample"])] = r["text"]
    tok = AutoTokenizer.from_pretrained(m["model"])
    prompts, truncated = [], []
    for p in pairs:
        a, cut_a = truncate(tok, texts[_text_key(p["a"], p["uid"])], j["solution_tokens"])
        b, cut_b = truncate(tok, texts[_text_key(p["b"], p["uid"])], j["solution_tokens"])
        prompts.append(judge_ids(tok, j, judge_text(j, tasks[p["uid"]].question, a, b, p["domain"])))
        truncated.append((cut_a, cut_b))
    check_lengths(prompts, 2 * j["max_tokens"], m["server"]["max_model_len"], "judge (with its retry)")
    chunked = _chunked(len(pairs), j["chunk_pairs"])
    d = base / "judge"
    open_stage(base, expected, len(pairs), len(chunked))
    todo = _todo(d, len(chunked), shard)
    log.info("%s judge shard %s: %d pairs in %d chunks, %d to do here", base, shard, len(pairs), len(chunked),
             len(todo))
    thinking = bool(j["thinking"])
    if todo:
        with serving(m, "judge", base, attach, overrides) as sampler:
            def request(i: int, attempt: int) -> Request:
                return Request(m["model"], prompts[i], 1, j["max_tokens"] * attempt,
                               stable_seed("method", "judge", pairs[i]["id"], attempt), temperature=0.0,
                               json_schema=SCHEMA)

            def generate(c: int):
                """Every pair once; a reply that does not parse once more, with twice the token budget."""
                replies = {i: list(group) for i, group in zip(chunked[c], sampler.generate(
                    [request(i, 1) for i in chunked[c]]))}
                retry = [i for i in chunked[c] if parse_verdict(replies[i][0].text, thinking)[0] is None]
                if retry:
                    log.info("chunk %d: %d replies did not parse; asking once more", c, len(retry))
                    for i, group in zip(retry, sampler.generate([request(i, 2) for i in retry])):
                        replies[i] += group
                return [replies[i] for i in chunked[c]]

            def score(c: int, results) -> list[dict]:
                return [verdict_record(pairs[i], j["model"], len(prompts[i]), truncated[i], replies, thinking)
                        for i, replies in zip(chunked[c], results)]

            run_chunks(lambda c: chunk_path(d, c), todo, generate, score, d / "timing.jsonl",
                       {"stage": "judge", "model": j["model"], "shard": shard})
    if not (base / "reader-sample.jsonl").exists() and not _todo(d, len(chunked), "0/1"):
        write_reader_sample(base, cfg, read_stage(base, "judge", expected=expected), tasks, texts)


# ---------------------------------------------------------------------------------------------------------------
# Command line


def main(argv=None) -> None:
    from evotaste.worker import log_uncommitted_changes, logged, server_options, server_overrides, setup_logging

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("config", type=Path)
    ap.add_argument("--model", help="subject model label (attempts, alternatives)")
    ap.add_argument("--shard", default="0/1", help="i/n: this process runs the chunks whose index is i mod n")
    ap.add_argument("--limit", type=int, help="smoke run: the first N drawn items (output under <name>-limitN)")
    ap.add_argument("--attach", help="URL of a running vLLM server of the stage's model (development)")
    server_options(ap)
    args = ap.parse_args(argv)
    if (args.model is not None) != (args.stage in ("attempts", "alternatives")):
        ap.error("--model is required for attempts and alternatives, and only there")
    name = load_config(args.config)["name"] + (f"-limit{args.limit}" if args.limit else "")
    label = "-".join(x for x in (args.stage, args.model, args.shard.replace("/", "of")) if x)
    setup_logging(f"method-{name}", label)
    log_uncommitted_changes(f"method-{name}")  # verifier processes run whatever code is on disk
    if args.stage == "items":
        logged(run_items, args.config, args.limit)
    elif args.stage == "pairs":
        logged(run_pairs, args.config, args.limit)
    elif args.stage == "judge":
        logged(run_judge, args.config, args.shard, args.limit, args.attach, server_overrides(args))
    else:
        logged(run_generation, args.config, args.stage, args.model, args.shard, args.limit, args.attach,
               server_overrides(args))


if __name__ == "__main__":
    main()
