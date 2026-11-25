"""Forking a run after seeding: what is refused, what is reused, and the state the new run starts from."""

import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from evotaste import config, worker
from evotaste.records import append_jsonl, read_json, read_jsonl, write_json, write_jsonl_gz

ROOT = Path(__file__).resolve().parents[1]
STRATA_TEXT = b"uid\tpasses\tsamples\nmath_train/1\t3\t8\n"
STRATA_SHA = hashlib.sha256(STRATA_TEXT).hexdigest()
MEMBERS = [{"id": "a", "context": "refute", "seed": 1}, {"id": "b", "context": "verify", "seed": 2},
           {"id": "c", "context": "refute", "seed": 3}, {"id": "d", "context": "refute", "seed": 4}]


def base_config(name, members=MEMBERS, **sections):
    return {"name": name, "model": "m", "seed": 7, "seeding": {"steps": 2}, "amplification": {"steps": 5},
            "strata": {"path": "strata.tsv", "sha256": STRATA_SHA}, "members": members, **sections}


def write_config(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


class StubRun:  # the probe-set check needs the model's tokenizer and the task layer; covered elsewhere
    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.cfg = config.load(self.dir / "config.yaml")
        self.contexts = config.load_contexts(self.cfg.contexts)

    def check_probe_set(self):
        pass


@pytest.fixture
def source(tmp_path, monkeypatch):
    """runs/src: prepared, a and b seeded (active at step 5), c still seeding, d seeded."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(worker, "Run", StubRun)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "suites-r2.yaml").write_text("suites: {train: [math_train]}\n")
    shutil.copyfile(ROOT / "configs" / "contexts.yaml", tmp_path / "configs" / "contexts.yaml")
    (tmp_path / "strata.tsv").write_bytes(STRATA_TEXT)
    src = worker.freeze(write_config(tmp_path / "src.yaml", base_config("src")))
    write_json(src / "prepared.json", {"revision": "abc123", "contexts": ["refute"]})
    write_jsonl_gz(src / "base" / "probe_responses.jsonl.gz", [{"uid": "x"}])
    write_json(src / "base" / "probe_summary.json", {"accuracy": 0.5})
    write_jsonl_gz(src / "base" / "contexts" / "refute.jsonl.gz", [{"uid": "x"}])
    write_json(src / "base" / "contexts" / "refute.json", {"accuracy": 0.4})
    for m in MEMBERS:
        d = src / "members" / m["id"]
        if m["id"] == "c":
            write_json(d / "state.json", {"id": "c", "slot": "c", "generation": 0, "context": "refute", "seed": 3,
                                          "status": "new", "step": 0, "probes": []})
            continue
        probes = [{"step": 0, "accuracy": 0.1, "cull": None}, {"step": 5, "accuracy": 0.2}]
        write_json(d / "state.json", {"id": m["id"], "slot": m["id"], "generation": 0, "context": m["context"],
                                      "seed": m["seed"], "parent": None, "status": "active", "step": 5,
                                      "probes": probes})
        (d / "checkpoints" / "step_000000").mkdir(parents=True)
        (d / "checkpoints" / "step_000000" / "adapter_model.safetensors").write_bytes(b"w" + m["id"].encode())
        (d / "checkpoints" / "step_000000" / "adapter_config.json").write_text('{"r": 32}')
        write_jsonl_gz(d / "probe_responses" / "step_000000.jsonl.gz", [{"step": 0}])
        for i in range(2):
            append_jsonl(d / "history.jsonl", {"phase": "seed", "step": i, "kl": 0.5})
        for t in range(5):
            append_jsonl(d / "history.jsonl", {"phase": "amp", "step": t})
    return src


def new_config(tmp_path, **changes):
    members = [MEMBERS[0], {**MEMBERS[1], "overrides": {"amplification": {"lr": 3e-5}}}, MEMBERS[2],
               {**MEMBERS[3], "seed": 40}, {"id": "e", "seed": 5, "overrides": {"seeding": {"method": "none"}}},
               {"id": "f", "seed": 6, "parent": {"member": "a", "step": 0}}]
    data = base_config("new", members, culling={"grace": 10}, server={"kv_cache": "12G"},
                       amplification={"steps": 5, "self_imitation": 0.5})
    data.update(changes)
    return write_config(tmp_path / "new.yaml", data)


def test_fork_reuses_completed_seeding_only(source, tmp_path):
    worker.fork(source, new_config(tmp_path), dry_run=False)
    new = Path("runs/new")
    assert (new / "strata.tsv").read_bytes() == STRATA_TEXT  # frozen with the run, like suites and contexts
    assert config.load(new / "config.yaml").strata.path == "runs/new/strata.tsv"
    fork = read_json(new / "prepared.json")["fork"]
    assert fork == {"source": "runs/src", "source_revision": "abc123", "members": ["a", "b"]}
    for f in ("probe_responses.jsonl.gz", "probe_summary.json", "contexts/refute.jsonl.gz", "contexts/refute.json"):
        assert (new / "base" / f).read_bytes() == (source / "base" / f).read_bytes()
    for m in ("a", "b"):
        d = new / "members" / m
        state = read_json(d / "state.json")
        assert (state["status"], state["step"], [p["step"] for p in state["probes"]]) == ("active", 0, [0])
        assert state["context"] == {"a": "refute", "b": "verify"}[m] and state["parent"] is None
        assert [r["phase"] for r in read_jsonl(d / "history.jsonl")] == ["seed", "seed"]
        assert [r["step"] for r in read_jsonl(d / "probes.jsonl")] == [0]
        ckpt = d / "checkpoints" / "step_000000" / "adapter_model.safetensors"
        assert ckpt.read_bytes() == b"w" + m.encode() and not ckpt.is_symlink()
        assert sorted(p.name for p in (d / "checkpoints").iterdir()) == ["step_000000"]
    assert not any((new / "members" / m).exists() for m in ("c", "d", "e", "f"))  # seeding / other seed / new / clone
    assert read_json(source / "members" / "a" / "state.json")["step"] == 5  # the source is untouched


def test_dry_run_creates_nothing(source, tmp_path, capsys):
    worker.fork(source, new_config(tmp_path), dry_run=True)
    assert not Path("runs/new").exists()
    out = capsys.readouterr().out
    assert "step-0 checkpoint (2, 0 MiB): a, b" in out and "copy 4 base reference files" in out
    for m, why in (("c", "still seeding"), ("d", "seed differ"), ("e", "not in the source"), ("f", "a clone")):
        assert f"start normally: {m} (" in out and why in out


@pytest.mark.parametrize("changes", [{"model": "other"}, {"seed": 8}, {"lora": {"rank": 16}},
                                     {"seeding": {"steps": 3}}, {"probes": {"every": 5}},
                                     {"trainer": {"weight_decay": 0.1}}, {"name": "src"}])
def test_fork_refuses_differing_artefact_settings(source, tmp_path, changes):
    with pytest.raises(SystemExit):
        worker.fork(source, new_config(tmp_path, **changes), dry_run=True)


def test_freeze_refuses_a_strata_table_without_the_pinned_hash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "suites-r2.yaml").write_text("suites: {}\n")
    shutil.copyfile(ROOT / "configs" / "contexts.yaml", tmp_path / "configs" / "contexts.yaml")
    (tmp_path / "strata.tsv").write_bytes(STRATA_TEXT + b"math_train/2\t1\t8\n")
    with pytest.raises(SystemExit, match="pinned SHA-256"):
        worker.freeze(write_config(tmp_path / "x.yaml", base_config("x")))
    assert not (tmp_path / "runs" / "x").exists()


def test_fork_refuses_other_contexts_or_an_unprepared_source(source, tmp_path):
    contexts = tmp_path / "configs" / "contexts.yaml"
    contexts.write_text(contexts.read_text() + "\n# edited\n")
    with pytest.raises(SystemExit, match="contexts content"):
        worker.fork(source, new_config(tmp_path), dry_run=True)
    shutil.copyfile(source / "contexts.yaml", contexts)
    (source / "prepared.json").unlink()
    with pytest.raises(SystemExit, match="not prepared"):
        worker.fork(source, new_config(tmp_path), dry_run=True)
