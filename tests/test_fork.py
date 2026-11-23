"""Forking a run after seeding: what is refused, what is reused, and the state the new run starts from."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from evotaste import config, worker
from evotaste.records import append_jsonl, read_json, read_jsonl, write_json, write_jsonl_gz

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = [{"id": "a", "arm": "plain", "seed": 1}, {"id": "b", "arm": "scaffold", "scaffold": "verify", "seed": 2},
           {"id": "c", "arm": "plain", "seed": 3}, {"id": "d", "arm": "plain", "seed": 4}]


def base_config(name, members=MEMBERS, **sections):
    return {"name": name, "model": "m", "seed": 7, "seeding": {"steps": 2, "pool_prompts": 16},
            "amplification": {"steps": 5}, "members": members, **sections}


def write_config(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


class StubRun:  # the probe-set check needs the model's tokenizer and datasets; covered elsewhere
    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.cfg = config.load(self.dir / "config.yaml")

    def check_probe_set(self):
        pass


@pytest.fixture
def source(tmp_path, monkeypatch):
    """runs/src: prepared, a and b seeded (active at step 5), c still seeding, d seeded."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(worker, "Run", StubRun)
    (tmp_path / "configs").mkdir()
    for name in ("suites.yaml", "scaffolds.yaml"):
        shutil.copyfile(ROOT / "configs" / name, tmp_path / "configs" / name)
    src = worker.freeze(write_config(tmp_path / "src.yaml", base_config("src")))
    write_json(src / "prepared.json", {"revision": "abc123", "teachers": ["plain", "verify"]})
    for teacher in ("plain", "verify"):
        write_jsonl_gz(src / "seeding" / f"{teacher}.jsonl.gz", [{"uid": "x"}])
        write_json(src / "seeding" / f"{teacher}.json", {"teacher": teacher})
    write_jsonl_gz(src / "base" / "probe_responses.jsonl.gz", [{"uid": "x"}])
    write_json(src / "base" / "probe_summary.json", {"accuracy": {}})
    for m in MEMBERS:
        d = src / "members" / m["id"]
        if m["id"] == "c":
            write_json(d / "state.json", {"id": "c", "slot": "c", "generation": 0, "arm": "plain", "scaffold": None,
                                          "seed": 3, "status": "new", "step": 0, "probes": []})
            continue
        probes = [{"step": 0, "accuracy": {"train": 0.1}, "cull": None}, {"step": 5, "accuracy": {"train": 0.2}}]
        write_json(d / "state.json", {"id": m["id"], "slot": m["id"], "generation": 0, "arm": m["arm"],
                                      "scaffold": m.get("scaffold"), "seed": m["seed"], "status": "active",
                                      "step": 5, "probes": probes})
        (d / "checkpoints" / "step_000000").mkdir(parents=True)
        (d / "checkpoints" / "step_000000" / "adapter_model.safetensors").write_bytes(b"w" + m["id"].encode())
        (d / "checkpoints" / "step_000000" / "adapter_config.json").write_text('{"r": 32}')
        write_jsonl_gz(d / "probe_responses" / "step_000000.jsonl.gz", [{"step": 0}])
        for i in range(2):
            append_jsonl(d / "history.jsonl", {"phase": "seed", "step": i, "loss": 0.5})
        for t in range(5):
            append_jsonl(d / "history.jsonl", {"phase": "amp", "step": t})
    return src


def new_config(tmp_path, **changes):
    members = [MEMBERS[0], {**MEMBERS[1], "overrides": {"amplification": {"lr": 3e-5}}}, MEMBERS[2],
               {**MEMBERS[3], "seed": 40}, {"id": "e", "arm": "perturb", "seed": 5}]
    data = base_config("new", members, culling={"grace": 10}, server={"kv_cache": "12G"},
                       amplification={"steps": 5, "truncation_penalty": 1.0})
    data.update(changes)
    return write_config(tmp_path / "new.yaml", data)


def test_fork_reuses_completed_seeding_only(source, tmp_path):
    worker.fork(source, new_config(tmp_path), dry_run=False)
    new = Path("runs/new")
    fork = read_json(new / "prepared.json")["fork"]
    assert fork == {"source": "runs/src", "source_revision": "abc123", "members": ["a", "b"]}
    assert (new / "seeding" / "verify.jsonl.gz").read_bytes() == (source / "seeding" / "verify.jsonl.gz").read_bytes()
    assert (new / "base" / "probe_summary.json").exists()
    for m in ("a", "b"):
        d = new / "members" / m
        state = read_json(d / "state.json")
        assert (state["status"], state["step"], [p["step"] for p in state["probes"]]) == ("active", 0, [0])
        assert [r["phase"] for r in read_jsonl(d / "history.jsonl")] == ["seed", "seed"]
        assert [r["step"] for r in read_jsonl(d / "probes.jsonl")] == [0]
        ckpt = d / "checkpoints" / "step_000000" / "adapter_model.safetensors"
        assert ckpt.read_bytes() == b"w" + m.encode() and not ckpt.is_symlink()
        assert sorted(p.name for p in (d / "checkpoints").iterdir()) == ["step_000000"]
    assert not any((new / "members" / m).exists() for m in ("c", "d", "e"))  # still seeding / other seed / new
    assert read_json(source / "members" / "a" / "state.json")["step"] == 5  # the source is untouched


def test_dry_run_creates_nothing(source, tmp_path, capsys):
    worker.fork(source, new_config(tmp_path), dry_run=True)
    assert not Path("runs/new").exists()
    out = capsys.readouterr().out
    assert "step-0 checkpoint (2, 0 MiB): a, b" in out and "seed normally: c" in out and "seed normally: d" in out


@pytest.mark.parametrize("changes", [{"model": "other"}, {"seed": 8}, {"lora": {"rank": 16}},
                                     {"seeding": {"steps": 3, "pool_prompts": 16}}, {"probes": {"every": 5}},
                                     {"trainer": {"weight_decay": 0.1}}, {"name": "src"}])
def test_fork_refuses_differing_artefact_settings(source, tmp_path, changes):
    with pytest.raises(SystemExit):
        worker.fork(source, new_config(tmp_path, **changes), dry_run=True)


def test_fork_refuses_other_suites_or_missing_pools(source, tmp_path):
    suites = tmp_path / "configs" / "suites.yaml"
    suites.write_text(suites.read_text() + "\n# edited\n")
    with pytest.raises(SystemExit, match="suites content"):
        worker.fork(source, new_config(tmp_path), dry_run=True)
    shutil.copyfile(source / "suites.yaml", suites)
    members = MEMBERS + [{"id": "f", "arm": "scaffold", "scaffold": "terse", "seed": 6}]
    with pytest.raises(SystemExit, match="terse"):
        worker.fork(source, new_config(tmp_path, members=members), dry_run=True)
    assert json.loads((source / "prepared.json").read_text())["revision"] == "abc123"
