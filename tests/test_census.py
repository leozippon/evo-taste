"""The census cannot reach the vault, and runs resumably in shards (generation replaced by a fake sampler)."""

import ast
import json
import pathlib

import pytest
import yaml

from evotaste import census, external
from evotaste.records import read_jsonl_gz
from evotaste.tasks import Suites


def config(tmp_path, **parts) -> pathlib.Path:
    cfg = yaml.safe_load(open("configs/census.yaml"))
    cfg["name"] = f"test-{tmp_path.name}"
    cfg["chunk_items"] = 7
    for part, settings in parts.items():
        cfg["parts"][part].update(settings)
    path = tmp_path / "census.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def chdir_results(tmp_path, monkeypatch):
    """Census output under tmp_path: results/ is a relative path."""
    root = pathlib.Path.cwd()
    monkeypatch.chdir(tmp_path)
    for name in ("configs", "evotaste"):
        (tmp_path / name).symlink_to(root / name)
    return tmp_path


def test_census_never_asks_for_the_vault(tmp_path, monkeypatch):
    """Every part's jobs come from the open parts only; the module names no receipt, guard or vault."""
    asked = []
    real = external.items

    def spy(source, part, receipt=None, suites=external.SUITES):
        asked.append(part)
        assert receipt is None
        return real(source, part, receipt, suites)

    monkeypatch.setattr(external, "items", spy)
    cfg_path = config(tmp_path)
    cfg, suites = census.load_config(cfg_path), Suites("configs/suites-r2.yaml")
    jobs = {p: census.jobs_of(p, cfg, suites, 5) for p in census.PARTS}
    assert set(asked) <= {"train", "selection"} and asked
    open_uids = census._open_uids(suites)
    for part, js in jobs.items():
        assert js, part
        outside = [j for j in js if j.task.uid not in open_uids and j.task.family not in suites.procedural]
        assert not outside, part
    assert len({j.context for j in jobs["contexts"]}) == 12
    tree = ast.parse(pathlib.Path("evotaste/census.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not names & {"BlindReceipt", "blind_guard", "blind_suite", "check_receipt", "VAULT"}
    assert "blind2" not in pathlib.Path("evotaste/census.py").read_text()


class FakeSampler:
    calls = []

    def __init__(self, url, model, run_name):
        pass

    def generate(self, requests):
        FakeSampler.calls.append(len(requests))
        from evotaste.sampler import Sample

        return [[Sample("The answer.\n\\boxed{1}", [1, 2, 3], "stop", [-0.1, -0.1, -0.1]) for _ in range(r.n)]
                for r in requests]

    def close(self):
        pass


def test_run_is_resumable_and_sharded(chdir_results, monkeypatch):
    import evotaste.sampler

    monkeypatch.setattr(evotaste.sampler, "Sampler", FakeSampler)
    cfg_path = config(chdir_results, selection={"samples": 2, "per_family": 1})
    cfg = census.load_config(cfg_path)
    base = census.out_dir(cfg, 3)
    jobs = census.jobs_of("selection", cfg, Suites("configs/suites-r2.yaml"), 3)
    n_chunks = len(census.chunks(jobs, cfg["chunk_items"]))
    assert n_chunks >= 4
    census.run_part(cfg_path, "selection", "1/3", 3, "http://unused", {})
    done = sorted(int(p.name[6:11]) for p in (base / "selection").glob("chunk-*.jsonl.gz"))
    assert done == list(range(1, n_chunks, 3))
    FakeSampler.calls.clear()
    census.run_part(cfg_path, "selection", "1/3", 3, "http://unused", {})  # a rerun of a finished shard does nothing
    assert FakeSampler.calls == []
    for i in (0, 2):
        census.run_part(cfg_path, "selection", f"{i}/3", 3, "http://unused", {})
    records = census.read_part(base, "selection", n_chunks)
    assert len(records) == 2 * len(jobs)
    assert {(r["uid"], r["sample"]) for r in records} == {(j.task.uid, k) for j in jobs for k in range(2)}
    first = read_jsonl_gz(census.chunk_path(base, "selection", 0))[0]
    assert {"reward", "correct_any", "complied", "channel", "truncated", "completion_tokens", "text"} <= set(first)
    timing = [json.loads(line) for line in (base / "selection" / "timing.jsonl").read_text().splitlines()]
    assert sorted(t["chunk"] for t in timing) == list(range(n_chunks))
    with pytest.raises(SystemExit, match="incomplete"):
        census.read_part(base, "train", 1)


def test_records_under_other_or_unknown_verifier_rules_are_not_continued(chdir_results, monkeypatch):
    import evotaste.sampler
    from evotaste.verify import RULES_VERSION

    monkeypatch.setattr(evotaste.sampler, "Sampler", FakeSampler)
    cfg_path = config(chdir_results, selection={"samples": 1, "per_family": 1})
    base = census.out_dir(census.load_config(cfg_path), 2)
    census.run_part(cfg_path, "selection", "0/2", 2, "http://unused", {})
    assert (base / "verifier_rules").read_text() == RULES_VERSION + "\n"
    written = sorted((base / "selection").glob("chunk-*.jsonl.gz"))
    (base / "verifier_rules").write_text("2026-11-25.audit-2\n")
    with pytest.raises(SystemExit, match="scored under verifier rules 2026-11-25.audit-2"):
        census.run_part(cfg_path, "selection", "1/2", 2, "http://unused", {})
    (base / "verifier_rules").unlink()  # a census from before the rules were recorded
    with pytest.raises(SystemExit, match="unknown verifier rules"):
        census.run_part(cfg_path, "selection", "1/2", 2, "http://unused", {})
    assert written and sorted((base / "selection").glob("chunk-*.jsonl.gz")) == written  # nothing was added
