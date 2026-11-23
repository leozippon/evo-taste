"""Harvest at a common step: who is ranked, what is refused, and how the analysis reads <lineage>@<step> labels."""

import json
from types import SimpleNamespace

import pytest

from evotaste import config
from evotaste.analysis.harvest import adapter_roles
from evotaste.evaluate import step_harvest
from evotaste.records import write_json
from evotaste.trainer import adapter_sha256

MEMBERS = [{"id": "a", "arm": "plain", "seed": 1}, {"id": "b", "arm": "plain", "seed": 2},
           {"id": "c", "arm": "plain", "seed": 3}, {"id": "bl", "arm": "plain", "seed": 9, "role": "baseline"}]


def lineage(run_dir, lid, slot, status="active", step=30, replaced_by=None, checkpoints=(15, 30)):
    d = run_dir / "members" / lid
    write_json(d / "state.json", {"id": lid, "slot": slot, "generation": int("~" in lid), "arm": "plain",
                                  "scaffold": None, "seed": 1, "status": status, "step": step, "probes": [],
                                  "cull_reason": "test" if status == "culled" else None, "replaced_by": replaced_by})
    for s in checkpoints:
        c = d / "checkpoints" / f"step_{s:06d}"
        c.mkdir(parents=True)
        (c / "adapter_config.json").write_text('{"r": 32}')
        (c / "adapter_model.safetensors").write_bytes(f"{lid}@{s}".encode())


@pytest.fixture
def run(tmp_path):
    cfg = config.parse({"name": "r", "model": "m", "seed": 0, "members": MEMBERS}, set())
    lineage(tmp_path, "a", "a", status="done")
    lineage(tmp_path, "b", "b")
    lineage(tmp_path, "c", "c", status="culled", step=30, replaced_by="c~1")  # culled after step 15
    lineage(tmp_path, "c~1", "c", step=15, checkpoints=(0, 15))
    lineage(tmp_path, "bl", "bl", step=60, checkpoints=(15, 30, 60))
    return SimpleNamespace(dir=tmp_path, cfg=cfg)


def summary_for(run, labels, macro=0.3):
    out = {"base": {"adapter_sha256": None, "macro": 0.1}}
    for i, label in enumerate(labels):
        lid, _, s = label.partition("@")
        path = run.dir / "members" / lid / "checkpoints" / f"step_{int(s):06d}"
        out[label] = {"adapter_sha256": adapter_sha256(path), "macro": macro + i / 100}
    return out


def test_step_harvest_ranks_current_lineages_and_lists_baselines(run):
    summary = summary_for(run, ["a@15", "b@15", "c~1@15", "bl@15", "bl@30", "bl@60"])
    ranking, baselines, problems = step_harvest(run, summary, 15)
    assert problems == []
    assert [e["label"] for e in ranking] == ["a@15", "b@15", "c~1@15"]  # c's slot: its current lineage
    assert all(e["step"] == 15 and len(e["sha256"]) == 64 for e in ranking)
    assert [(e["label"], e["step"]) for e in baselines] == [("bl@15", 15), ("bl@30", 30), ("bl@60", 60)]
    assert all(e["lineage"] != "bl" for e in ranking)  # baselines are never ranked


def test_step_harvest_reports_every_gap(run):
    ranking, _, problems = step_harvest(run, summary_for(run, ["a@15", "bl@30"]), 30)
    assert [e["label"] for e in ranking] == []
    text = "\n".join(problems)
    assert "a: a@30 is not evaluated" in text and "b: b@30 is not evaluated" in text
    assert "c: c~1 has no checkpoint at step 30" in text
    wrong = summary_for(run, ["a@15", "b@15", "c~1@15"])
    wrong["b@15"]["adapter_sha256"] = "0" * 64
    assert any("differs" in p for p in step_harvest(run, wrong, 15)[2])


def test_exhausted_slot_is_reported(run):
    write_json(run.dir / "members" / "c~1" / "state.json",
               {**json.loads((run.dir / "members" / "c~1" / "state.json").read_text()), "status": "culled",
                "cull_reason": "test"})
    problems = step_harvest(run, summary_for(run, ["a@15", "b@15", "c~1@15"]), 15)[2]
    assert problems == ["c: is exhausted (c~1 culled: test)"]


def test_analysis_reads_step_labels():
    specs = {m["id"]: {"arm": m["arm"], "scaffold": None, "role": m.get("role", "member")} for m in MEMBERS}
    roles = adapter_roles(["base", "a@15", "b@15", "c~1@15", "bl@15", "bl@60"], specs)
    assert {a: r["role"] for a, r in roles.items()} == {"base": "base", "a@15": "member", "b@15": "member",
                                                       "c~1@15": "member", "bl@15": "baseline", "bl@60": "baseline"}
    roles = adapter_roles(["a", "a@15", "b@15", "b@30"], specs)  # several labels of a lineage: checkpoints
    assert {a: r["role"] for a, r in roles.items()} == {"a": "member", "a@15": "checkpoint", "b@15": "checkpoint",
                                                       "b@30": "checkpoint"}
