"""Evaluation summaries: one adapter per label, and the full (family, instance, sample) grid, or no summary."""

import pytest

from evotaste.evaluate import IncompleteEvaluation, eval_params, same_params, summarize
from evotaste.tasks import Task

TASKS = [Task(f"fam/0/{i}", "fam", "q", "a", entry={}) for i in range(3)] + [Task("gsm8k_train/1", "gsm8k_train", "q", "1")]


def records(label="m0", sha="h", samples=2, drop=0):
    rs = [{"adapter": label, "adapter_sha256": sha, "family": t.family, "uid": t.uid, "sample": i,
           "reward": float(i == 0), "parsed": True, "truncated": False, "completion_tokens": 5}
          for t in TASKS for i in range(samples)]
    return rs[:len(rs) - drop]


def test_complete_grid_is_summarized():
    s = summarize(records() + records("base", None), TASKS, 2)
    assert s["m0"]["n"] == 8 and s["m0"]["macro"] == 0.5 and s["m0"]["by_family"]["gsm8k_train"] == 0.5
    assert s["base"]["adapter_sha256"] is None


def test_incomplete_grid_is_not_summarized():
    with pytest.raises(IncompleteEvaluation, match="m0: 7 of 8"):
        summarize(records(drop=1) + records("base", None), TASKS, 2)


def test_one_adapter_per_label():
    mixed = records()
    mixed[-1] = {**mixed[-1], "adapter_sha256": "other"}  # a shard that resolved m0 to another checkpoint
    with pytest.raises(ValueError, match="2 different adapters"):
        summarize(mixed, TASKS, 2)


def test_no_duplicates_or_foreign_records():
    with pytest.raises(ValueError, match="duplicate"):
        summarize(records() + records()[:1], TASKS, 2)
    with pytest.raises(ValueError, match="outside"):
        summarize(records(samples=3), TASKS, 2)


def test_external_sources_can_have_more_items_than_families():
    tasks = TASKS + [Task(f"gsm8k_train/{i}", "gsm8k_train", "q", "1") for i in range(2, 6)]  # 3 per family, 5 items
    full = [{"adapter": "m0", "adapter_sha256": "h", "family": t.family, "uid": t.uid, "sample": 0, "reward": 1.0,
             "parsed": True, "truncated": False, "completion_tokens": 5} for t in tasks]
    assert summarize(full, tasks, 1)["m0"]["n_by_family"] == {"fam": 3, "gsm8k_train": 5}
    with pytest.raises(IncompleteEvaluation, match="m0: 7 of 8"):
        summarize(full[:-1], tasks, 1)


def test_per_source_params_and_tag():
    params, tag = eval_params("selection", 50, 1, None)
    assert tag == "n50x1" and params["per_source"] == 50  # default: behaviour and tag as before
    params, tag = eval_params("selection", 50, 1, 500)
    assert tag == "n50x1s500" and params["per_source"] == 500
    old = {"suite": "selection", "per_family": 50, "samples": 1}  # params.json written before per_source existed
    assert same_params(old, eval_params("selection", 50, 1, None)[0])
    assert not same_params(old, eval_params("selection", 50, 1, 500)[0])
    with pytest.raises(SystemExit):
        eval_params("selection", 50, 1, 0)
