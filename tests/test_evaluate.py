"""Evaluation summaries: one adapter per label, and the full (family, instance, sample) grid, or no summary."""

import pytest

from evotaste.evaluate import IncompleteEvaluation, summarize
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
