"""Resume from a checkpoint: the next update is identical, and records of steps after the checkpoint are dropped."""

import copy
from types import SimpleNamespace

import pytest
import torch

from evotaste import config
from evotaste.member import Lineage, Member, lineage_dir
from evotaste.records import append_jsonl, read_jsonl, write_json, write_jsonl_gz
from evotaste.trainer import Sequence
from tests.conftest import make_policy, random_groups


def _fake_run(tmp_path, policy):
    cfg = config.parse({"name": "t", "model": "tiny", "seed": 0,
                        "members": [{"id": "m0", "arm": "plain", "seed": 1}]}, scaffold_names=set())
    return SimpleNamespace(dir=tmp_path, cfg=cfg, policy=policy)


def test_resume_reproduces_training_state(tmp_path, base_model):
    policy = make_policy(base_model)
    run = _fake_run(tmp_path, policy)
    seqs_a, seqs_b = random_groups(seed=1), random_groups(seed=2)
    member = Member(run, Lineage("m0", "m0", 0, "plain", None, 1, status="active", step=4))
    member.weights = policy.new_member(seed=1, b_std=0.05)
    policy.update(member.weights, seqs_a, lr=1e-3)
    member.write_checkpoint()
    write_json(member.dir / "state.json", vars(member.lin))
    uninterrupted = copy.deepcopy(member.weights)
    policy.update(uninterrupted, seqs_b, lr=1e-3)

    # Records written after the checkpoint, as by a worker killed during step 5.
    for i in range(3):
        append_jsonl(member.dir / "history.jsonl", {"phase": "seed", "step": i})
    for t in range(6):
        append_jsonl(member.dir / "history.jsonl", {"phase": "amp", "step": t})
        write_jsonl_gz(member.dir / "rollouts" / f"step_{t:06d}.jsonl.gz", [{"step": t}])
    for s in (0, 4, 5):
        append_jsonl(member.dir / "probes.jsonl", {"step": s})
        write_jsonl_gz(member.dir / "probe_responses" / f"step_{s:06d}.jsonl.gz", [{"step": s}])
    (member.checkpoint_dir(5).with_name("step_000005.tmp")).mkdir()
    with open(member.dir / "history.jsonl", "a") as f:
        f.write('{"phase": "amp", "st')  # a line cut short by the kill

    resumed = Member(run, Lineage(**vars(member.lin)))
    resumed.restore()
    assert [(r["phase"], r["step"]) for r in read_jsonl(resumed.dir / "history.jsonl")] == \
        [("seed", 0), ("seed", 1), ("seed", 2), ("amp", 0), ("amp", 1), ("amp", 2), ("amp", 3)]
    assert [r["step"] for r in read_jsonl(resumed.dir / "probes.jsonl")] == [0, 4]
    assert sorted(p.name for p in (resumed.dir / "rollouts").iterdir()) == [f"step_{t:06d}.jsonl.gz" for t in range(4)]
    assert sorted(p.name for p in (resumed.dir / "probe_responses").iterdir()) == ["step_000000.jsonl.gz",
                                                                                   "step_000004.jsonl.gz"]
    assert [p.name for p in (resumed.dir / "checkpoints").iterdir()] == ["step_000004"]
    policy.update(resumed.weights, seqs_b, lr=1e-3)
    for name in uninterrupted.lora:
        torch.testing.assert_close(resumed.weights.lora[name], uninterrupted.lora[name], rtol=0, atol=0)


def test_new_lineage_restarts_clean(tmp_path, base_model):
    run = _fake_run(tmp_path, make_policy(base_model))
    d = lineage_dir(run, "m0")
    append_jsonl(d / "history.jsonl", {"phase": "seed", "step": 0})
    (d / "checkpoints" / "step_000000.tmp").mkdir(parents=True)
    member = Member(run, Lineage("m0", "m0", 0, "plain", None, 1))
    member.restore()
    assert not (d / "history.jsonl").exists() and not (d / "checkpoints").exists()


def test_culling_rules_and_replacement(tmp_path, base_model):
    cfg = config.parse({"name": "t", "model": "tiny", "seed": 0, "culling": {"grace": 4, "max_replacements": 1},
                        "members": [{"id": "m0", "arm": "plain", "seed": 1}]}, scaffold_names=set())
    run = SimpleNamespace(dir=tmp_path, cfg=cfg, policy=None,
                          base_probe={"accuracy": {"train": 0.3, "selection": 0.2, "foundation": 0.5}})
    member = Member(run, Lineage("m0", "m0", 0, "plain", None, 1, status="active"))

    def probe(step, train=0.4, foundation=0.5, parsed=0.9, truncated=0.1):
        return {"step": step, "accuracy": {"train": train, "selection": 0.2, "foundation": foundation},
                "parsed": parsed, "truncated": truncated}

    assert member.cull_reason(probe(2, train=0.0, foundation=0.0, parsed=0.0, truncated=1.0)) is None  # grace
    assert "parseable" in member.cull_reason(probe(4, parsed=0.4))
    assert "truncated" in member.cull_reason(probe(4, truncated=0.6))
    assert "foundation" in member.cull_reason(probe(4, foundation=0.2))
    member.lin.probes = [probe(2, train=0.1)]  # below base, but inside the grace period
    assert member.cull_reason(probe(4, train=0.1)) is None
    member.lin.probes.append(probe(4, train=0.1))
    assert "two consecutive" in member.cull_reason(probe(8, train=0.1))
    assert member.cull_reason(probe(8, train=0.3)) is None  # equal to base is not below

    member.cull("test")
    assert member.lin.status == "culled" and member.lin.replaced_by == "m0~1"
    write_json(member.dir / "state.json", vars(member.lin))
    from evotaste.member import current_lineage, load_lineage

    nxt = current_lineage(run, "m0")
    assert (nxt.id, nxt.generation, nxt.status) == ("m0~1", 1, "new") and nxt.seed != 1
    second = Member(run, nxt)
    second.cull("test")  # beyond max_replacements: the slot is exhausted
    write_json(second.dir / "state.json", vars(second.lin))
    assert second.lin.replaced_by is None and current_lineage(run, "m0") is None
    assert load_lineage(run, "m0").replaced_by == "m0~1"


def test_adapter_files_roundtrip(tmp_path, base_model):
    policy = make_policy(base_model)
    w = policy.new_member(seed=3, b_std=0.1)
    policy.save_adapter(w, tmp_path / "a")
    loaded = policy.load_adapter(tmp_path / "a")
    assert all(torch.equal(loaded[k], w.lora[k]) for k in w.lora)
    assert (tmp_path / "a" / "adapter_config.json").exists()
    seqs = [Sequence([1, 2, 3], [4, 5], 1.0)]
    assert torch.equal(policy.logprobs(w, seqs)[0], policy.logprobs(type(w)(loaded), seqs)[0])


def test_light_checkpoints_are_pruned_and_resumable(tmp_path, base_model):
    """Light checkpoints every 3 steps, probe checkpoints at 0 and 5; a kill after step 7 resumes from step 6."""
    policy = make_policy(base_model)
    run = _fake_run(tmp_path, policy)
    member = Member(run, Lineage("m0", "m0", 0, "plain", None, 1, status="active", step=0, probes=[{"step": 0}]))
    member.weights = policy.new_member(seed=1, b_std=0.05)
    member.write_checkpoint()
    write_json(member.dir / "state.json", vars(member.lin))
    snapshot = None
    for t in range(8):
        policy.update(member.weights, random_groups(seed=10 + t), lr=1e-3)
        append_jsonl(member.dir / "history.jsonl", {"phase": "amp", "step": t})
        write_jsonl_gz(member.dir / "rollouts" / f"step_{t:06d}.jsonl.gz", [{"step": t}])
        member.lin.step = t + 1
        if member.lin.step == 5:  # a probe step, as finish_probe does it
            member.lin.probes.append({"step": 5})
            member.write_checkpoint()
            write_json(member.dir / "state.json", vars(member.lin))
            member.prune_checkpoints()
        elif member.lin.step % 3 == 0:
            member.light_checkpoint()
            snapshot = copy.deepcopy(member.weights)
        names = sorted(p.name for p in (member.dir / "checkpoints").iterdir())
        if member.lin.step == 5:
            assert names == ["step_000000", "step_000005"]  # the light checkpoint at 3 is gone
    assert sorted(p.name for p in (member.dir / "checkpoints").iterdir()) == ["step_000000", "step_000005",
                                                                              "step_000006"]
    assert [p.parent.name for p in (member.dir / "checkpoints").glob("*/optimizer.pt")] == ["step_000006"]
    member.write_checkpoint()  # a kill between writing the step-8 checkpoint and committing the state
    (member.dir / "checkpoints" / "step_000008").rename(member.dir / "checkpoints" / "step_000009")

    resumed = Member(run, Lineage(**read_json_state(member.dir)))
    resumed.restore()
    assert resumed.lin.step == 6 and [p["step"] for p in resumed.lin.probes] == [0, 5]
    assert [r["step"] for r in read_jsonl(resumed.dir / "history.jsonl")] == [0, 1, 2, 3, 4, 5]
    assert sorted(p.name for p in (resumed.dir / "rollouts").iterdir()) == [f"step_{t:06d}.jsonl.gz" for t in range(6)]
    assert sorted(p.name for p in (resumed.dir / "checkpoints").iterdir()) == ["step_000000", "step_000005",
                                                                               "step_000006"]
    seqs = random_groups(seed=99)
    policy.update(resumed.weights, seqs, lr=1e-3)
    policy.update(snapshot, seqs, lr=1e-3)
    for name in snapshot.lora:
        torch.testing.assert_close(resumed.weights.lora[name], snapshot.lora[name], rtol=0, atol=0)


def read_json_state(d):
    import json

    return json.loads((d / "state.json").read_text())


def test_oom_retry_bound(monkeypatch):
    from evotaste import member as member_module

    monkeypatch.setattr(member_module, "OOM_WAIT_S", 0)
    monkeypatch.setattr(member_module, "gpu_processes", lambda: "")
    released = []
    policy = SimpleNamespace(release_memory=lambda: released.append(1))
    calls = []

    def flaky(failures):
        def fn():
            calls.append(1)
            if len(calls) <= failures:
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return "done"
        return fn

    assert member_module.retry_on_oom(policy, "m0 step 3 update", flaky(3)) == "done"
    assert len(calls) == 4 and len(released) == 3
    calls.clear()
    with pytest.raises(torch.OutOfMemoryError):
        member_module.retry_on_oom(policy, "m0 step 3 update", flaky(100))
    assert len(calls) == member_module.OOM_ATTEMPTS


@pytest.mark.parametrize("where", ["after_backward", "inside_step"])
def test_update_retried_after_oom_is_the_uninterrupted_update(base_model, monkeypatch, where):
    """An OOM after the gradients were accumulated, or halfway through the optimizer step, leaves no trace."""
    from evotaste import member as member_module

    monkeypatch.setattr(member_module, "OOM_WAIT_S", 0)
    monkeypatch.setattr(member_module, "gpu_processes", lambda: "")
    policy = make_policy(base_model)
    start = policy.new_member(seed=7, b_std=0.05)
    policy.update(start, random_groups(seed=1), lr=1e-3)  # with optimizer state
    seqs = random_groups(seed=2)
    reference = copy.deepcopy(start)
    policy.update(reference, seqs, lr=1e-3)

    failed = []
    if where == "after_backward":
        real = policy._accumulate

        def accumulate_then_fail(active):
            out = real(active)
            if not failed:
                failed.append(1)
                raise torch.OutOfMemoryError("CUDA out of memory")
            return out
        monkeypatch.setattr(policy, "_accumulate", accumulate_then_fail)
    else:
        real_step = torch.optim.AdamW.step

        def step_then_fail(self, *a, **k):
            out = real_step(self, *a, **k)  # parameters already changed
            if not failed:
                failed.append(1)
                raise torch.OutOfMemoryError("CUDA out of memory")
            return out
        monkeypatch.setattr(torch.optim.AdamW, "step", step_then_fail)
    member_module.retry_on_oom(policy, "m0 update", lambda: policy.update(start, seqs, lr=1e-3))
    assert failed
    for name in reference.lora:
        torch.testing.assert_close(start.lora[name], reference.lora[name], rtol=0, atol=0)
    for (_, a), (_, b) in zip(sorted(start.optimizer["state"].items()), sorted(reference.optimizer["state"].items())):
        torch.testing.assert_close(a["exp_avg"], b["exp_avg"], rtol=0, atol=0)


def test_resume_refuses_a_lost_optimizer_state(tmp_path, base_model):
    policy = make_policy(base_model)
    run = _fake_run(tmp_path, policy)
    member = Member(run, Lineage("m0", "m0", 0, "plain", None, 1, status="active", step=5))
    member.weights = policy.new_member(seed=1, b_std=0.05)
    policy.update(member.weights, random_groups(seed=1), lr=1e-3)
    member.write_checkpoint()
    write_json(member.dir / "state.json", vars(member.lin))
    append_jsonl(member.dir / "history.jsonl", {"phase": "amp", "step": 2, "sequences": 9})
    (member.checkpoint_dir(5) / "optimizer.pt").unlink()
    with pytest.raises(RuntimeError, match="optimizer.pt is missing"):
        Member(run, Lineage(**vars(member.lin))).restore()
    filter_all = member.dir / "history.jsonl"  # no update with sequences yet: no optimizer state is right
    filter_all.write_text('{"phase": "amp", "step": 2, "sequences": 0}\n')
    resumed = Member(run, Lineage(**vars(member.lin)))
    resumed.restore()
    assert resumed.weights.optimizer is None


def test_worker_logs_uncommitted_changes(tmp_path, monkeypatch, caplog):
    import subprocess

    from evotaste import worker

    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "a.py"], check=True)
    subprocess.run(["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-qm", "a"], check=True)
    with caplog.at_level("INFO"):
        worker.log_uncommitted_changes("r")
    assert "working tree clean at launch" in caplog.text
    (tmp_path / "a.py").write_text("x = 2\n")
    caplog.clear()
    with caplog.at_level("INFO"):
        worker.log_uncommitted_changes("r")
    assert "working tree dirty at launch" in caplog.text and "a.py" in caplog.text
    patch = next((tmp_path / "logs" / "r").glob("uncommitted-*.patch"))
    assert b"+x = 2" in patch.read_bytes() and patch.name[12:24] in caplog.text
