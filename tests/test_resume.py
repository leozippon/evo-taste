"""Resume from a checkpoint and clone starts: the next update is identical, and records after the checkpoint go."""

import copy
from types import SimpleNamespace

import pytest
import torch

from evotaste import config
from evotaste.member import Lineage, Member, current_lineage, first_lineage, lineage_dir, load_lineage
from evotaste.records import append_jsonl, read_json, read_jsonl, write_json, write_jsonl_gz
from evotaste.trainer import Sequence
from tests.conftest import make_policy, random_groups

PARENT = {"id": "p", "seed": 1, "context": "refute"}
CLONES = [{"id": "c1", "seed": 2, "parent": {"member": "p", "step": 3}},
          {"id": "c2", "seed": 3, "parent": {"member": "p", "step": 3}}]


def _fake_run(tmp_path, policy, members=(PARENT,), **sections):
    cfg = config.parse({"name": "t", "model": "tiny", "seed": 0, "amplification": {"steps": 8},
                        "strata": {"path": "strata.tsv", "sha256": "0" * 64}, "members": list(members), **sections},
                       {"refute"})
    return SimpleNamespace(dir=tmp_path, cfg=cfg, policy=policy)


def _commit(member):
    write_json(member.dir / "state.json", vars(member.lin))


def test_resume_reproduces_training_state(tmp_path, base_model):
    policy = make_policy(base_model)
    run = _fake_run(tmp_path, policy)
    seqs_a, seqs_b = random_groups(seed=1), random_groups(seed=2)
    member = Member(run, Lineage("p", "p", 0, "refute", 1, status="active", step=4))
    member.weights = policy.new_member(seed=1, b_std=0.05)
    policy.update(member.weights, seqs_a, lr=1e-3)
    member.write_checkpoint()
    _commit(member)
    uninterrupted = copy.deepcopy(member.weights)
    policy.update(uninterrupted, seqs_b, lr=1e-3)

    # Records written after the checkpoint, as by a worker killed during step 5.
    for i in range(3):
        append_jsonl(member.dir / "history.jsonl", {"phase": "seed", "step": i})
    for t in range(6):
        append_jsonl(member.dir / "history.jsonl", {"phase": "amp", "step": t, "sequences": 4})
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
    d = lineage_dir(run, "p")
    append_jsonl(d / "history.jsonl", {"phase": "seed", "step": 0})
    (d / "checkpoints" / "step_000000.tmp").mkdir(parents=True)
    member = Member(run, first_lineage(run.cfg, "p"))
    member.restore()
    assert not (d / "history.jsonl").exists() and not (d / "checkpoints").exists()


def _probe(step, accuracy=0.4, complied=0.9, truncated=0.1):
    return {"step": step, "accuracy": accuracy, "complied": complied, "truncated": truncated}


def test_culling_rules_and_replacement(tmp_path, base_model):
    run = _fake_run(tmp_path, None, culling={"grace": 4, "max_replacements": 1})
    run.base_probe = {"accuracy": 0.3}
    member = Member(run, Lineage("p", "p", 0, "refute", 1, status="active"))
    assert member.cull_reason(_probe(2, accuracy=0.0, complied=0.0, truncated=1.0)) is None  # grace
    assert "wrapper asked for" in member.cull_reason(_probe(4, complied=0.4))
    assert "truncated" in member.cull_reason(_probe(4, truncated=0.6))
    assert "half the base" in member.cull_reason(_probe(4, accuracy=0.1))
    member.lin.probes = [_probe(2, accuracy=0.2)]  # below base, but inside the grace period
    assert member.cull_reason(_probe(4, accuracy=0.2)) is None
    member.lin.probes.append(_probe(4, accuracy=0.2))
    assert "two consecutive" in member.cull_reason(_probe(8, accuracy=0.2))
    assert member.cull_reason(_probe(8, accuracy=0.3)) is None  # equal to base is not below

    member.cull("test")
    assert member.lin.status == "culled" and member.lin.replaced_by == "p~1"
    _commit(member)
    nxt = current_lineage(run, "p")
    assert (nxt.id, nxt.generation, nxt.status, nxt.context) == ("p~1", 1, "new", "refute") and nxt.seed != 1
    second = Member(run, nxt)
    second.cull("test")  # beyond max_replacements: the slot is exhausted
    _commit(second)
    assert second.lin.replaced_by is None and current_lineage(run, "p") is None
    assert load_lineage(run, "p").replaced_by == "p~1"


def test_disabled_culling_records_and_replaces_nothing(tmp_path, base_model, monkeypatch):
    from evotaste import member as member_module

    run = _fake_run(tmp_path, make_policy(base_model), culling={"enabled": False, "grace": 2})
    run.base_probe = {"accuracy": 0.9}
    run.probe_tasks = []
    monkeypatch.setattr(member_module, "response_records", lambda pairs, token_ids=False: [])
    monkeypatch.setattr(member_module, "summarize_probe", lambda records: {
        "accuracy": 0.1, "by_domain": {}, "complied": 0.9, "truncated": 0.0})
    member = Member(run, Lineage("p", "p", 0, "refute", 1, status="active", step=4))
    member.weights = run.policy.new_member(seed=1)
    member.finish_probe([])
    summary = read_jsonl(member.dir / "probes.jsonl")[0]
    assert summary["cull"] == "probe accuracy below half the base model's" and summary["culled"] is False
    assert read_json(member.dir / "state.json")["status"] == "active" and member.lin.replaced_by is None
    assert not lineage_dir(run, "p~1").exists()


def test_adapter_files_roundtrip(tmp_path, base_model):
    policy = make_policy(base_model)
    w = policy.new_member(seed=3, b_std=0.1)
    policy.save_adapter(w, tmp_path / "a")
    loaded = policy.load_adapter(tmp_path / "a")
    assert all(torch.equal(loaded[k], w.lora[k]) for k in w.lora)
    assert (tmp_path / "a" / "adapter_config.json").exists()
    seqs = [Sequence([1, 2, 3], [4, 5], 1.0)]
    assert torch.equal(policy.logprobs(w, seqs)[0], policy.logprobs(type(w)(loaded), seqs)[0])


def _train_parent(run, policy, steps, light_every=None):
    """The parent lineage after `steps` updates, as the worker leaves it: a probe checkpoint at 0, light checkpoints
    every `light_every` steps and at its clone points, history lines, state committed after every checkpoint."""
    parent = Member(run, Lineage("p", "p", 0, "refute", 1, status="active", step=0, probes=[{"step": 0}]))
    parent.weights = policy.new_member(seed=1, b_std=0.05)
    parent.write_checkpoint()
    _commit(parent)
    for t in range(steps):
        stats = policy.update(parent.weights, random_groups(seed=10 + t), lr=1e-3)
        append_jsonl(parent.dir / "history.jsonl", {"phase": "amp", "step": t, "sequences": stats["sequences"]})
        parent.lin.step = t + 1
        if parent.checkpoint_due(light_every or 10 ** 6):
            parent.light_checkpoint()
    return parent


def test_light_checkpoints_are_pruned_and_resumable(tmp_path, base_model):
    """Light checkpoints every 3 steps, probe checkpoints at 0 and 5; a kill after step 7 resumes from step 6."""
    policy = make_policy(base_model)
    run = _fake_run(tmp_path, policy, members=[{**PARENT}])
    member = Member(run, Lineage("p", "p", 0, "refute", 1, status="active", step=0, probes=[{"step": 0}]))
    member.weights = policy.new_member(seed=1, b_std=0.05)
    member.write_checkpoint()
    _commit(member)
    snapshot = None
    for t in range(8):
        policy.update(member.weights, random_groups(seed=10 + t), lr=1e-3)
        append_jsonl(member.dir / "history.jsonl", {"phase": "amp", "step": t, "sequences": 9})
        write_jsonl_gz(member.dir / "rollouts" / f"step_{t:06d}.jsonl.gz", [{"step": t}])
        member.lin.step = t + 1
        if member.lin.step == 5:  # a probe step, as finish_probe does it
            member.lin.probes.append({"step": 5})
            member.write_checkpoint()
            _commit(member)
            member.prune_checkpoints()
        elif member.checkpoint_due(3):
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

    resumed = Member(run, Lineage(**read_json(member.dir / "state.json")))
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


def _assert_same_state(a, b):
    for name in a.lora:
        assert torch.equal(a.lora[name], b.lora[name]), name
    assert a.optimizer["state"].keys() == b.optimizer["state"].keys()
    for k in a.optimizer["state"]:
        for field in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(a.optimizer["state"][k][field], b.optimizer["state"][k][field]), (k, field)


def test_clone_continues_the_parent_bit_exactly(tmp_path, base_model):
    """A clone starts from the parent's checkpoint at the clone point with weights, optimizer state and step counter,
    and the same update then gives the same result as in the parent; the clone point keeps its optimizer state."""
    policy = make_policy(base_model)
    run = _fake_run(tmp_path, policy, members=[PARENT, *CLONES])
    clone = Member(run, first_lineage(run.cfg, "c1"))
    assert clone.is_clone and clone.lin.parent == {"lineage": "p", "step": 3} and clone.lin.context == "refute"
    _train_parent(run, policy, 2, light_every=1)
    assert clone.parent_checkpoint() is None  # the parent has not reached the clone point
    parent = _train_parent(run, policy, 5, light_every=4)  # light checkpoints at 3 (clone point) and 4
    assert parent.clone_points == {3}
    ckpts = sorted(p.name for p in (parent.dir / "checkpoints").iterdir())
    assert ckpts == ["step_000000", "step_000003", "step_000004"]
    assert sorted(p.parent.name for p in (parent.dir / "checkpoints").glob("*/optimizer.pt")) == [
        "step_000003", "step_000004"]

    reference = _train_parent(run, policy, 3)  # an uninterrupted parent at step 3 (deterministic on the CPU)
    parent = _train_parent(run, policy, 5, light_every=4)
    ckpt = clone.parent_checkpoint()
    assert ckpt == parent.checkpoint_dir(3)
    clone.start_clone(ckpt)
    assert clone.lin.step == 3
    _assert_same_state(clone.weights, reference.weights)
    seqs = random_groups(seed=13)  # the parent's step-3 batch
    policy.update(clone.weights, seqs, lr=1e-3)
    policy.update(reference.weights, seqs, lr=1e-3)
    _assert_same_state(clone.weights, reference.weights)

    # The clone commits its own copy of the clone point; it resumes from it like any lineage.
    clone.lin.status = "active"
    clone.weights = copy.deepcopy(reference.weights)
    clone.lin.step = 4
    clone.write_checkpoint()
    _commit(clone)
    resumed = Member(run, Lineage(**read_json(clone.dir / "state.json")))
    resumed.restore()
    _assert_same_state(resumed.weights, reference.weights)


def test_clone_refuses_a_parent_culled_before_the_clone_point(tmp_path, base_model):
    policy = make_policy(base_model)
    run = _fake_run(tmp_path, policy, members=[PARENT, *CLONES])
    parent = _train_parent(run, policy, 2, light_every=1)
    parent.lin.status = "culled"
    _commit(parent)
    with pytest.raises(RuntimeError, match="culled at step 2, before the clone point 3"):
        Member(run, first_lineage(run.cfg, "c1")).parent_checkpoint()


def test_clone_refuses_a_lost_optimizer_state(tmp_path, base_model):
    policy = make_policy(base_model)
    run = _fake_run(tmp_path, policy, members=[PARENT, *CLONES])
    parent = _train_parent(run, policy, 3, light_every=10)
    (parent.checkpoint_dir(3) / "optimizer.pt").unlink()
    clone = Member(run, first_lineage(run.cfg, "c1"))
    with pytest.raises(RuntimeError, match="optimizer.pt is missing"):
        clone.start_clone(clone.parent_checkpoint())


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
        real = policy._step

        def fail_then_step(member, opt):
            if not failed:
                failed.append(1)
                raise torch.OutOfMemoryError("CUDA out of memory")
            return real(member, opt)
        monkeypatch.setattr(policy, "_step", fail_then_step)
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
    _assert_same_state(start, reference)


def test_resume_refuses_a_lost_optimizer_state(tmp_path, base_model):
    policy = make_policy(base_model)
    run = _fake_run(tmp_path, policy)
    member = Member(run, Lineage("p", "p", 0, "refute", 1, status="active", step=5))
    member.weights = policy.new_member(seed=1, b_std=0.05)
    policy.update(member.weights, random_groups(seed=1), lr=1e-3)
    member.write_checkpoint()
    _commit(member)
    append_jsonl(member.dir / "history.jsonl", {"phase": "amp", "step": 2, "sequences": 9})
    (member.checkpoint_dir(5) / "optimizer.pt").unlink()
    with pytest.raises(RuntimeError, match="optimizer.pt is missing"):
        Member(run, Lineage(**vars(member.lin))).restore()
    (member.dir / "history.jsonl").write_text('{"phase": "amp", "step": 2, "sequences": 0}\n')  # no update yet
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
