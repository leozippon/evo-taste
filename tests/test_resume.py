"""Resume from a checkpoint: the next update is identical, and records of steps after the checkpoint are dropped."""

import copy
from types import SimpleNamespace

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
