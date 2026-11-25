"""Run-config schema: defaults, the effective settings of members and clones, and every refusal."""

from pathlib import Path

import pytest

from evotaste import config

ROOT = Path(__file__).resolve().parents[1]
CONTEXTS = config.load_contexts(ROOT / "configs" / "contexts.yaml")


STRATA = {"path": "strata.tsv", "sha256": "0" * 64}


def parse(members, **sections):
    return config.parse({"name": "t", "model": "m", "seed": 0, "strata": STRATA, "members": members, **sections},
                        CONTEXTS)


def test_contexts_file():
    assert len(CONTEXTS) == 12 and {"placebo", "refute", "verify", "procedure"} <= set(CONTEXTS)
    text = CONTEXTS["refute"]
    assert text.startswith("How you work on every problem: Treat every answer") and text.endswith("Just work this way.")


def test_defaults_are_the_pilot_defaults():
    cfg = parse([{"id": "a", "seed": 1, "context": "refute"}])
    seeding, amp, culling = cfg.settings(cfg.members[0])
    assert (seeding.method, seeding.steps, seeding.batch, seeding.lr) == ("opcd", 40, 64, 1e-4)
    assert seeding.max_tokens == 4096
    assert (amp.prompts, amp.rollouts, amp.lr, amp.max_tokens) == (64, 4, 1e-4, 4096)
    assert (amp.length.free, amp.length.slope, amp.advantage, amp.self_imitation) == (2048, 0.25, "centered", 0.0)
    assert cfg.probes.every == 15 and cfg.probes.max_tokens == 4096 and culling.enabled
    assert (cfg.lora.rank, cfg.lora.alpha) == (32, 64)


def test_overrides_merge_nested_sections():
    cfg = parse([{"id": "a", "seed": 1, "context": "map",
                  "overrides": {"seeding": {"domains": ["math"]},
                                "amplification": {"self_imitation": 0.5, "length": {"slope": 0.1}}}}])
    seeding, amp, _ = cfg.settings(cfg.members[0])
    assert seeding.domains == ("math",) and amp.self_imitation == 0.5
    assert (amp.length.free, amp.length.slope) == (2048, 0.1)


def test_a_clone_continues_its_parent():
    cfg = parse([{"id": "p", "seed": 1, "context": "refute", "overrides": {"amplification": {"self_imitation": 0.5}}},
                 {"id": "c1", "seed": 2, "parent": {"member": "p", "step": 60},
                  "overrides": {"amplification": {"steps": 120}}},
                 {"id": "c2", "seed": 3, "parent": {"member": "p", "step": 60},
                  "overrides": {"amplification": {"steps": 120}}}])
    p, c1, c2 = cfg.members
    assert c1.parent == config.Parent("p", 60) and cfg.context_of(c1) == "refute"
    assert cfg.settings(c1)[1].self_imitation == 0.5 and cfg.settings(c1)[1].steps == 120
    assert cfg.settings(c1) == cfg.settings(c2) and cfg.settings(c1)[0] == cfg.settings(p)[0]
    assert cfg.clone_points("p") == {60} and cfg.clone_points("c1") == set()


def test_clone_chains_inherit_along_the_chain():
    """Pilot C: a context seeded once, cloned at step 0 into both update rules, and a rule's member cloned again."""
    cfg = parse([{"id": "s", "seed": 1, "context": "refute", "overrides": {"amplification": {"steps": 0}}},
                 {"id": "adv", "seed": 2, "parent": {"member": "s", "step": 0},
                  "overrides": {"amplification": {"steps": 60}}},
                 {"id": "si", "seed": 3, "parent": {"member": "s", "step": 0},
                  "overrides": {"amplification": {"steps": 60, "self_imitation": 0.5}}},
                 {"id": "si-c1", "seed": 4, "parent": {"member": "si", "step": 45},
                  "overrides": {"amplification": {"steps": 105}}}], clonable_steps=[45])
    s, adv, si, c1 = cfg.members
    assert [cfg.context_of(m) for m in cfg.members] == ["refute"] * 4
    assert [cfg.start_step(m) for m in cfg.members] == [0, 0, 0, 45]
    assert cfg.settings(c1)[1].self_imitation == 0.5 and cfg.settings(c1)[1].steps == 105  # the parent's rule
    assert cfg.settings(adv)[1].self_imitation == 0.0 and cfg.settings(adv)[0] == cfg.settings(s)[0]
    assert cfg.clone_points("s") == {0, 45} and cfg.clone_points("si") == {45} and cfg.clone_points("si-c1") == {45}


@pytest.mark.parametrize("members, sections, match", [
    ([{"id": "a", "seed": 1}], {}, "context is required"),  # opcd without a context
    ([{"id": "a", "seed": 1, "context": "refute", "overrides": {"seeding": {"method": "none"}}}], {}, "only for"),
    ([{"id": "a", "seed": 1, "context": "popper"}], {}, "unknown context"),
    ([{"id": "a", "seed": 1, "context": "refute", "overrides": {"seeding": {"method": "sft"}}}], {}, "seeding.method"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"advantage": "whitened"}}, "advantage"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"self_imitation": -0.1}}, "self_imitation"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"length": {"free": 4096}}}, "length"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"length": {"slope": -1}}}, "length"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"rollouts": 1}}, "rollouts"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"seeding": {"domains": []}}, "domains"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"truncation_penalty": 0.5}}, "unknown keys"),
    ([{"id": "a", "seed": 1, "context": "refute", "arm": "plain"}], {}, "unknown keys"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"culling": {"enabled": "no"}}, "expected bool"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"seeding": {"lr": "fast"}}, "expected a number"),
    ([{"id": "a", "seed": 1}, {"id": "a", "seed": 2}], {"seeding": {"method": "none"}}, "unique"),
    ([{"id": "a@1", "seed": 1}], {"seeding": {"method": "none"}}, "letters"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"clonable_steps": [-1]}, "clonable_steps"),
    # D1 of the 2026-11-25 audit: any shared seed, between seeded members, clones of different parents or points
    ([{"id": "a", "seed": 7, "context": "refute"}, {"id": "b", "seed": 7, "context": "verify"},
      {"id": "t1", "seed": 9, "parent": {"member": "a", "step": 0}},
      {"id": "t2", "seed": 9, "parent": {"member": "b", "step": 0}},
      {"id": "k1", "seed": 9, "parent": {"member": "a", "step": 10}, "overrides": {"amplification": {"steps": 70}}}],
     {}, r"seeds must be distinct; repeated: \[7, 9\]"),
])
def test_refusals(members, sections, match):
    with pytest.raises(ValueError, match=match):
        parse(members, **sections)


PARENT = {"id": "p", "seed": 1, "context": "refute"}


@pytest.mark.parametrize("clone, match", [
    ({"parent": {"member": "q", "step": 10}}, "not a member"),
    ({"parent": {"member": "c", "step": 10}}, "cycle"),  # its own parent
    ({"parent": {"member": "p", "step": 10}, "context": "refute"}, "do not set"),
    ({"parent": {"member": "p", "step": 10}, "overrides": {"seeding": {"lr": 1e-3}}}, "do not set"),
    ({"parent": {"member": "p", "step": 61}}, "within the parent's course"),
    ({"parent": {"member": "p", "step": 60}}, "below the clone's amplification.steps"),  # nothing left to train
    ({"parent": {"member": "p", "step": -1}}, "within the parent's course"),
    ({"parent": {"member": "p", "step": 10}, "seed": 1}, "seeds must be distinct"),  # the parent's seed
    ({"parent": {"member": "p", "step": "ten"}}, "expected int"),
])
def test_clone_refusals(clone, match):
    with pytest.raises(ValueError, match=match):
        parse([PARENT, {"id": "c", "seed": 2, **clone}])


def test_training_needs_the_pinned_strata_table():
    data = {"name": "t", "model": "m", "seed": 0, "members": [{"id": "a", "seed": 1, "context": "refute"}]}
    with pytest.raises(ValueError, match="strata"):
        config.parse(data, CONTEXTS)
    seeding_only = config.parse({**data, "amplification": {"steps": 0}}, CONTEXTS)  # seeding pilots need none
    assert seeding_only.strata is None
    with pytest.raises(ValueError, match="sha256"):
        config.parse({**data, "strata": {"path": "x"}}, CONTEXTS)


def test_clone_chain_refusals():
    clone = {"id": "c", "seed": 2, "parent": {"member": "p", "step": 10}}
    with pytest.raises(ValueError, match="cycle: c <- g <- c"):
        parse([PARENT, {**clone, "parent": {"member": "g", "step": 20}}, {"id": "g", "seed": 3,
                                                                          "parent": {"member": "c", "step": 20}}])
    with pytest.raises(ValueError, match=r"within the parent's course \(10\.\.60\)"):  # before its parent started
        parse([PARENT, clone, {"id": "g", "seed": 3, "parent": {"member": "c", "step": 5}}])
    parse([PARENT, clone, {"id": "g", "seed": 3, "parent": {"member": "c", "step": 10}}])  # at its parent's start


def test_template_and_pilot_b_parse():
    cfg = config.load(ROOT / "configs" / "run" / "pilot-template.yaml")
    assert not cfg.culling.enabled and cfg.clonable_steps == (45,)
    twins = [m for m in cfg.members if m.parent]
    assert twins and all(m.parent.step == 0 and cfg.settings(m)[1].steps == 60 for m in twins)
    assert {cfg.settings(m)[1].self_imitation for m in twins} == {0.0, 0.5}
    b = config.load(ROOT / "configs" / "run" / "pilot-b.yaml")
    settings = [b.settings(m) for m in b.members]
    assert len(b.members) == 26 and not b.culling.enabled and b.strata is None
    assert all(a.steps == 0 for _, a, _ in settings)
    assert sorted(s.method for s, _, _ in settings).count("opcd") == 23
    assert {m.context for m in b.members if m.context} == set(CONTEXTS)
    math_only = [m.context for m, (s, _, _) in zip(b.members, settings) if s.domains == ("math",)]
    assert sorted(math_only) == sorted(set(CONTEXTS) - {"placebo"})
    assert sorted(m.context for m, (s, _, _) in zip(b.members, settings) if s.method == "imitation") == [
        "refute", "verify"]
