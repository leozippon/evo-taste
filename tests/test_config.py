"""Run-config schema: defaults, the effective settings of members and clones, and every refusal."""

import dataclasses
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
    assert (amp.exaggeration, amp.practice, amp.imitate_within_free) == (None, None, False)  # pilot C's update
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
    ([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"exaggeration": {"gamma": 1.0}}},
     "missing required keys"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"exaggeration": {"gamma": -1, "weight": 1}}},
     "exaggeration"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"exaggeration": {"gamma": 1, "weight": -1}}},
     "exaggeration"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"exaggeration": {"gamma": 1, "weight": 1,
                                                                                       "beta": 1}}}, "unknown keys"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"imitate_within_free": "yes"}}, "expected bool"),
    *[([{"id": "a", "seed": 1, "context": "refute"}],
       {"amplification": {"practice": {"eta": 4.0, "decay": 0.8, "floor": 0.25, "ceiling": 4.0, **bad}}}, "practice")
      for bad in ({"eta": -1}, {"decay": 1.0}, {"decay": -0.1}, {"floor": 0}, {"floor": 1.5, "ceiling": 2},
                  {"ceiling": 0.5})],
    ([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"practice": {"eta": 4.0, "decay": 0.8}}},
     "missing required keys"),
    ([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"practice": {
        "eta": 4.0, "decay": 0.8, "floor": 0.25, "ceiling": 4.0, "reference": "member"}}}, "practice.reference"),
    *[([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"taste": bad}}, match)
      for bad, match in (({"clip": 2.0}, "missing required keys"), ({"strength": 0.25, "clip": 0}, "taste needs"),
                         ({"strength": 0.5}, "taste needs"), ({"strength": -0.25, "clip": 4}, "taste needs"),
                         ({"strength": float("nan")}, "taste needs"), ({"strength": "high"}, "expected a number"),
                         ({"strength": 0.25, "sign": 1}, "unknown keys"), (0.25, "expected a mapping"),
                         ({"strength": 0.25, "residualize": "yes"}, "expected bool"))],
    *[([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {"revisit": bad}}, match)
      for bad, match in (({"share": 0.33}, "missing required keys"), ({"source": "own"}, "missing required keys"),
                         ({"share": -0.1, "source": "own"}, "revisit needs"),
                         ({"share": 1.5, "source": "own"}, "revisit needs"),
                         ({"share": float("nan"), "source": "own"}, "revisit needs"),
                         ({"share": 0.33, "source": "theirs"}, "revisit needs"),
                         ({"share": 0.33, "source": "own", "min_seen": 0}, "revisit needs"),
                         ({"share": 0.33, "source": "own", "min_seen": 4.5}, "expected int"),
                         ({"share": 0.33, "source": "own", "imitate": -0.5}, "revisit needs"),
                         ({"share": 0.33, "source": "own", "replay": 0.5}, "unknown keys"),
                         ({"share": 0.33, "source": None}, "expected str"), (0.33, "expected a mapping"))],
    ([{"id": "a", "seed": 1, "context": "refute"}], {"amplification": {
        "revisit": {"share": 0.33, "source": "own"}, "practice": {"eta": 0.0, "decay": 0.8, "floor": 0.25,
                                                                  "ceiling": 4.0}}}, "cannot be combined"),
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


PRACTICE = {"eta": 4.0, "decay": 0.8, "floor": 0.25, "ceiling": 4.0}


def test_amplification_mechanisms_per_member():
    cfg = parse([{"id": "a", "seed": 1, "context": "refute", "overrides": {"amplification": {
                  "exaggeration": {"gamma": 1, "weight": 0.5}, "practice": PRACTICE}}},
                 {"id": "b", "seed": 2, "context": "refute", "overrides": {"amplification": {"practice": {"eta": 0.0}}}}],
                amplification={"imitate_within_free": True, "practice": PRACTICE})
    a, b = (cfg.settings(m)[1] for m in cfg.members)
    assert a.exaggeration == config.Exaggeration(1.0, 0.5) and a.practice == config.Practice(4.0, 0.8, 0.25, 4.0)
    assert b.exaggeration is None and b.practice == config.Practice(0.0, 0.8, 0.25, 4.0)  # merged key by key
    assert a.imitate_within_free and b.imitate_within_free


def test_taste_and_base_referenced_practice():
    """Pilot E's layout: one parent cloned into the three signs of the taste-shaped reward; nothing else differs."""
    variants = {"own": 0.25, "neu": 0.0, "rev": -0.25}
    cfg = parse([PARENT] + [{"id": f"p-{v}", "seed": 10 + i, "parent": {"member": "p", "step": 45},
                             "overrides": {"amplification": {"steps": 90, "taste": {"strength": s}}}}
                            for i, (v, s) in enumerate(variants.items())],
                amplification={"practice": {**PRACTICE, "reference": "base"}})
    parent, *clones = (cfg.settings(m)[1] for m in cfg.members)
    assert parent.taste is None and [c.taste for c in clones] == [config.Taste(s, 2.0) for s in variants.values()]
    assert {dataclasses.replace(c, taste=None) for c in clones} == {dataclasses.replace(parent, steps=90)}
    assert parent.practice == config.Practice(4.0, 0.8, 0.25, 4.0, "base")
    plain = parse([PARENT], amplification={"practice": PRACTICE})
    assert plain.settings(plain.members[0])[1].practice.reference is None  # WP-E2's rule by default


def test_revisits_and_the_length_neutral_taste_score():
    """Pilot F's layout: one parent cloned into own and matched revisits and plain training; nothing else differs."""
    arms = {"fro": {"share": 0.33, "source": "own"}, "mat": {"share": 0.33, "source": "matched"}, "pln": None}
    cfg = parse([PARENT] + [{"id": f"p-{a}", "seed": 10 + i, "parent": {"member": "p", "step": 45},
                             "overrides": {"amplification": {"steps": 90, **({"revisit": rv} if rv else {})}}}
                            for i, (a, rv) in enumerate(arms.items())])
    parent, fro, mat, pln = (cfg.settings(m)[1] for m in cfg.members)
    assert parent.revisit is None and pln.revisit is None and parent.taste is None
    assert fro.revisit == config.Revisit(0.33, "own", 4, 0.0) and mat.revisit == config.Revisit(0.33, "matched")
    assert {dataclasses.replace(a, revisit=None) for a in (fro, mat, pln)} == {dataclasses.replace(parent, steps=90)}
    taste = parse([{**PARENT, "overrides": {"amplification": {"taste": {"strength": 0.25, "residualize": True}}}}])
    assert taste.settings(taste.members[0])[1].taste == config.Taste(0.25, 2.0, True)
    assert config.Taste(0.25).residualize is False  # pilot E's score


def test_a_clone_keeps_its_parents_practice_after_step_0():
    parent = {"id": "p", "seed": 1, "context": "refute", "overrides": {"amplification": {"practice": PRACTICE}}}
    changed = {"amplification": {"practice": {**PRACTICE, "eta": 2.0}}}
    with pytest.raises(ValueError, match="practice settings"):
        parse([parent, {"id": "c", "seed": 2, "parent": {"member": "p", "step": 10}, "overrides": changed}])
    with pytest.raises(ValueError, match="practice settings"):  # nor start practising on a record it does not have
        parse([{**parent, "overrides": {}}, {"id": "c", "seed": 2, "parent": {"member": "p", "step": 10},
                                             "overrides": {"amplification": {"practice": PRACTICE}}}])
    cfg = parse([parent, {"id": "c", "seed": 2, "parent": {"member": "p", "step": 0}, "overrides": changed},
                 {"id": "k", "seed": 3, "parent": {"member": "p", "step": 10},
                  "overrides": {"amplification": {"exaggeration": {"gamma": 1, "weight": 1}}}}])
    assert cfg.settings(cfg.members[1])[1].practice.eta == 2.0  # at step 0 nothing has been recorded yet
    assert cfg.settings(cfg.members[2])[1].practice == cfg.settings(cfg.members[0])[1].practice


def test_pilot_c_and_d_configs():
    c = config.load(ROOT / "configs" / "run" / "pilot-c.yaml")  # resumable: no new key, pilot C's update
    assert all((a.exaggeration, a.practice, a.imitate_within_free) == (None, None, False)
               for _, a, _ in map(c.settings, c.members))
    d = config.load(ROOT / "configs" / "run" / "pilot-d.yaml")
    pilot_e, pilot_f = {}, {}  # parent -> arm -> settings
    for run in (c, d):
        for m in run.members:
            arm = m.id.rsplit("-", 1)[-1]
            if arm in ("own", "neu", "rev"):
                pilot_e.setdefault(m.parent, {})[arm] = run.settings(m)[1]
            elif arm in ("fro", "mat", "pln"):
                pilot_f.setdefault((run.name, m.parent), {})[arm] = run.settings(m)[1]
            else:
                assert run.settings(m)[1].taste is None and run.settings(m)[1].revisit is None
    assert sorted((name, p.member, p.step) for name, p in pilot_f) == [
        ("pilot-c", "placebo-adv", 45)] + [("pilot-d", f"adv-{i}", 45) for i in (1, 2, 3)]
    for (name, parent), arms in pilot_f.items():  # nothing else differs from the parent's rule
        run = c if name == "pilot-c" else d
        assert {arm: a.revisit for arm, a in arms.items()} == {
            "fro": config.Revisit(0.33, "own"), "mat": config.Revisit(0.33, "matched"), "pln": None}
        rule = dataclasses.replace(run.settings(run.member(parent.member))[1], steps=90)
        assert all(dataclasses.replace(a, revisit=None) == rule for a in arms.values())
    assert all(a.taste is None for arms in pilot_f.values() for a in arms.values())
    assert sorted(p.member for p in pilot_e) == ["placebo-adv", "relations-adv", "unseeded-adv", "unseeded-b-adv"]
    for parent, arms in pilot_e.items():
        assert parent.step == 45 and {arm: a.taste for arm, a in arms.items()} == {
            "own": config.Taste(0.25), "neu": config.Taste(0.0), "rev": config.Taste(-0.25)}
        rule = dataclasses.replace(c.settings(c.member(parent.member))[1], steps=90)
        assert all(dataclasses.replace(a, taste=None) == rule for a in arms.values())  # nothing else differs
    arms = {}
    for m in [m for m in d.members if m.parent is None]:
        seeding, amp, _ = d.settings(m)
        assert seeding.method == "none" and m.context is None and m.parent is None and amp.imitate_within_free
        assert amp.taste is None and (amp.practice is None or amp.practice.reference is None)
        assert (amp.steps, amp.prompts, amp.rollouts, amp.self_imitation) == (45, 48, 4, 0.0)
        arms.setdefault(m.id.split("-")[0], set()).add((amp.exaggeration, amp.practice))
    ex, pr = config.Exaggeration(1.0, 0.01), config.Practice(4.0, 0.8, 0.25, 4.0)  # weight: see the config
    assert arms == {"adv": {(None, None)}, "exag": {(ex, None)}, "prac": {(None, pr)}, "both": {(ex, pr)}}
    assert len(d.members) == 16 + 9 and d.clonable_steps == (45,) and not d.culling.enabled
    assert {m.seed for m in d.members}.isdisjoint(m.seed for m in c.members)
    assert (d.seed, d.strata, d.lora, d.seeding, d.probes, d.trainer) == (
        c.seed, c.strata, c.lora, c.seeding, c.probes, c.trainer)  # forkable from pilot C, same training prompts


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
