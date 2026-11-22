"""Analysis tools: variance decomposition, lineage classifier, answer masking, and runs in progress."""

import fcntl
import re

import numpy as np
import pandas as pd

from evotaste.analysis import report, style
from evotaste.analysis.style import NAMES, classifier_text, identifiability, sharpness, variance_components
from evotaste.records import append_jsonl, write_json, write_jsonl_gz


def test_variance_components_recover_a_planted_member_effect_and_nothing_without_one():
    rng = np.random.default_rng(0)
    m, p = 8, 2000
    prompt = rng.normal(0, 3, p)  # large prompt effects must not leak into the member component
    member = np.linspace(-1, 1, m)
    noise = rng.normal(0, 1, (m, p))
    planted = variance_components(member[:, None] + prompt[None, :] + noise)
    assert abs(planted["between"] - member.var(ddof=1)) < 0.05
    assert abs(planted["within"] - 1) < 0.05 and planted["p"] < 1e-10
    null = variance_components(prompt[None, :] + noise)
    assert abs(null["between"]) < 0.01 and abs(null["within"] - 1) < 0.05 and null["p"] > 0.001


def _feature_rows(rng, lineages, uids, planted):
    """Probe-response feature rows: 'tokens' carries prompt effects plus planted[lineage]; other features are noise."""
    prompt = dict(zip(uids, rng.normal(0, 50, len(uids))))
    return [{"lineage": lin, "step": 0, "suite": "train", "uid": u,
             **{f: rng.normal() for f in NAMES}, "tokens": 200 + prompt[u] + planted[lin] + rng.normal(0, 10)}
            for lin in lineages for u in uids]


def test_sharpness_in_base_sd_units_recovers_the_planted_effect():
    rng = np.random.default_rng(1)
    uids = [f"u{i}" for i in range(1500)]
    planted = {"m0": 0.0, "m1": 20.0, "m2": 40.0, "m3": 60.0}
    members = pd.DataFrame(_feature_rows(rng, list(planted), uids, planted)).sample(frac=1, random_state=0)
    base = pd.DataFrame(_feature_rows(rng, ["base"], uids, {"base": 0.0}))
    rows = sharpness(members, base).query("suite == 'all'").set_index("feature")
    expected = np.var(list(planted.values()), ddof=1) / base["tokens"].std(ddof=1) ** 2
    assert abs(rows.loc["tokens", "between"] / expected - 1) < 0.1
    noise = rows.drop(index=["tokens", "mean"])
    assert noise["between"].abs().max() < 0.01 and (noise["p"] < 0.001).sum() <= 1


STYLES = {  # one habit per lineage, in words that occur in no prompt content
    "m0": "Let me verify every condition carefully.",
    "m1": "Case one fails, so consider the next case.",
    "m2": "### Plan\n- outline\n- execute",
    "m3": "Hmm, wait, actually that was a mistake.",
}
CONTENT = {"train": "matrix queen maze token grid", "selection": "polynomial integral root factor",
           "foundation": "apples price shop farmer"}


def _responses(rng, habit_rate=0.8, answer_of=None, digits_of=None):
    """Every lineage answers the same prompts of three suites with different content."""
    rows = []
    for suite, n in [("train", 60), ("selection", 40), ("foundation", 40)]:
        for i in range(n):
            content = " ".join(rng.choice(CONTENT[suite].split(), 10))
            for lin, habit in STYLES.items():
                if answer_of is None:
                    text = f"{content} {rng.integers(1000)}\n{habit if rng.random() < habit_rate else ''}\n" \
                           f"<answer>{rng.integers(100)}</answer>"
                else:  # identical text except a lineage-specific answer span and lineage-specific digits
                    text = f"{content} {digits_of[lin]}\n<answer>{answer_of[lin]} {digits_of[lin]}</answer>"
                rows.append({"lineage": lin, "step": 0, "suite": suite, "uid": f"{suite}/{i}",
                             "ctext": classifier_text(text)})
    return pd.DataFrame(rows)


def _heldout(rows: pd.DataFrame) -> pd.Series:
    return rows[(rows["train_suite"] == "train") & (rows["test_suite"] == "selection+foundation")].iloc[0]


def test_classifier_identifies_planted_styles_across_suites():
    rows, _, ngrams = identifiability(_responses(np.random.default_rng(2)), {lin: lin for lin in STYLES})
    r = _heldout(rows)
    assert r["chance"] == 0.25 and r["accuracy"] > 0.7 and r["ci_lo"] > 0.5 and r["p_binom"] < 1e-10
    assert set(ngrams["lineage"]) == set(STYLES)


def test_classifier_is_at_chance_on_shuffled_labels():
    df = _responses(np.random.default_rng(3))
    df["lineage"] = df["lineage"].sample(frac=1, random_state=0).to_numpy()
    r = _heldout(identifiability(df, {lin: lin for lin in STYLES})[0])
    assert r["ci_lo"] < 0.25 < r["ci_hi"] and abs(r["accuracy"] - 0.25) < 0.08


def test_answer_spans_and_digits_are_removed_before_classification():
    text = "Step 1: add 12 and 30.\n<answer>secret 42</answer> done\n<answer>hidden 7"
    assert classifier_text(text) == "Step 0: add 0 and 0.\n  done\n "
    assert not re.search(r"[1-9]", classifier_text(text))
    # Lineages that differ only inside their answer spans and in their digits cannot be told apart.
    answer_of = {lin: f"word{chr(97 + i)}" for i, lin in enumerate(STYLES)}
    digits_of = {lin: str(i + 1) * (i + 2) for i, lin in enumerate(STYLES)}
    df = _responses(np.random.default_rng(4), answer_of=answer_of, digits_of=digits_of)
    assert df.groupby("uid")["ctext"].nunique().max() == 1
    r = _heldout(identifiability(df, {lin: lin for lin in STYLES})[0])
    assert r["accuracy"] == 0.25


# -- a run in progress ----------------------------------------------------------------------------------------------

def _probe(step, suite, family, i, text, reward):
    return {"step": step, "suite": suite, "uid": f"{family}/0/{i}", "family": family, "question": f"q {family} {i}",
            "answer": "a",
            "text": text, "completion_tokens": len(text.split()) + 1, "finish_reason": "stop", "extracted": "a",
            "score": reward, "reward": reward, "parsed": True, "truncated": False, "score_error": None, "logprob": -1.0}


PROBE_SET = [("train", "maze", 3), ("train", "n_queens", 3), ("selection", "lcm", 3), ("foundation", "gsm8k_train", 4)]


def _probes(step, habit, rng):
    return [_probe(step, suite, family, i, f"{habit} working on {family} {rng.integers(99)}\n<answer>a</answer>",
                   float(rng.random() < 0.5)) for suite, family, n in PROBE_SET for i in range(n)]


def _state(path, lineage, slot, arm, scaffold, status, step, generation=0, cull=None, replaced_by=None):
    write_json(path / "members" / lineage / "state.json", {
        "id": lineage, "slot": slot, "generation": generation, "arm": arm, "scaffold": scaffold, "seed": 1,
        "status": status, "step": step, "probes": [], "cull_reason": cull, "replaced_by": replaced_by})


def _amp(path, lineage, steps):
    for t in steps:
        append_jsonl(path / "members" / lineage / "history.jsonl", {
            "phase": "amp", "step": t, "time": f"2026-11-22T10:{t:02d}:00+00:00", "reward": 0.2 + 0.01 * t,
            "reward_by_family": {"maze": 0.1, "n_queens": 0.3} if t % 2 else {"maze": 0.2},
            "completion_tokens": 100.0 + t, "parsed": 0.9, "truncated": 0.05, "groups": 5, "prompts": 8,
            "check": {"token_mean_abs": 0.005 + 0.001 * t}})


def test_tools_run_on_a_run_in_progress(tmp_path, capsys):
    rng = np.random.default_rng(5)
    run = tmp_path / "runs" / "ip"
    (run / "seeding").mkdir(parents=True)
    (run / "scaffolds.yaml").write_text("preamble: P\ncontrol:\n  cot: {text: c}\nscaffolds:\n  verify: {text: v}\n")
    (run / "config.yaml").write_text(
        f"name: ip\nmodel: tiny\nseed: 0\nscaffolds: {run / 'scaffolds.yaml'}\namplification: {{steps: 30}}\n"
        "probes: {every: 10}\nmembers:\n- {id: a, arm: plain, seed: 1}\n"
        "- {id: b, arm: scaffold, scaffold: verify, seed: 2}\n- {id: c, arm: scaffold, scaffold: verify, seed: 3}\n"
        "- {id: d, arm: perturb, seed: 4, overrides: {amplification: {lr: 2.0e-4}}}\n")
    write_jsonl_gz(run / "seeding" / "plain.jsonl.gz", [
        {"uid": f"maze/1/{i}", "family": "maze", "prompt_ids": [1], "completion_ids": [1, 2, 3],
         "text": "Plain pool answer.\n<answer>a</answer>", "teacher_logprob": -1.0} for i in range(4)])
    write_json(run / "seeding" / "plain.json", {"teacher": "plain", "prompts": 8, "samples": 32, "correct": 4,
                                                "prompts_with_correct": 4,
                                                "by_family": {"maze": {"correct": 4}, "n_queens": {"correct": 0}}})
    write_jsonl_gz(run / "seeding" / "verify.jsonl.gz", [])  # pool being written: no summary yet
    base = [{**r, "step": None} for r in _probes(None, "Base reply.", rng)]
    base[6]["question"] = "another question"  # lcm/0/0: the base model saw a different question under this uid
    write_jsonl_gz(run / "base" / "probe_responses.jsonl.gz", base)
    write_json(run / "base" / "probe_summary.json", {
        "n": len(base), "accuracy": {"train": 0.5, "selection": 0.4, "foundation": 0.6}, "parsed": 1.0,
        "truncated": 0.0, "completion_tokens": 7.0})
    # a: active with a running worker: 14 steps (4 past its step-10 checkpoint, step 3 written twice), a line cut
    # short by a kill, and a probe file after the checkpoint (killed between probe and checkpoint).
    _state(run, "a", "a", "plain", None, "active", 10)
    _amp(run, "a", [0, 1, 2, 3, 3, *range(4, 14)])
    with open(run / "members" / "a" / "history.jsonl", "a") as f:
        f.write('{"phase": "amp", "st')
    for s in (0, 10, 20):
        write_jsonl_gz(run / "members" / "a" / "probe_responses" / f"step_{s:06d}.jsonl.gz",
                       _probes(s, "Let me verify this.", rng))
    # b: still seeding; c: not started; d: culled at step 10 with two uncommitted steps after it, replaced by d~1,
    # which has its step-0 probe and two uncommitted steps but no running worker.
    _state(run, "b", "b", "scaffold", "verify", "new", 0)
    append_jsonl(run / "members" / "b" / "history.jsonl", {"phase": "seed", "step": 0})
    _state(run, "d", "d", "perturb", None, "culled", 10, cull="fewer than half of the probe answers are parseable",
           replaced_by="d~1")
    _amp(run, "d", range(12))
    _state(run, "d~1", "d", "perturb", None, "active", 0, generation=1)
    _amp(run, "d~1", range(2))
    for lineage, steps in (("d", (0, 10)), ("d~1", (0,))):
        for s in steps:
            write_jsonl_gz(run / "members" / lineage / "probe_responses" / f"step_{s:06d}.jsonl.gz",
                           _probes(s, "Case one, then case two.", rng))

    (run / "locks").mkdir()
    with open(run / "locks" / "a.lock", "w") as lock:  # the worker of slot a holds its lock, as evotaste.worker does
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report.main([str(run), "--families", "--out", str(tmp_path / "report")])
        out = capsys.readouterr().out
        style.main([str(run), "--out", str(tmp_path / "style")])
        style_out = capsys.readouterr().out
    for expected in ("pending", "new", "culled d at step 10", "replaced by d~1", "14/30", "dropped as unpaired"):
        assert expected in out
    members = pd.read_csv(tmp_path / "report" / "members.csv")
    assert list(members["lineage"]) == ["a", "b", "c", "d", "d~1"]
    assert list(members["steps_done"]) == [14, 0, 0, 10, 0]
    assert list(members["steps_uncommitted"]) == [4, 0, 0, 0, 0]
    assert list(members["worker"]) == [True, False, False, False, False]
    assert members.loc[3, "amp_lr"] == 2e-4
    probes = pd.read_csv(tmp_path / "report" / "probes.csv")
    assert set(zip(probes["lineage"], probes["step"])) == {("a", 0), ("a", 10), ("d", 0), ("d", 10), ("d~1", 0)}
    assert set(probes.loc[probes["unpaired"] > 0, "suite"]) == {"selection"}
    assert (probes.query("suite == 'selection'")["instances"] == 2).all()
    assert list(pd.read_csv(tmp_path / "report" / "seeding.csv")["teacher"]) == ["plain"]

    assert "not possible" in style_out and "step 0: 1 (lcm)" in style_out
    ident = pd.read_csv(tmp_path / "style" / "identifiability.csv")
    assert set(ident["members"]) == {2, 3}  # step 0: a, d, d~1; step 10: a, d
    assert "lcm/0/0" not in set(pd.read_csv(tmp_path / "style" / "features.csv.gz").query("source == 'member'")["uid"])
