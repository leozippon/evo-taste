"""The blind guard refuses without a committed, unmodified manifest listing the adapter's hash; base is exempt."""

import json
import subprocess

import pytest

from evotaste.evaluate import LEDGER, blind_guard, manifest_path
from evotaste.tasks import BlindReceipt
from evotaste.trainer import adapter_sha256


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.invalid")
    git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README").write_text("x")
    git(tmp_path, "add", "README")
    git(tmp_path, "commit", "-qm", "init")
    return tmp_path


@pytest.fixture
def adapter(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"weights")
    (d / "adapter_config.json").write_text('{"r": 32}')
    return d


def write_manifest(repo, sha, section="ranking"):
    path = manifest_path("r", repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"run": "r", "ranking": [{"lineage": "m1", "sha256": "0" * 64}], "baselines": []}
    manifest[section].append({"lineage": "m0", "sha256": sha})
    path.write_text(json.dumps(manifest))
    return path


def ledger_lines(repo):
    path = repo / LEDGER
    return path.read_text().count("| blind") if path.exists() else 0


def test_refusals(repo, adapter):
    sha = adapter_sha256(adapter)
    with pytest.raises(PermissionError, match="not tracked"):  # no manifest
        blind_guard("blind1", "r", {"m0": adapter}, 4, 1, repo)
    path = write_manifest(repo, sha)
    with pytest.raises(PermissionError, match="not tracked"):  # manifest exists but is not committed
        blind_guard("blind1", "r", {"m0": adapter}, 4, 1, repo)
    git(repo, "add", str(path.relative_to(repo)))
    git(repo, "commit", "-qm", "manifest")
    path.write_text(path.read_text().replace("m0", "m1"))
    with pytest.raises(PermissionError, match="differs"):  # modified after the commit
        blind_guard("blind1", "r", {"m0": adapter}, 4, 1, repo)
    git(repo, "checkout", "--", str(path.relative_to(repo)))
    (adapter / "adapter_model.safetensors").write_bytes(b"other weights")
    with pytest.raises(PermissionError, match="matching content hash"):  # adapter changed since the harvest
        blind_guard("blind2", "r", {"m0": adapter}, 4, 1, repo)
    assert ledger_lines(repo) == 0  # nothing refused was logged as an opening
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text('{"r": 64}')
    with pytest.raises(PermissionError, match="matching content hash"):  # same weights, different LoRA config
        blind_guard("blind2", "r", {"m0": adapter}, 4, 1, repo)
    (adapter / "adapter_config.json").write_text('{"r": 32}')
    receipt = blind_guard("blind2", "r", {"m0": adapter, "base": None}, 4, 1, repo)
    assert isinstance(receipt, BlindReceipt) and receipt.vault == "blind2"
    assert ledger_lines(repo) == 1


def test_base_model_is_exempt_but_logged(repo):
    receipt = blind_guard("blind1", "r", {"base": None}, 8, 2, repo)
    assert receipt.vault == "blind1" and "base model only" in receipt.ledger_line
    assert ledger_lines(repo) == 1
    with pytest.raises(ValueError):
        blind_guard("selection", "r", {"base": None}, 8, 2, repo)


def test_baselines_listed_in_the_manifest_are_accepted(repo, adapter):
    path = write_manifest(repo, adapter_sha256(adapter), section="baselines")
    git(repo, "add", str(path.relative_to(repo)))
    git(repo, "commit", "-qm", "manifest")
    assert blind_guard("blind1", "r", {"long": adapter}, 4, 1, repo).vault == "blind1"


def test_member_roles():
    from evotaste import config

    base = {"name": "t", "model": "m", "seed": 0, "seeding": {"method": "none"}, "amplification": {"steps": 0}}
    cfg = config.parse({**base, "members": [{"id": "a", "seed": 1}, {"id": "long", "seed": 2, "role": "baseline"}]},
                       set())
    assert [m.role for m in cfg.members] == ["member", "baseline"]
    with pytest.raises(ValueError, match="role"):
        config.parse({**base, "members": [{"id": "a", "seed": 1, "role": "judge"}]}, set())
