import hashlib
import importlib.util
from pathlib import Path
import socket

import pytest
import torch

spec = importlib.util.spec_from_file_location("smoke", Path(__file__).parents[1] / "scripts/m1_1_satlas_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Network access in offline test")
    monkeypatch.setattr(socket.socket, "connect", deny)


def test_digest_rejects_missing_and_mismatching_expected_hash(tmp_path):
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"fixture")
    for expected in (None, "invalid", "0" * 64):
        with pytest.raises(ValueError):
            smoke.checkpoint_digest(path, expected)
    expected = hashlib.sha256(b"fixture").hexdigest()
    assert smoke.checkpoint_digest(path, expected) == expected


def test_synthetic_reproducible_and_uniform():
    first, masks = smoke.synthetic(torch, 8, 12)
    second, repeated = smoke.synthetic(torch, 8, 12)
    assert torch.equal(first, second)
    assert first.shape == (1, 3, 8, 8) and first.dtype == torch.float32
    assert 0 <= first.min() <= first.max() <= 1
    for name, mask in masks.items():
        assert torch.equal(mask, repeated[name])
        assert mask.shape == (1, 8, 8) and mask.dtype == torch.long
    assert masks["background"].count_nonzero() == 0
    assert torch.all(masks["canopy"] == 1)
    assert set(masks["mixed"].unique().tolist()) == {0, 1}


def test_output_validation_rejects_invalid_probabilities_and_loss():
    probabilities = torch.full((1, 2, 4, 4), 0.5)
    smoke.validate_output(torch, probabilities, torch.tensor(1.0), 4)
    for invalid in (probabilities[:, :1], probabilities * 0, probabilities * float("nan")):
        with pytest.raises(ValueError):
            smoke.validate_output(torch, invalid, torch.tensor(1.0), 4)
    with pytest.raises(ValueError):
        smoke.validate_output(torch, probabilities, torch.tensor(float("nan")), 4)


def test_gradient_validation_requires_propagation():
    layer = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError):
        smoke.gradient_evidence(torch, layer)
    layer(torch.ones(1, 2)).sum().backward()
    assert smoke.gradient_evidence(torch, layer)["finite_nonzero_parameters"] == 2
    layer.weight.grad.fill_(float("nan"))
    with pytest.raises(ValueError):
        smoke.gradient_evidence(torch, layer)


def test_missing_checkpoint_is_not_verified(tmp_path, capsys):
    assert smoke.main(["--init", "checkpoint", "--checkpoint", str(tmp_path / "absent"),
                       "--sha256", "0" * 64]) == 2
    assert '"status": "NOT_VERIFIED"' in capsys.readouterr().out


def test_digest_failure_precedes_deserialization(tmp_path, monkeypatch, capsys):
    path = tmp_path / "untrusted.pth"
    path.write_bytes(b"not a pickle")
    def deny_load(*args, **kwargs):
        raise AssertionError("Deserialization before hash validation")
    monkeypatch.setattr(torch, "load", deny_load)
    assert smoke.main(["--init", "checkpoint", "--checkpoint", str(path),
                       "--sha256", "0" * 64]) == 1
    assert "SHA-256 mismatch" in capsys.readouterr().out


def test_official_fpn_load_is_strict():
    from satlaspretrain_models import Model
    from satlaspretrain_models.models.fpn import FPN
    channels = [[4, 8], [8, 16]]
    source = FPN(channels)
    weights = {"intermediates.0." + key: value for key, value in source.state_dict().items()}
    loaded = Model._initialize_fpn(None, channels, weights)
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in source.state_dict().items())
    weights.pop(next(iter(weights)))
    with pytest.raises(RuntimeError, match="Missing key"):
        Model._initialize_fpn(None, channels, weights)


def test_checkpoint_load_uses_safe_cpu_options(tmp_path, monkeypatch, capsys):
    path = tmp_path / "fixture.pth"
    path.write_bytes(b"fixture")
    calls = []
    def load_fixture(path, **kwargs):
        calls.append(kwargs)
        return {}
    monkeypatch.setattr(torch, "load", load_fixture)
    assert smoke.main(["--init", "checkpoint", "--checkpoint", str(path),
                       "--sha256", hashlib.sha256(b"fixture").hexdigest()]) == 1
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert "Expected official nonempty tensor state dict" in capsys.readouterr().out
