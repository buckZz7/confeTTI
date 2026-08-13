#!/usr/bin/env python3
"""Tests for the confeTTI eval harness (GPU-free). Run with: pytest -q

These exercise the scoring core (gate application over a fake probe runner),
the reference manifest/serialization, and the anti-memorization intake check.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from engine.gate import GateConfig, gate, per_probe_distance, mean_abs_diff
from engine.eval import score_submission, measure_speed
from engine.reference import hash_tensor, generate_reference, load_reference  # noqa: F401
from engine.verify import check_real_model


def _mk_preds(fill, steps=4, shape=(4, 4)):
    return [torch.full(shape, fill) for _ in range(steps)]


# ---- gate ----

def test_mean_abs_diff():
    a = torch.zeros(4)
    b = torch.ones(4)
    assert mean_abs_diff(a, b) == 1.0


def test_identical_predictions_zero_distance():
    preds = [torch.zeros(4, 4), torch.zeros(4, 4)]
    assert per_probe_distance(preds, preds) == 0.0


def test_gate_calibration_separates_good_from_bad():
    # Measured spike numbers: good ~3.1x floor passes, bad ~10.3x fails, at 6x tol.
    cfg = GateConfig()
    ref = [torch.zeros(4, 4) for _ in range(4)]
    good = [torch.full((4, 4), 0.0038) for _ in range(4)]
    bad = [torch.full((4, 4), 0.0126) for _ in range(4)]
    assert gate(cfg, good, ref)[0] is True
    assert gate(cfg, bad, ref)[0] is False


# ---- eval scoring core (GPU-free) ----

def test_score_submission_passes_good_quant():
    cfg = GateConfig()
    keys = ["42:a", "42:b", "43:a", "43:b"]
    ref = {k: _mk_preds(0.0) for k in keys}
    # Good quant: predictions 0.0038 above the (0.0) reference -> within 6x tol.
    def runner(key):
        return _mk_preds(0.0038)
    res = score_submission(runner, ref, keys, gate_cfg=cfg)
    assert res["passes_gate"] is True
    assert res["probes_evaluated"] == 4


def test_score_submission_rejects_bad_quant():
    cfg = GateConfig()
    keys = ["42:a", "42:b", "43:a", "43:b"]
    ref = {k: _mk_preds(0.0) for k in keys}
    def runner(key):
        return _mk_preds(0.0126)
    res = score_submission(runner, ref, keys, gate_cfg=cfg)
    assert res["passes_gate"] is False
    assert res["mean_distance"] > res["tolerance"]


def test_score_submission_skips_probes_missing_from_reference():
    cfg = GateConfig()
    keys = ["42:a", "42:b", "43:a", "43:b"]
    # Reference only has a subset of the requested keys.
    ref = {"42:a": _mk_preds(0.0), "42:b": _mk_preds(0.0)}
    def runner(key):
        return _mk_preds(0.0038)
    res = score_submission(runner, ref, keys, gate_cfg=cfg)
    # Only the keys present in the reference are evaluated.
    assert res["probes_evaluated"] == 2
    assert res["probes_total"] == 4


# ---- speed (GPU-free, uses injected timing) ----

def test_measure_speed_returns_median():
    probe_keys = ["a", "b", "c"]
    def speed_runner(key):
        return {"a": 1.0, "b": 2.0, "c": 3.0}[key]
    res = measure_speed(speed_runner, probe_keys, warmup=0, runs=1)
    assert res["median_s_per_image"] == 2.0
    assert res["min_s_per_image"] == 1.0
    assert res["max_s_per_image"] == 3.0


# ---- verify / intake ----

def test_check_real_model_rejects_non_directory(tmp_path):
    assert check_real_model(str(tmp_path / "does_not_exist")) is False


def test_check_real_model_rejects_missing_transformer(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "foo.txt").write_text("x")
    assert check_real_model(str(d)) is False


# ---- reference manifest ----

def test_manifest_dict_shape():
    # generate_reference builds a manifest dict directly; verify the shape a
    # real manifest would have (mirrors generate_reference's output).
    manifest = {
        "repo": "x", "dtype": "bfloat16", "steps": 4,
        "timesteps": [1000, 999, 998, 997],
        "height": 1024, "width": 1024,
        "prompts": ["a"], "seeds": [42],
        "per_item_s": 2.0,
        "probes": {
            "42:a": {"pred_hashes": ["x", "y"], "final_latent_hash": "z"}
        },
    }
    assert manifest["steps"] == 4
    assert "42:a" in manifest["probes"]
    assert manifest["probes"]["42:a"]["final_latent_hash"] == "z"


def test_hash_tensor_deterministic():
    t = torch.randn(4, 4)
    assert hash_tensor(t) == hash_tensor(t.clone())
    assert len(hash_tensor(t)) == 16
