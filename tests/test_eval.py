#!/usr/bin/env python3
"""Tests for the confeTTI eval harness (GPU-free). Run with: pytest -q

These exercise the scoring core (output-fidelity gate over a fake probe runner),
the reference manifest/serialization, and the anti-memorization intake check.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from engine.gate import GateConfig, gate, image_distance, _normalize
from engine.eval import score_submission, measure_speed
from engine.reference import hash_tensor  # noqa: F401
from engine.verify import check_real_model


def _img(fill, shape=(1, 3, 16, 16)):
    return torch.full(shape, fill)


# ---- gate ----

def test_identical_images_zero_distance():
    # The same image should be near-zero LPIPS distance.
    a = torch.full((1, 3, 256, 256), 0.5)
    d = image_distance(a, a.clone())
    assert d < 0.001


def test_different_images_large_distance():
    # Very different images (all black vs all white) should be high LPIPS.
    black = torch.zeros(1, 3, 256, 256)
    white = torch.ones(1, 3, 256, 256)
    d = image_distance(black, white)
    assert d > 0.5


def test_gate_tolerance():
    cfg = GateConfig(tolerance=0.30)
    a = torch.full((1, 3, 256, 256), 0.5)
    ok, d = gate(cfg, a, a.clone())
    assert ok is True
    assert d < 0.01


# ---- eval scoring core (GPU-free) ----

def test_score_submission_passes_close_image():
    cfg = GateConfig(tolerance=0.30)
    keys = ["42:a", "42:b", "43:a", "43:b"]
    ref = {k: torch.full((1, 3, 256, 256), 0.5) for k in keys}
    def runner(key):
        return torch.full((1, 3, 256, 256), 0.5)
    res = score_submission(runner, ref, keys, gate_cfg=cfg)
    assert res["passes_gate"] is True
    assert res["probes_evaluated"] == 4


def test_score_submission_rejects_different_image():
    cfg = GateConfig(tolerance=0.30)
    keys = ["42:a", "42:b"]
    ref = {k: torch.zeros(1, 3, 256, 256) for k in keys}
    def runner(key):
        return torch.ones(1, 3, 256, 256)  # all white vs ref all black
    res = score_submission(runner, ref, keys, gate_cfg=cfg)
    assert res["passes_gate"] is False


def test_score_submission_skips_probes_missing_from_reference():
    cfg = GateConfig(tolerance=0.30)
    keys = ["42:a", "42:b", "43:a", "43:b"]
    ref = {"42:a": torch.full((1, 3, 256, 256), 0.5),
           "42:b": torch.full((1, 3, 256, 256), 0.5)}
    def runner(key):
        return torch.full((1, 3, 256, 256), 0.5)
    res = score_submission(runner, ref, keys, gate_cfg=cfg)
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
    manifest = {
        "repo": "x", "dtype": "bfloat16", "steps": 4,
        "timesteps": [1000, 999, 998, 997],
        "height": 1024, "width": 1024,
        "prompts": ["a"], "seeds": [42],
        "per_item_s": 2.0,
        "probes": {
            "42:a": {"pred_hashes": ["x", "y"], "final_latent_hash": "z",
                     "final_image_hash": "w"}
        },
    }
    assert manifest["steps"] == 4
    assert "42:a" in manifest["probes"]
    assert manifest["probes"]["42:a"]["final_image_hash"] == "w"


def test_hash_tensor_deterministic():
    t = torch.randn(4, 4)
    assert hash_tensor(t) == hash_tensor(t.clone())
    assert len(hash_tensor(t)) == 16
