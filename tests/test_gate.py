#!/usr/bin/env python3
"""Tests for confeTTI engine. Run with: pytest -q

These tests exercise the gate logic (pure tensor math, no GPU needed) and the
reference serialization.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from engine.gate import GateConfig, gate, per_probe_distance, mean_abs_diff


def test_mean_abs_diff():
    a = torch.zeros(4)
    b = torch.ones(4)
    assert mean_abs_diff(a, b) == 1.0


def test_identical_predictions_pass_with_zero_distance():
    preds = [torch.zeros(4, 4), torch.zeros(4, 4)]
    assert per_probe_distance(preds, preds) == 0.0


def test_gate_passes_identical():
    cfg = GateConfig()
    preds = [torch.zeros(4, 4), torch.zeros(4, 4)]
    ok, d = gate(cfg, preds, preds)
    assert ok is True
    assert d == 0.0


def test_gate_passes_good_quant():
    # Good quant ~3x floor, within a 6x tolerance.
    cfg = GateConfig(noise_floor=0.0012, tolerance_mult=6.0)
    ref = [torch.zeros(4, 4) for _ in range(4)]
    good = [torch.full((4, 4), 0.0038) for _ in range(4)]  # ~3.1x floor
    ok, d = gate(cfg, good, ref)
    assert ok is True


def test_gate_rejects_bad_quant():
    # Bad quant ~10x floor, over a 6x tolerance.
    cfg = GateConfig(noise_floor=0.0012, tolerance_mult=6.0)
    ref = [torch.zeros(4, 4) for _ in range(4)]
    bad = [torch.full((4, 4), 0.0126) for _ in range(4)]  # ~10.3x floor
    ok, d = gate(cfg, bad, ref)
    assert ok is False


def test_gate_calibration_separates():
    # The measured spike numbers: good 3.1x floor passes, bad 10.3x fails.
    cfg = GateConfig()
    ref = [torch.zeros(4, 4) for _ in range(4)]
    good = [torch.full((4, 4), 0.0038) for _ in range(4)]
    bad = [torch.full((4, 4), 0.0126) for _ in range(4)]
    assert gate(cfg, good, ref)[0] is True
    assert gate(cfg, bad, ref)[0] is False
