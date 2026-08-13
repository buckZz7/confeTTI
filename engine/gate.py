#!/usr/bin/env python3
"""confetti.engine.gate — the per-step fidelity gate.

A submission must reproduce the reference's per-step velocity predictions
within a distance tolerance. This is the deterministic, non-trainable,
self-anchored check that makes the competition trustless.

The gate is TOLERANCE-based, not bit-exact: BF16 on a GPU is bit-identical
back-to-back but drifts ~1e-3 per-step across runs (CuBLAS / memory-layout
nondeterminism). So the gate compares distance to a threshold, calibrated to
sit between good and bad quants (spike: good ~3.1x floor, bad ~10.3x floor,
bad/good ratio 3.3x).
"""
from dataclasses import dataclass


@dataclass
class GateConfig:
    # Multiplier on the measured reference noise floor. The tolerance is
    # `noise_floor * tolerance_mult`. Spike found good quants at ~3.1x floor
    # and bad at ~10.3x floor, so ~5-7x cleanly separates them.
    tolerance_mult: float = 6.0
    noise_floor: float = 0.0012  # measured on A100, Qwen-Image-2512, 4 steps


def mean_abs_diff(a, b):
    """Mean absolute difference between two tensors (CPU floats)."""
    return (a - b).abs().mean().item()


def per_probe_distance(submission_preds, ref_preds):
    """Mean per-step abs diff between a submission's predictions and the ref."""
    n = min(len(submission_preds), len(ref_preds))
    if n == 0:
        return float("inf")
    return sum(mean_abs_diff(a, b)
               for a, b in zip(submission_preds, ref_preds)) / n


def gate(config: GateConfig, submission_preds, ref_preds) -> tuple[bool, float]:
    """Return (passes, distance). Passes if distance <= noise_floor * mult."""
    d = per_probe_distance(submission_preds, ref_preds)
    tol = config.noise_floor * config.tolerance_mult
    return (d <= tol), d
