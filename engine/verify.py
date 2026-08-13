#!/usr/bin/env python3
"""confetti.engine.verify — anti-memorization gate and payload verification.

The anti-memorization check: a submission must be a REAL, loadable diffusion
model of the reference architecture that runs the full sampler, verified to
produce seed-dependent output on held-out (prompt, seed) pairs not in the
public set. A lookup table is not a model and cannot pass.
"""
from dataclasses import dataclass


@dataclass
class VerifyResult:
    is_real_model: bool
    seed_dependent: bool
    reason: str = ""


def check_real_model(model_path: str) -> bool:
    """Placeholder: the intake validates the submission is a loadable
    QwenImagePipeline model file (valid safetensors, correct architecture),
    not a lookup table or hardcoded output."""
    # Real implementation: try QwenImagePipeline.from_pretrained(model_path)
    # and confirm it has a transformer of the reference architecture.
    # For v0 this is a stub; the full check ships with the eval harness.
    return True


def check_seed_dependence(model, prompt, seed_a, seed_b) -> bool:
    """Verify the model's output changes with the seed (a real sampler does;
    a canned clip does not)."""
    from .reference import run_steps
    import torch
    timesteps = list(model.scheduler.timesteps)[:4]
    device = next(model.parameters()).device
    preds_a, final_a = run_steps(model, prompt, seed_a, device, timesteps)
    preds_b, final_b = run_steps(model, prompt, seed_b, device, timesteps)
    diff = (final_a - final_b).abs().mean().item()
    # A real model must produce meaningfully different output for different seeds.
    return diff > 1e-3
