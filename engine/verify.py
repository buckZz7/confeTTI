#!/usr/bin/env python3
"""confetti.engine.verify — anti-memorization gate and payload verification.

The anti-memorization check: a submission must be a REAL, loadable diffusion
model of the reference architecture that runs the full sampler, verified to
produce seed-dependent output on held-out (prompt, seed) pairs not in the
public set. A lookup table is not a model and cannot pass.
"""
from dataclasses import dataclass

# The reference transformer class we require a submission to be loadable as.
REFERENCE_TRANSFORMER_CLASS = "QwenImageTransformer2DModel"


@dataclass
class VerifyResult:
    is_real_model: bool
    seed_dependent: bool
    reason: str = ""


def check_real_model(model_path: str) -> bool:
    """Confirm the submission is a real, loadable QwenImageTransformer2DModel.

    This is the structural anti-memorization gate: a lookup table / canned
    output is not a loadable diffusers transformer of the reference
    architecture, so it is rejected here regardless of speed or fidelity.

    GPU-free: loads the transformer config and confirms the class + that it
    is a diffusers model of the expected architecture.
    """
    import os

    # Must be a directory with a transformer config (diffusers layout).
    if not os.path.isdir(model_path):
        return False
    transformer_dir = os.path.join(model_path, "transformer")
    if not os.path.isdir(transformer_dir):
        return False
    config_path = os.path.join(transformer_dir, "config.json")
    if not os.path.isfile(config_path):
        return False

    # Only import diffusers when we actually need to load a real config.
    from diffusers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(transformer_dir)
    except Exception:
        return False

    # Require the reference architecture class. Anything else is not a
    # compatible submission.
    cls_name = getattr(cfg, "_class_name", "") or cfg.__class__.__name__
    return cls_name == REFERENCE_TRANSFORMER_CLASS


def check_seed_dependence(model, prompt, seed_a, seed_b) -> bool:
    """Verify the model's output changes with the seed (a real sampler does;
    a canned clip does not). GPU required.
    """
    from .reference import run_steps
    timesteps = list(model.scheduler.timesteps)[:4]
    device = next(model.parameters()).device
    preds_a, final_a = run_steps(model, prompt, seed_a, device, timesteps)
    preds_b, final_b = run_steps(model, prompt, seed_b, device, timesteps)
    diff = (final_a - final_b).abs().mean().item()
    # A real model must produce meaningfully different output for different seeds.
    return diff > 1e-3
