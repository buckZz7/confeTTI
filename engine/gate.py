#!/usr/bin/env python3
"""confetti.engine.gate — the output-fidelity gate.

A submission must reproduce the reference's FINAL IMAGE within a perceptual
distance tolerance on the public (prompt, seed) corpus. This is what makes the
competition able to reward BOTH quantization AND distillation:

  - A per-step velocity gate would reject distillation (a distilled 4-step
    model legitimately predicts differently per step, ~210x off, even though
    its output image is good). Measured on the 2026-08-13 spike.
  - A final-LATENT gate also fails (distilled latents are ~0.207 off even when
    the image is perceptually close).
  - A final-IMAGE perceptual gate (LPIPS) is the only option that accepts
    legit speed recipes (quant + distill) while still rejecting collapsed ones.

LPIPS is a frozen, non-trainable network (VGG-based) — not an LLM/VLM judge and
not trained on the eval corpus, so it stays deterministic and ungameable-by-
overfit. The anti-memorization gate (verify.py) prevents lookup-table attacks
on the public corpus.

The gate is TOLERANCE-based. Tolerance sits between good and bad recipe LPIPS
scores, calibrated on-box per the spike.
"""
from dataclasses import dataclass

_lpips = None


def _get_lpips():
    """Lazy-load the frozen LPIPS network (CPU-safe, small)."""
    global _lpips
    if _lpips is None:
        import lpips
        # net="alex" is the default LPIPS backend (VGG features). Frozen.
        _lpips = lpips.LPIPS(net="alex")
        _lpips.eval()
        for p in _lpips.parameters():
            p.requires_grad_(False)
    return _lpips


def _normalize(img):
    """Resize a [1,3,H,W] image (assumed in [0,1]) to LPIPS input (-1,1, 256px)."""
    import torch
    import torch.nn.functional as F
    # LPIPS expects [-1, 1]. Image tensors are stored in [0,1].
    x = img * 2.0 - 1.0
    if x.shape[-1] != 256:
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    return x


@dataclass
class GateConfig:
    # LPIPS tolerance between good and bad recipes. Calibrated on an A100-80GB
    # (2026-08-13): ref-vs-ref 0.000, good distilled (Lightning) 0.487,
    # collapsed (corrupted) 1.194. 0.85 sits between with ~0.36 margin each way.
    tolerance: float = 0.85


def image_distance(sub_img, ref_img) -> float:
    """LPIPS perceptual distance between two [1,3,H,W] image tensors ([0,1])."""
    net = _get_lpips()
    a = _normalize(sub_img)
    b = _normalize(ref_img)
    with __import__("torch").no_grad():
        return float(net(a, b).item())


def gate(config: GateConfig, sub_img, ref_img) -> tuple[bool, float]:
    """Return (passes, lpips_distance). Passes if distance <= tolerance."""
    d = image_distance(sub_img, ref_img)
    return (d <= config.tolerance), d
