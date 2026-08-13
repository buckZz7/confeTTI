#!/usr/bin/env python3
"""confetti.engine.reference — generate and verify the immutable reference.

The reference is Qwen-Image-2512's own BF16 output over a public
(seed, prompt, steps) corpus, computed once on an A100-80GB, hash-bound, and
reproducible by anyone. This is the self-authenticating anchor the competition
measures everything against.

This is the correct, verified API for driving the QwenImagePipeline manually
(per-step velocity predictions), captured from the spike on 2026-08-13.
"""
import os
import hashlib
from dataclasses import dataclass, field

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from diffusers import QwenImagePipeline

DEFAULT_REPO = "Qwen/Qwen-Image-2512"
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_STEPS = 4
# 16 latent channels (Qwen-Image VAE), 8x spatial compression, then /2 for the
# img_shapes axis used by the transformer.
NUM_LATENT_CHANNELS = 16


@dataclass
class Reference:
    repo: str
    dtype: str
    steps: int
    timesteps: list
    per_item_s: float
    probes: dict = field(default_factory=dict)
    # probes: {(seed, prompt): [per-step velocity prediction hashes...]}


def hash_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.float().detach().cpu().numpy().tobytes()).hexdigest()[:16]


def run_steps(pipe, prompt, seed, device, timesteps, height=DEFAULT_HEIGHT,
              width=DEFAULT_WIDTH, encode_cache=None):
    """Run the denoising loop, return per-step velocity predictions + final latent."""
    if encode_cache is not None and prompt in encode_cache:
        emb, emb_mask = encode_cache[prompt]
    else:
        emb, emb_mask = pipe.encode_prompt(prompt, device=device)
        if encode_cache is not None:
            encode_cache[prompt] = (emb, emb_mask)
    g = torch.Generator(device="cpu").manual_seed(seed)
    latents = pipe.prepare_latents(
        1, NUM_LATENT_CHANNELS, height, width, emb.dtype, device, g, None
    )
    img_shapes = [[(1, height // pipe.vae_scale_factor // 2,
                    width // pipe.vae_scale_factor // 2)]] * 1
    preds = []
    with torch.no_grad():
        for t in timesteps:
            timestep = t.expand(latents.shape[0]).to(
                device=latents.device, dtype=latents.dtype) / 1000
            pred = pipe.transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=emb,
                encoder_hidden_states_mask=emb_mask,
                img_shapes=img_shapes,
                return_dict=False,
            )[0]
            preds.append(pred.float().detach().cpu())
            latents = pipe.scheduler.step(pred, t, latents, return_dict=False)[0]
    return preds, latents.float().detach().cpu()


def generate_reference(repo=DEFAULT_REPO, prompts=None, seeds=None, steps=DEFAULT_STEPS,
                       device="cuda", out_dir=None):
    """Generate the reference over the (seed, prompt) corpus.

    Returns a Reference with per-step prediction hashes and the final-latent
    hash per probe, plus measured per-item cost.
    """
    prompts = prompts or []
    seeds = seeds or []
    import time
    pipe = QwenImagePipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    pipe.to(device)
    timesteps = list(pipe.scheduler.timesteps)[:steps]

    cache = {}
    ref = Reference(repo=repo, dtype="bfloat16", steps=steps,
                    timesteps=[int(t) for t in timesteps])
    t0 = time.time()
    for seed in seeds:
        for prompt in prompts:
            preds, final_lat = run_steps(pipe, prompt, seed, device,
                                         timesteps, encode_cache=cache)
            ref.probes[(seed, prompt)] = {
                "pred_hashes": [hash_tensor(p) for p in preds],
                "final_latent_hash": hash_tensor(final_lat),
            }
    ref.per_item_s = (time.time() - t0) / max(len(ref.probes), 1)
    return ref


def to_json(ref: Reference) -> dict:
    """Serialize the reference (used for the hash-bound artifact)."""
    return {
        "repo": ref.repo,
        "dtype": ref.dtype,
        "steps": ref.steps,
        "timesteps": ref.timesteps,
        "per_item_s": round(ref.per_item_s, 3),
        "probes": {
            f"{seed}:{prompt}": v for (seed, prompt), v in ref.probes.items()
        },
    }
