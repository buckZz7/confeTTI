#!/usr/bin/env python3
"""confetti.engine.reference — generate, store, and load the immutable reference.

The reference is Qwen-Image-2512's own BF16 output over a public
(seed, prompt, steps) corpus, computed once on an A100-80GB, stored, and
reproducible by anyone. This is the self-anchored anchor the gate measures
every submission against.

Storage layout (two files per reference dir):
  reference.json   manifest: model, dtype, steps, timesteps, corpus, hashes, cost
  reference.pt     torch dict: {probe_key: [per-step velocity prediction tensors]}

The gate needs the reference's actual per-step prediction TENSORS, so we persist
them (not just hashes). The hashes in the manifest are the determinism/verify
check; the tensors are the comparison baseline.
"""
import os
import json
import time
import hashlib

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

DEFAULT_REPO = "Qwen/Qwen-Image-2512"
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_STEPS = 4
NUM_LATENT_CHANNELS = 16

MANIFEST_NAME = "reference.json"
TENSORS_NAME = "reference.pt"
IMAGES_NAME = "reference_images.pt"


def hash_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.float().detach().cpu().numpy().tobytes()).hexdigest()[:16]


def run_steps(pipe, prompt, seed, device, timesteps, height=DEFAULT_HEIGHT,
              width=DEFAULT_WIDTH, encode_cache=None, decode=True):
    """Run the denoising loop, return per-step predictions, final latent, and
    (optionally) the decoded final image.

    This is the verified QwenImagePipeline call contract (from the 2026-08-13
    spike): encode_prompt (no CFG kwarg), prepare_latents (single tensor),
    transformer(hidden_states, timestep=t/1000 on device, encoder_hidden_states,
    encoder_hidden_states_mask, img_shapes), scheduler.step.
    """
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
    final_lat = latents.float().detach().cpu()
    if decode:
        with torch.no_grad():
            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor,
                                    return_dict=False)[0]
            image = (image / 2 + 0.5).clamp(0, 1).float().cpu()
        return preds, final_lat, image
    return preds, final_lat, None


def generate_reference(repo=DEFAULT_REPO, prompts=None, seeds=None, steps=DEFAULT_STEPS,
                       device="cuda", height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH,
                       out_dir=None):
    """Generate the reference over the (seed, prompt) corpus and persist it.

    Returns the output dir containing reference.json + reference.pt.
    """
    prompts = prompts or []
    seeds = seeds or []
    if out_dir is None:
        out_dir = "reference"
    os.makedirs(out_dir, exist_ok=True)

    from diffusers import QwenImagePipeline  # lazy: GPU path only
    pipe = QwenImagePipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    pipe.to(device)
    timesteps = list(pipe.scheduler.timesteps)[:steps]
    cache = {}

    tensors = {}
    images = {}
    hashes = {}
    t0 = time.time()
    for seed in seeds:
        for prompt in prompts:
            key = f"{seed}:{prompt}"
            preds, final_lat, image = run_steps(pipe, prompt, seed, device, timesteps,
                                                height=height, width=width, encode_cache=cache)
            tensors[key] = preds
            images[key] = image  # [1,3,H,W] in [0,1]
            hashes[key] = {
                "pred_hashes": [hash_tensor(p) for p in preds],
                "final_latent_hash": hash_tensor(final_lat),
                "final_image_hash": hash_tensor(image),
            }
    per_item_s = (time.time() - t0) / max(len(tensors), 1)

    manifest = {
        "repo": repo,
        "dtype": "bfloat16",
        "steps": steps,
        "timesteps": [int(t) for t in timesteps],
        "height": height,
        "width": width,
        "prompts": prompts,
        "seeds": seeds,
        "per_item_s": round(per_item_s, 3),
        "probes": hashes,
    }
    with open(os.path.join(out_dir, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)
    torch.save({"probes": tensors}, os.path.join(out_dir, TENSORS_NAME))
    torch.save({"images": images}, os.path.join(out_dir, IMAGES_NAME))
    return out_dir


def load_reference(ref_dir):
    """Load a reference dir, return (manifest, tensors, images).

    manifest: dict. tensors: {key: [per-step pred tensors]}.
    images: {key: [1,3,H,W] image tensor in [0,1]}.
    """
    with open(os.path.join(ref_dir, MANIFEST_NAME)) as f:
        manifest = json.load(f)
    tensors = torch.load(os.path.join(ref_dir, TENSORS_NAME),
                         map_location="cpu", weights_only=True)["probes"]
    images = torch.load(os.path.join(ref_dir, IMAGES_NAME),
                        map_location="cpu", weights_only=True)["images"]
    return manifest, tensors, images


def reference_probe_keys(manifest):
    """The canonical probe keys (seed:prompt) for this reference."""
    return list(manifest["probes"].keys())
