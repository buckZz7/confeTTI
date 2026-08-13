#!/usr/bin/env python3
"""confetti.engine.eval — the eval harness.

Evaluates a submission (a model + runtime recipe) against the immutable
reference:

  1. Anti-memorization intake check (real loadable model + seed-dependence).
  2. Output-fidelity gate: final image within LPIPS tolerance of the
     reference's final image over the public corpus.
  3. Speed: wall-clock seconds per image on the eval box.

The scoring core (`score_submission`) is decoupled from the GPU runner so it is
testable without a GPU — tests inject a fake probe runner that returns canned
image tensors.
"""
import os
import time
import json
import argparse

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from .gate import GateConfig, gate
from .reference import load_reference, DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_STEPS
from .verify import check_real_model


def score_submission(probe_runner, ref_images, probe_keys, gate_cfg=None):
    """Score a submission given a probe_runner that returns final images.

    probe_runner(key) -> torch.Tensor  ([1,3,H,W] image in [0,1])
    ref_images: {key: image tensor} from the reference
    probe_keys: canonical probe keys to evaluate over

    Returns a result dict: passes_gate, mean_distance, tolerance,
    probes_evaluated, probes_total, gate_time_s.
    """
    gate_cfg = gate_cfg or GateConfig()
    distances = []
    n_probes = len(probe_keys)
    start = time.time()
    for key in probe_keys:
        if key not in ref_images:
            continue
        sub_img = probe_runner(key)
        ref_img = ref_images[key]
        ok, d = gate(gate_cfg, sub_img, ref_img)
        distances.append(d)
    total = time.time() - start

    mean_d = (sum(distances) / len(distances)) if distances else float("inf")
    passes = mean_d <= gate_cfg.tolerance if distances else False

    return {
        "passes_gate": passes,
        "mean_distance": round(mean_d, 5),
        "tolerance": gate_cfg.tolerance,
        "probes_evaluated": len(distances),
        "probes_total": n_probes,
        "gate_time_s": round(total, 2),
    }


def build_probe_runner(pipe, device, timesteps, height, width, seeds, prompts):
    """GPU probe runner: run the submission model over each (seed, prompt) probe.

    Returns a closure probe_runner(key) -> final image tensor, and a
    speed_runner(key) -> wall-clock seconds.
    """
    from .reference import run_steps
    cache = {}

    def runner(key):
        seed_str, prompt = key.split(":", 1)
        seed = int(seed_str)
        preds, _, image = run_steps(pipe, prompt, seed, device, timesteps,
                                    height=height, width=width, encode_cache=cache)
        return image

    def speed_runner(key):
        seed_str, prompt = key.split(":", 1)
        seed = int(seed_str)
        t0 = time.time()
        run_steps(pipe, prompt, seed, device, timesteps,
                  height=height, width=width, encode_cache=cache)
        return time.time() - t0

    return runner, speed_runner


def measure_speed(speed_runner, probe_keys, warmup=1, runs=3):
    """Wall-clock seconds per image, median across probes after warmup."""
    for _ in range(warmup):
        speed_runner(probe_keys[0])

    times = []
    for key in probe_keys:
        for _ in range(runs):
            times.append(speed_runner(key))
    times.sort()
    med = times[len(times) // 2] if times else 0.0
    return {
        "median_s_per_image": round(med, 4),
        "min_s_per_image": round(min(times), 4) if times else 0.0,
        "max_s_per_image": round(max(times), 4) if times else 0.0,
        "samples": len(times),
    }


def run_eval(submission_path, reference_dir, device="cuda",
             height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, steps=DEFAULT_STEPS):
    """End-to-end eval of a submission against a reference. GPU required."""
    from diffusers import QwenImagePipeline  # lazy: GPU path only

    # 1. Intake: must be a real, loadable model.
    if not check_real_model(submission_path):
        return {"passes_gate": False, "error": "not a real loadable model"}

    manifest, tensors, ref_images = load_reference(reference_dir)
    probe_keys = list(manifest["probes"].keys())
    timesteps = [torch.tensor(t, dtype=torch.float32) for t in manifest["timesteps"]]

    # 2. Load submission pipeline.
    pipe = QwenImagePipeline.from_pretrained(submission_path, torch_dtype=torch.bfloat16)
    pipe.to(device)

    runner, speed_runner = build_probe_runner(
        pipe, device, timesteps, height, width, manifest["seeds"], manifest["prompts"]
    )

    # 3. Gate.
    result = score_submission(runner, ref_images, probe_keys)
    if not result["passes_gate"]:
        result["speed"] = None
        return result

    # 4. Speed (only if gate passed — collapsed recipes don't race).
    result["speed"] = measure_speed(speed_runner, probe_keys)

    return result


def main():
    ap = argparse.ArgumentParser(description="confeTTI eval harness")
    ap.add_argument("--submission", required=True, help="path to submission model/repo")
    ap.add_argument("--reference", required=True, help="path to reference dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="result.json")
    args = ap.parse_args()

    result = run_eval(args.submission, args.reference, device=args.device)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
