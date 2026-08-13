# confeTTI 🎉

**Text-to-image runtime for a single RTX 5090.**

A submission is a model + runtime recipe that reproduces a fixed full-precision
reference's output while generating images as fast as possible on one RTX 5090
(32GB). The **fastest recipe that holds the quality floor leads** the board.

## What you're optimizing

End-to-end wall-clock seconds per generated image at a fixed resolution and
step budget, on one RTX 5090. The reference is Qwen-Image-2512 in BF16
(~57.7GB — it does not fit the card). The race is: take a frontier model that
can't run on a 32GB card, make it run, and make it run fast.

Any part of the stack is a legitimate lever:

- Kernels and attention (SageAttention-style, flash/memory-efficient attention)
- CUDA graphs, TensorRT, sfast, compilation passes
- Quantization to fit 32GB (INT4, NVFP4, GGUF, FP8) — a means to fit, not the goal
- Distillation / fewer-step samplers
- Memory and offload strategies
- Batching / throughput under concurrency

## The quality floor (the gate)

A recipe must reproduce the reference's **final image** within a perceptual
(LPIPS) distance tolerance on fixed (prompt, seed) probes. This is a
deterministic, frozen, model-free comparison — a recipe that trades quality
for speed (skipped steps, collapsed output) fails the gate regardless of how
fast it is.

The gate compares *final images*, not per-step predictions. This is what lets
the competition reward **distillation** alongside quantization: a distilled
4-step model legitimately predicts differently at each step (a per-step gate
rejects it by ~210x even though its output image is good), but reaches a
perceptually close final image. Output-fidelity is the only gate that accepts
both legitimate speed levers while still rejecting collapsed recipes.

The gate is tolerance-based (LPIPS threshold), calibrated on-box to sit between
good and bad recipes. LPIPS is a frozen, non-trainable network — not an LLM/VLM
judge and not trained on the eval corpus.

## Reference model (verified from source, 2026-08-13)

- Model: `Qwen/Qwen-Image-2512` (HuggingFace)
- License: `apache-2.0`, ungated
- Architecture: `QwenImageTransformer2DModel`, flow matching, 60 layers,
  24 heads x 128 (3072 hidden), in 64 / out 16 channels, patch 2
- BF16 footprint: transformer 40.86GB + text encoder 16.58GB + VAE 0.25GB
  = ~57.7GB. Full-precision reference is generated once on an A100-80GB;
  submissions run on the 5090.

## Eval

- **Gate:** LPIPS distance between the submission's final image and the
  reference's final image over the public (prompt, seed) corpus. Frozen,
  deterministic, no judge in the loop.
- **Race:** wall-clock seconds per image at fixed resolution/steps, measured
  on the same RTX 5090 box for every submission (same-box, no cross-box
  variance). Two-sided guard: implausibly-fast and underclaiming runs are
  both rejected.

## Status

Lean v0 core (engine + gate + reference tooling + tests). Governance stack
(REVIEW, EVAL-TRUST, config, CI) is added when the repo goes live.

## Repo layout (v0)

```
confeTTI/
  engine/           core harness: reference.py, gate.py, verify.py
  corpus/           public (prompt, seed, steps) corpus (immutable)
  tests/            test files
  kings/            published current-best recipe per lane
  submissions/      submitted recipes
  README.md, .gitignore, run.sh
```

## Getting started (developers)

```bash
git clone git@github.com:buckZz7/confeTTI.git
cd confeTTI
python3 -m venv .venv && source .venv/bin/activate
pip install -e .   # or: pip install diffusers torch accelerate safetensors
pytest -q          # run the harness tests
```

See `engine/` docs for the full eval spec.

## License

MIT (repo code). The reference model (Qwen-Image-2512) is Apache 2.0.
