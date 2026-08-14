# confeTTI — LPIPS Gate Calibration (2026-08-13)

Measured on an A100-80GB pod (RunPod `mo5yqg1f359s9q`), Qwen-Image-2512 BF16
reference, 4-step schedule, 3 prompts x 2 seeds = 6 probes.

## Results

| Recipe | Mean LPIPS vs ref | Gate (tol 0.85) |
|--------|-------------------|------------------|
| Reference (ref vs ref) | **0.0000** | pass (noise floor) |
| Distilled (Lightning 4-step LoRA) | **0.4873** | pass |
| Collapsed (heavily corrupted weights) | **1.1942** | fail |

## Interpretation

- **Noise floor = 0.0000**: the identical model vs itself is bit-exact after
  decode, so the gate baseline is clean.
- **Good distilled = 0.487**: a legitimate 4-step distilled model (which the
  per-step gate rejected by 210x) now passes cleanly — confirming the
  output-fidelity gate is the right choice. It's well below the 0.85 tolerance.
- **Collapsed = 1.194**: a broken recipe is clearly above tolerance.

## Tolerance decision

- Good cluster max: 0.487
- Bad cluster min: 1.194
- **Tolerance = 0.85** (midpoint of 0.487 and 1.194) — ~0.36 margin on both
  sides. This is comfortably between a good distilled recipe and a collapsed
  one.

## Caveats

1. The "collapsed" recipe was *random weight corruption*, a proxy for a truly
   broken/quality-collapsed model (e.g. a recipe that skips steps or produces
   garbage). Real broken recipes may behave somewhat differently, but the
   margin is large enough that this is low-risk.
2. Only one good-distilled model was measured (Lightning). A good *quantized*
   (FP8/INT4) recipe should score even lower than distilled (closer to 0.0),
   so 0.85 has additional headroom toward the good side.
3. LPIPS is resolution-normalized to 256x256 internally, so the tolerance is
   stable across image resolutions.
4. The tolerance should be re-validated on the race box (5090) at final eval
   config (full corpus, race resolution), and re-checked if the reference
   model or step budget changes.

## Files

- Test harness: `/opt/data/qwen-image-competition/lpips_calibration.py`
- Gate default: `engine/gate.py` `GateConfig.tolerance = 0.85`
