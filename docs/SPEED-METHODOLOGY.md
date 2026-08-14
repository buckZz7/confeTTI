# confeTTI — Speed Methodology

How wall-clock speed is measured *fairly* on the race box. This is the other
half of the eval: the gate (LPIPS output-fidelity) decides *whether* a recipe
qualifies; this spec decides *how fast* it is, in a way that is reproducible,
same-box, and resistant to gaming.

## Principles

1. **Same box, same conditions.** Every submission is timed on the *same*
   physical RTX 5090 eval node, under the same driver/container/deps. No
   cross-box comparison, ever.
2. **The number is what the box produced.** Wall-clock from the submission's
   own execution — not an estimate, not a claim, not a profiler extrapolation.
3. **Thermal and state drift are controlled, not ignored.** A 5090 under a
   long run throttles. The protocol must make timing robust to that.
4. **No hidden configuration.** The exact timing procedure is public and the
   eval runs it identically for every submission.

## The race metric

**Seconds per image** at a fixed image size and step budget, measured as the
median wall-clock over a timed window (below). Lower is better.

- Resolution: fixed (e.g. 1024x1024) — set by the eval config, not the submission.
- Step budget: fixed for the *race lane* (the submission may declare fewer
  steps only if it passes the gate at that step count — see Lanes).

### Lanes

The gate is output-fidelity (LPIPS vs the reference final image). Because the
gate compares *images*, a recipe is free to use any step count and any
schedule *as long as it passes the gate at that step count*. This is what
makes distillation a first-class lever (a 4-step model that reaches the same
image wins over a 28-step model that doesn't).

To keep the race apples-to-apples, define **lanes** by step budget:

- Each submission declares its step count at submission time.
- The eval runs the gate at that step count.
- **Within a lane**, recipes are ranked by seconds/image.
- A recipe in a *fewer-step* lane that passes the gate is not directly
  compared to a *more-step* lane; the leaderboard shows all lanes, and the
  overall leader is the fastest passing recipe across lanes.

(Open question — see Decision Log: should there be a single "fixed 28-step"
reference lane in addition to free-step lanes, so quantization-vs-distillation
can be compared head-to-head on identical step count?)

## Timing protocol

For each submission, after the gate passes:

1. **Load** the model to the box, run a **warmup** generation (discard).
   This absorbs compile/first-touch/weight-paging, so we time steady-state,
   not cold-start.
2. **Timed window:** run the submission over the eval corpus
   (N prompts x seeds) and record wall-clock per image.
3. **Multiple passes:** repeat the corpus R times (e.g. R=3), so timing is a
   distribution, not a single draw.
4. **Score = median** seconds/image across all timed runs. Median (not min,
   not mean) is robust to a couple of outlier runs (thermal dips, scheduler
   hiccups) without being gaming-able by a single fast draw.
5. **Report** median, min, max, and sample count for transparency.

### Warmup count

- Minimum 1 warmup run. If the submission uses a JIT/compile pass
  (torch.compile, TensorRT, CUDA graphs), the warmup must be enough for the
  compile to complete and the first compiled run to be warm.
- The eval detects "compiled" submissions (env flags / runtime deps) and
  raises the warmup count accordingly (e.g. 3 warmup runs). This prevents a
  submission from hiding its compile time in the timed window.

## Box drift / calibration guard

A single 5090 is not thermostable over hours. To keep timing comparable
across submissions evaluated at different times of day:

- **Calibration model:** a fixed, known-good reference recipe
  (e.g. the BF16 reference at its step count) is re-run at the start and end
  of every eval session, plus every 30 minutes during a long session.
- If the calibration model's median seconds/image drifts by more than a
  threshold (e.g. 10%) from its baseline, the box is considered drifted:
  the eval session is flagged, affected runs are re-run, or the session is
  paused until the box cools/recovers.
- A submission is only scored against the calibration baseline taken in the
  *same window* as its own run, never against a session-stale baseline.

This is the single most important fairness control — it stops "I submitted at
3am when the box was cold and idle" from being an unearned advantage.

## Two-sided speed guard

The number must be what the box actually produced. Reject:

- **Implausibly fast:** a seconds/image lower than the physical floor for the
  declared step count and architecture (below a known-good kernel lower
  bound). Catches "I claim I did 40 transformer steps in 0.3s" which is
  physically impossible on a 5090.
- **Underclaiming:** a submission that reports a *slower* number than it
  actually achieves (a miner sandbagging to under-promise and over-deliver,
  or to route around a future tie-break). The eval times what it times; a
  submission cannot choose its own reported number.

## What is NOT scored

- Startup / model load time (not part of steady-state generation speed).
- First-image cold latency (handled by warmup).
- Download time, install time, or compile time (all outside the timed window,
  but a slow-compiling submission is penalized by having to pay it outside
  the window — that's a real cost to the miner, not the score).

## Decision Log

| # | Question | Decision | Date |
|---|----------|----------|------|
| 1 | Single fixed-step lane vs free-step lanes | Free-step lanes (gate enforces quality) | 2026-08-13 |
| 2 | Score = median vs min | Median (robust to outliers) | 2026-08-13 |
| 3 | Calibration re-run interval | 30 min + session start/end | 2026-08-13 |
| 4 | Drift threshold | 10% median shift | 2026-08-13 |
| 5 | Whether to add a fixed 28-step "reference lane" | **OPEN** | — |

## Open Questions

- Q1 (lane #5): should there be a fixed-step reference lane in addition to
  free-step lanes, so quantization and distillation can be compared on
  identical step count? This may matter for how the leaderboard reads.
- Q2: batching — is the race single-image latency or images/sec under
  concurrency? (Currently single-image; batching as a lever implies a
  throughput lane.)
- Q3: does the eval box guarantee a minimum idle time before each timed run
  to normalize thermal state, or does the calibration guard cover it?
