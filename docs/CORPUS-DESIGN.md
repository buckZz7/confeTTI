# confeTTI — Corpus Design

The public (prompt, seed, steps) set that anchors the reference and the gate.
Everything about it is public and deterministic; the corpus is generated once
and immutable.

## Purpose

The corpus serves three roles:

1. **Reference anchoring** — the BF16 reference's final images are computed
   over this corpus; the hash-bound reference is what submissions are gated
   against.
2. **Gate coverage** — the gate must exercise a spread of prompt types so a
   recipe can't pass by being good at only one domain.
3. **Cost control** — each probe costs ~2-3s (denoise + VAE decode + LPIPS),
   so the corpus size must be calibrated against eval budget per submission.

## Size and budget

- Target **N prompts**, each paired with **S seeds** (e.g. N=100, S=4 →
  400 probes) — placeholder; calibrated against eval cost and LPIPS
  separation confidence.
- Per-submission eval cost ≈ N×S × (gen time + LPIPS time).
  At ~2.5s/gen + ~0.1s/LPIPS on an A100 for reference, or a quant/distill
  recipe on the 5090, 400 probes ≈ 15-20 min per submission. Tuned so the
  gate is strong (large N) but eval stays cheap enough for frequent
  submissions.

## Domains (prompt categories)

The corpus spans categories that stress different parts of a T2I model:

| Domain | Count share | Stress |
|--------|------------|--------|
| Photoreal (natural) | ~25% | realism, lighting |
| Photoreal (human) | ~10% | anatomy, faces |
| Text-in-image | ~10% | rendering legible text (a known failure mode) |
| Stylized / illustrative | ~20% | style adherence |
| Scene / composition | ~15% | multi-object layout |
| Abstract / texture | ~10% | no real-world referent |
| Challenging / rare concepts | ~10% | long-tail, adversarial-ish |

Domains are explicitly balanced so a recipe that overfits one category (e.g.
only looks good at photorealism) can't carry the gate.

## Seed derivation (deterministic)

Seeds are NOT human-picked (no cherry-picking bias). Derive them
deterministically from the corpus index:

```
seed(prompt_index, seed_index) = (PRIME_A * prompt_index + PRIME_B * seed_index) mod 2**32
```

with fixed primes. The exact formula is public and fixed at corpus creation.
This guarantees:
- Reproducible: anyone recomputes the same seeds.
- Unbiased: seeds are a deterministic function of position, not chosen.
- Coverage: different prompts get different seeds even at the same
  seed_index.

## Steps

- Fixed step budget per lane (see SPEED-METHODOLOGY.md). The corpus stores
  the reference at the reference lane's step count.
- A submission in a fewer-step lane is gated at ITS step count against the
  same reference images (the gate compares final images, which is
  step-count-agnostic).

## Immutability

- The corpus (prompts + seed formula + step config) is frozen at reference
  creation.
- A hash of the corpus manifest is stored in the reference manifest.
- The corpus is published with the reference so anyone can reproduce it.

## Generation of the corpus file

A small generator (`tools/gen_corpus.py`) produces a deterministic
`corpus.json`:

```json
{
  "version": 1,
  "seed_formula": {"prime_a": 2654435761, "prime_b": 1597334677},
  "steps": 28,
  "domains": {"photoreal_natural": 25, ...},
  "prompts": [
    {"id": 0, "domain": "photoreal_natural",
     "text": "A red fox in a snowy forest, golden hour light..."}
  ],
  "seeds_per_prompt": 4
}
```

## Decision Log

| # | Question | Decision | Date |
|---|----------|----------|------|
| 1 | Corpus size | N=100 prompts x S=4 seeds (placeholder, calibrate vs eval cost) | 2026-08-13 |
| 2 | Seed derivation | Deterministic linear formula (not human-picked) | 2026-08-13 |
| 3 | Domain balance | 7 categories, text-in-image explicitly included | 2026-08-13 |

## Open Questions

- Q1: Is a single reference step count (e.g. 28) sufficient, or should the
  reference also be generated at a "fast" step count (e.g. 8) to anchor a
  distillation lane?
- Q2: Should human-realism prompts be capped to avoid an implicit bias
  toward models strong on faces?
- Q3: Is LPIPS separation confidence adequate at 400 probes, or do we need
  more for a stable gate tolerance?
