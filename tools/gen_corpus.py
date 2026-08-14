#!/usr/bin/env python3
"""confetti.tools.gen_corpus — generate the deterministic public corpus.

Produces corpus.json per docs/CORPUS-DESIGN.md: 7 prompt domains, a fixed
deterministic seed formula, and the step config. Running this with the same
seed formula always yields the same corpus, so it is reproducible by anyone.

Usage:
    python -m tools.gen_corpus [--out corpus.json] [--n-prompts 100]
                               [--seeds-per-prompt 4] [--steps 28]
"""
import argparse
import json
import random
import sys
import os

# Fixed primes for the deterministic seed formula (public, immutable).
PRIME_A = 2654435761  # golden-ratio hash constant
PRIME_B = 1597334677

# Domain balance (percent shares, from CORPUS-DESIGN.md). Order fixed.
DOMAINS = {
    "photoreal_natural": 25,
    "photoreal_human": 10,
    "text_in_image": 10,
    "stylized_illustrative": 20,
    "scene_composition": 15,
    "abstract_texture": 10,
    "challenging_rare": 10,
}

# A prompt seed-pool per domain. In a real launch these get hand-curated to be
# genuinely varied; the placeholder set below is illustrative and balanced.
DOMAIN_PROMPTS = {
    "photoreal_natural": [
        "A red fox in a snowy forest, golden hour light, photorealistic, 35mm",
        "Moss-covered cliffs above a crashing ocean at dusk, photoreal",
        "A macro shot of a dragonfly resting on a dewy leaf, sharp focus",
        "Rolling fog over a mountain valley at sunrise, aerial photoreal",
        "A wolf in a pine forest during a snowstorm, cinematic photoreal",
    ],
    "photoreal_human": [
        "A portrait of an elderly fisherman with weathered skin, studio light",
        "A candid photo of a street performer in a city plaza, mid-day",
        "An athlete mid-stride at a track meet, frozen motion, photoreal",
        "A child laughing while flying a kite in a park, golden light",
        "A professional headshot of a woman in a navy blazer, soft key light",
    ],
    "text_in_image": [
        "A vintage neon sign reading 'CAFE' on a rainy night street",
        "A poster with the text 'CONFETTI' in bold serif on a dark wall",
        "An open book with the title 'THE BEGINNING' on its cover, overhead",
        "A storefront awning that says 'BAKERY' in white on red, day",
        "A chalkboard menu board with the word 'SPECIALS' hand-lettered",
    ],
    "stylized_illustrative": [
        "A watercolor illustration of a whale breaching at sunset",
        "A flat vector illustration of a mountain town, bold colors",
        "A retro sci-fi book cover illustration of a space station, 1980s",
        "A children's book illustration of a friendly dragon in a meadow",
        "An anime-style illustration of a girl with a glowing umbrella at night",
    ],
    "scene_composition": [
        "A busy farmers market with stalls, people, and produce, wide shot",
        "A living room with a fireplace, reading chair, and bookshelves",
        "A city skyline at blue hour with a river and bridges",
        "An ornate temple courtyard with lanterns and koi pond, symmetrical",
        "A crowded train station platform at rush hour, dynamic composition",
    ],
    "abstract_texture": [
        "Close-up of swirling iridescent oil on water, abstract macro",
        "A fractal-like pattern of concentric glowing rings, dark background",
        "Layered translucent silk in motion, soft gradients, abstract",
        "Geometric concrete architecture shot from below, strong shadows",
        "Rippling sand dunes with sharp ridges at noon, minimal texture",
    ],
    "challenging_rare": [
        "A platypus wearing a tiny top hat on a surfboard, dramatic light",
        "An impossible Escher-like staircase in a desert landscape",
        "A glass sculpture of a jellyfish in a dark gallery, spotlit",
        "A robotic raccoon repairing a vintage typewriter, cinematic",
        "An aurora over a frozen lake with a lone cabin, ultra-wide",
    ],
}


def seed_for(prompt_index: int, seed_index: int) -> int:
    """Deterministic seed from indices. Reproducible, unbiased, covers well."""
    return (PRIME_A * prompt_index + PRIME_B * seed_index) % (2**32)


def build_corpus(n_prompts: int, seeds_per_prompt: int, steps: int):
    # Allocate prompt counts per domain proportionally to the shares.
    total_share = sum(DOMAINS.values())
    prompts = []
    idx = 0
    for domain, share in DOMAINS.items():
        count = max(1, round(n_prompts * share / total_share))
        pool = DOMAIN_PROMPTS[domain]
        for k in range(count):
            text = pool[idx % len(pool)]  # deterministic rotation through pool
            prompts.append({"id": idx, "domain": domain, "text": text})
            idx += 1
    # Trim/pad to exactly n_prompts (deterministic).
    prompts = prompts[:n_prompts]
    while len(prompts) < n_prompts:
        d = "photoreal_natural"
        prompts.append({"id": len(prompts), "domain": d, "text": DOMAIN_PROMPTS[d][0]})

    return {
        "version": 1,
        "seed_formula": {"prime_a": PRIME_A, "prime_b": PRIME_B},
        "steps": steps,
        "seeds_per_prompt": seeds_per_prompt,
        "domains": DOMAINS,
        "prompts": prompts,
        # Precompute the full probe list (seed:prompt_id) deterministically.
        "probes": [
            f"{seed_for(p['id'], s)}:{p['id']}"
            for p in prompts
            for s in range(seeds_per_prompt)
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="Generate the confeTTI corpus")
    ap.add_argument("--out", default="corpus.json")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--seeds-per-prompt", type=int, default=4)
    ap.add_argument("--steps", type=int, default=28)
    args = ap.parse_args()

    corpus = build_corpus(args.n_prompts, args.seeds_per_prompt, args.steps)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(corpus, f, indent=2)
    # Print a short summary.
    print(json.dumps({
        "version": corpus["version"],
        "prompts": len(corpus["prompts"]),
        "seeds_per_prompt": corpus["seeds_per_prompt"],
        "total_probes": len(corpus["probes"]),
        "steps": corpus["steps"],
        "domains": {k: sum(1 for p in corpus["prompts"] if p["domain"] == k)
                    for k in DOMAINS},
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
