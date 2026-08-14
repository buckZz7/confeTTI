#!/usr/bin/env python3
"""Tests for the corpus generator (deterministic, no deps)."""
import json
from tools.gen_corpus import build_corpus, seed_for, PRIME_A, PRIME_B, DOMAINS


def test_corpus_deterministic():
    a = build_corpus(100, 4, 28)
    b = build_corpus(100, 4, 28)
    assert a == b  # same inputs -> identical corpus (reproducible)


def test_probe_count():
    c = build_corpus(100, 4, 28)
    assert len(c["probes"]) == 100 * 4


def test_domain_balance_present():
    c = build_corpus(100, 4, 28)
    domains = {p["domain"] for p in c["prompts"]}
    assert set(DOMAINS.keys()) <= domains  # every domain represented


def test_seed_deterministic():
    assert seed_for(0, 0) == seed_for(0, 0)
    # Different prompt indices give different seeds.
    assert seed_for(0, 0) != seed_for(1, 0)
    # Within 32-bit range.
    assert 0 <= seed_for(100, 4) < 2**32


def test_seed_covers_variety():
    # Across prompt/seed indices, seeds should be well-distributed.
    seeds = {seed_for(i, s) for i in range(50) for s in range(4)}
    assert len(seeds) > 150  # mostly unique
