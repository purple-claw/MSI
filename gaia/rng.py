"""Deterministic seed derivation and per-layer RNG streams.

Seeds come from blake2b, never Python's ``hash()`` (which is salted per process).
"""

import hashlib

import numpy as np

_MASK_63 = (1 << 63) - 1


def derive_seed(*parts: str | int) -> int:
    """Return a stable 63-bit seed derived from the given parts."""
    payload = "/".join(map(str, parts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MASK_63


def rng_for(home_seed: int, layer: str) -> np.random.Generator:
    """Return an independent generator for one layer of one home."""
    return np.random.default_rng(derive_seed(home_seed, layer))
