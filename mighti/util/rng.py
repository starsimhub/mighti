"""
Deterministic random number helpers for MIGHTI.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

__all__ = ["get_rng", "seed_everything"]


def _mix_seed(base_seed: int, salt: str) -> int:
    """
    Mix a base integer seed with a salt string into a 32-bit seed.
    Uses SHA256 for stability across Python versions/platforms.
    """

    msg = f"{int(base_seed)}::{salt}".encode("utf-8")
    digest = hashlib.sha256(msg).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def get_rng(sim: Any | None, *, salt: str = "mighti") -> np.random.Generator:
    """
    Return a deterministic NumPy Generator for a given StarSim sim.

    - If the sim has ``pars.rand_seed``, the RNG will be deterministically seeded.
    - Each unique ``salt`` gets its own independent stream (cached per sim).
    - If no sim/seed is available, returns an unseeded default generator.
    """

    if sim is None:
        return np.random.default_rng()

    cache = getattr(sim, "_mighti_rngs", None)
    if cache is None:
        cache = {}
        setattr(sim, "_mighti_rngs", cache)

    if salt in cache:
        return cache[salt]

    seed = getattr(getattr(sim, "pars", None), "rand_seed", None)
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(_mix_seed(int(seed), str(salt)))

    cache[salt] = rng
    return rng


def seed_everything(seed: int) -> None:
    """
    Best-effort seeding of common RNGs for scripts.

    Prefer passing ``rand_seed=...`` into ``ss.Sim`` and using ``get_rng()`` in
    model code. This function is mainly for example driver scripts that still
    use libraries relying on NumPy's legacy global RNG.
    """

    import random

    random.seed(int(seed))
    np.random.seed(int(seed))

