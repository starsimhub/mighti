"""
Pytest configuration for MIGHTI.

Why this exists:
- `starsim` uses Numba with `cache=True` in several njit-decorated functions.
- In some environments (e.g. sandboxed runners, read-only site-packages),
  Numba cannot create cache files alongside the installed package, which can
  cause import-time errors during test collection.

We redirect Numba's cache into the repo (which is writable) before any test
module imports `starsim`.
"""

from __future__ import annotations

import os
from pathlib import Path


def pytest_configure(config):  # noqa: ARG001
    # Only set if not already defined by the user/CI.
    cache_dir = os.environ.get("NUMBA_CACHE_DIR")
    if not cache_dir:
        repo_root = Path(__file__).resolve().parents[1]
        cache_dir = str(repo_root / ".numba_cache")
        os.environ["NUMBA_CACHE_DIR"] = cache_dir

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Numba caching can fail on very new Python versions / some installations with
    # "no locator available" errors even when the cache directory is writable.
    # For tests we default to disabling caching to make import-time behavior robust.
    os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")

