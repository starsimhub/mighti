"""
Pytest configuration for MIGHTI.

Why this exists:
- `starsim` uses Numba with `cache=True` in several njit-decorated functions.
- In some environments (e.g. sandboxed runners, read-only site-packages),
  Numba cannot create cache files alongside the installed package, which can
  cause import-time errors during test collection.

Numba reads ``NUMBA_CACHE_DIR`` and locator settings when it is first imported.
That happens while pytest collects test modules, so we configure the
environment at **import time** (below), not only inside ``pytest_configure``.
"""

import os
from pathlib import Path


def _ensure_numba_cache_env() -> None:
    """Set Numba cache env before any test module imports starsim/numba."""
    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = os.environ.get("NUMBA_CACHE_DIR")
    if not cache_dir:
        cache_dir = str(repo_root / ".numba_cache")
        os.environ["NUMBA_CACHE_DIR"] = cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # Skip InTreeCacheLocator for code under site-packages (no writable locator).
    os.environ.setdefault(
        "NUMBA_CACHE_LOCATOR_CLASSES",
        "UserWideCacheLocator,IPythonCacheLocator",
    )


_ensure_numba_cache_env()


def pytest_configure(config):  # noqa: ARG001
    _ensure_numba_cache_env()
