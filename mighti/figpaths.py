"""
Helpers for reproducible output paths and CLI flags.

This is a lightweight utility originally used by the functionality repository.
It is kept in MIGHTI main for compatibility and to standardize where example
scripts write artifacts.

Conventions
-----------
- Default output directory is repo-root / "outputs"
- Override with CLI: --outdir SOME/PATH
- Or environment variable: MIGHTI_OUTDIR
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

__all__ = ["repo_root", "resolve_outdir", "add_repro_args"]


def repo_root() -> Path:
    """Return the repository root (parent of the `mighti/` package directory)."""
    return Path(__file__).resolve().parents[1]


def resolve_outdir(outdir: Optional[str | os.PathLike] = None, default: str = "outputs") -> Path:
    """
    Resolve the output directory for scripts.

    - If `outdir` is None, uses env var MIGHTI_OUTDIR if set, else `default`.
    - Relative paths are resolved relative to repo root (not current working dir).
    - Directory is created if missing.
    """
    if outdir is None or str(outdir).strip() == "":
        outdir = os.environ.get("MIGHTI_OUTDIR", default)

    outdir_path = Path(outdir).expanduser()
    if not outdir_path.is_absolute():
        outdir_path = repo_root() / outdir_path

    outdir_path.mkdir(parents=True, exist_ok=True)
    return outdir_path


def add_repro_args(parser, *, default_outdir: str = "outputs"):
    """
    Add standard reproduction args to an argparse parser.

    Adds:
      --outdir PATH        (default: env MIGHTI_OUTDIR or `default_outdir`)
      --show / --no-show   (default: no-show)
    """
    parser.add_argument(
        "--outdir",
        default=os.environ.get("MIGHTI_OUTDIR", default_outdir),
        help=f"Output directory for artifacts (default: {default_outdir} or env MIGHTI_OUTDIR).",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Display figures interactively (default: off).",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Do not display figures (recommended for reproducible runs).",
    )
    parser.set_defaults(show=False)
    return parser

