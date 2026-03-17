"""
Path helpers for stable data access.

Resolution order for processed data directory:
1) Explicit argument
2) Environment variable: MIGHTI_DATA_DIR
3) Repository default: <repo>/data/processed
"""

from pathlib import Path
import os

__all__ = ["get_repo_root", "get_data_dir", "get_processed_path"]


def get_repo_root():
    """Return repository root inferred from this module location."""
    return Path(__file__).resolve().parents[2]


def get_data_dir(data_dir=None):
    """Return processed data directory as a Path."""
    if data_dir:
        return Path(data_dir).expanduser().resolve()

    env_data_dir = os.environ.get("MIGHTI_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir).expanduser().resolve()

    return get_repo_root() / "data" / "processed"


def get_processed_path(filename, data_dir=None, *, must_exist=False):
    """Build a path under the processed data directory."""
    path = get_data_dir(data_dir) / filename
    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}\n"
            "Set MIGHTI_DATA_DIR or pass data_dir to point to processed inputs."
        )
    return path
