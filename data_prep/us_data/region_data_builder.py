"""
Region data "ensurer" used by example scripts.

The example entrypoints (`mighti_main.py`, `mighti_life_expectancy.py`) call
`ensure_region_data(...)` before running simulations. In this repo, the inputs
are expected to live under `mighti/data/` (e.g. `{region}_mx.csv`,
`{region}_age_distribution.csv`) and can be generated from raw sources using
scripts under `raw_data/`.

This module provides a small compatibility layer so those scripts don't crash
with `ModuleNotFoundError` in checkouts where the original `data_prep/` package
is absent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import prepare_data_for_year

logger = logging.getLogger(__name__)


def ensure_region_data(
    *,
    region: str,
    start_year: int,
    end_year: int,
    overwrite: bool = False,
) -> None:
    """
    Ensure the core `mighti/data/` inputs exist for a region.

    Currently this function is intentionally conservative:
    - It validates that base wide-format inputs exist:
      - `mighti/data/{region}_mx.csv`
      - `mighti/data/{region}_age_distribution.csv`
    - It generates (if missing) the derived inputs used by the example scripts:
      - `mighti/data/{region}_mortality_rates.csv` (from `{region}_mx.csv`)
      - `mighti/data/{region}_age_distribution_{start_year}.csv`

    Args:
        region: Region identifier (e.g. "eswatini", "nyc").
        start_year: First simulation year (used to create year-specific age distribution).
        end_year: Last simulation year (kept for API compatibility; not currently used).
        overwrite: If True, regenerate derived outputs even if they exist.
    """
    # Resolve repo root: .../data_prep/us_data/region_data_builder.py -> repo root is parents[2]
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "mighti" / "data"

    base_mx = data_dir / f"{region}_mx.csv"
    base_age = data_dir / f"{region}_age_distribution.csv"

    missing = [p.name for p in (base_mx, base_age) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required base inputs under `mighti/data/` for region "
            f"'{region}': {', '.join(missing)}. "
            "If you are starting from raw sources, generate these files using "
            "the utilities under `raw_data/` (see `raw_data/README.md`)."
        )

    # Derived inputs used by the example scripts
    mortality_rates = data_dir / f"{region}_mortality_rates.csv"
    if overwrite or not mortality_rates.exists():
        logger.info("Generating %s", mortality_rates.name)
        prepare_data_for_year.prepare_data(region)

    age_distribution_year = data_dir / f"{region}_age_distribution_{start_year}.csv"
    if overwrite or not age_distribution_year.exists():
        logger.info("Generating %s", age_distribution_year.name)
        prepare_data_for_year.prepare_data_for_year(region, start_year)

    # `end_year` intentionally unused for now (kept for backwards compatibility).
    _ = end_year

