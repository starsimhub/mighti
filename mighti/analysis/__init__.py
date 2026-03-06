"""
MIGHTI analysis helpers.

This package collects post-simulation analysis utilities (life expectancy, plots, etc).

Design goals
------------
- Keep `import mighti` / `import mighti.analysis` lightweight.
- Prefer namespace usage:
    - `mi.analysis.life_expectancy.calculate_life_expectancy(...)`
    - `mi.analysis.plotting.plot_mean_prevalence(...)`
- Provide a small set of convenience re-exports (lazy-loaded) for commonly used
  life-expectancy helpers, without importing plotting/matplotlib eagerly.
"""

import importlib


# Allow direct access to submodules without eager imports
_SUBMODULES = {"life_expectancy", "plotting"}


# Convenience re-exports (lazy): name -> module
# NOTE: keep plotting functions out of this list to avoid importing matplotlib
# when users only want life expectancy utilities.
_EXPORTS = {
    # life_expectancy.py (pure numpy/pandas)
    "calculate_mortality_rates": "life_expectancy",
    "calculate_life_table_from_mx": "life_expectancy",
    "load_un_mx_from_wide": "life_expectancy",
    "load_un_ex_from_wide": "life_expectancy",
    "observed_e0_from_un_ex": "life_expectancy",
    "calculate_life_expectancy": "life_expectancy",
    "calculate_life_table": "life_expectancy",
    "life_table_from_mx": "life_expectancy",
    "reference_ex_from_mx_df": "life_expectancy",
    "make_ex_lookup": "life_expectancy",
    "calculate_life_expectancy_from_mx_df": "life_expectancy",
    "calculate_life_expectancy_from_age_sex_mx_analyzer": "life_expectancy",
}


__all__ = sorted([*_SUBMODULES, *_EXPORTS.keys()])


def __getattr__(name):
    """Lazy attribute resolver for submodules and convenience exports."""
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")

    modname = _EXPORTS.get(name)
    if modname is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = importlib.import_module(f"{__name__}.{modname}")
    return getattr(mod, name)


def __dir__():
    return sorted(list(globals().keys()) + __all__)

