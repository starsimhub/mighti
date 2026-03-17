"""
Misc utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from .paths import get_processed_path

__all__ = ["make_p_death_fn", "make_dur_inf_fn"]


def _load_condition_prognoses():
    """Load condition prognoses from processed data, with legacy fallback."""
    candidates = [
        get_processed_path("condition_prognoses.csv"),
        Path(__file__).resolve().parents[1] / "data" / "condition_prognoses.csv",  # legacy
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        "condition_prognoses.csv not found. "
        "Set MIGHTI_DATA_DIR to your processed data directory or provide the legacy file."
    )


def make_p_death_fn(name=None, sim=None, uids=None):
    """Read in condition prognoses and return probability of instantaneous death."""

    ppl = sim.people  # Shorten
    death_prob = pd.Series(0.0, index=uids)  # placeholder for storing probabilities by age
    raw_progs = _load_condition_prognoses()
    df = raw_progs.loc[raw_progs.condition == name]
    abins = np.append(df.age.unique(), 120)  # Add 120 as the upper age bin limit
    for sex in ["male", "female"]:
        for ai, lower_age in enumerate(abins[:-1]):
            upper_age = abins[ai + 1]
            meets_criteria = (ppl.age[uids] >= lower_age) & (ppl.age[uids] < upper_age) & (ppl[sex][uids])
            death_prob[uids[meets_criteria]] = df.p_instdeath[(df.age == lower_age) & (df.sex == sex)].values[0]
    return death_prob


def make_dur_inf_fn(name=None, sim=None, uids=None):
    """Read in condition prognoses and return mean/scale for infection duration."""

    ppl = sim.people  # Shorten
    mean = pd.Series(0.0, index=uids)  # placeholder for storing mean durations
    scale = pd.Series(0.0, index=uids)  # placeholder for storing scale
    raw_progs = _load_condition_prognoses()
    df = raw_progs.loc[raw_progs.condition == name]
    abins = np.append(df.age.unique(), 120)  # Add 120 as the upper age bin limit
    for sex in ["male", "female"]:
        for ai, lower_age in enumerate(abins[:-1]):
            upper_age = abins[ai + 1]
            meets_criteria = (ppl.age[uids] >= lower_age) & (ppl.age[uids] < upper_age) & (ppl[sex][uids])
            mean[uids[meets_criteria]] = df.dur_mean[(df.age == lower_age) & (df.sex == sex)].values[0]
            scale[uids[meets_criteria]] = df.dur_var[(df.age == lower_age) & (df.sex == sex)].values[0]
    return mean, scale

