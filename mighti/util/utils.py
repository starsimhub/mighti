"""
Misc utilities.
"""

from pathlib import Path

import pandas as pd
import numpy as np

import starsim as ss

__all__ = [
    "make_p_death_fn",
    "make_dur_inf_fn",
    "dt_to_years",
    "annual_probability_to_timestep",
    "duration_years_to_timesteps",
    "convert_condition_parameters_to_dt",
    "birth_mother_baby_pairs",
]


def dt_to_years(dt):
    """Return a timestep expressed in years."""
    if hasattr(dt, "years"):
        return float(dt.years)
    return float(dt)


def annual_probability_to_timestep(probability, dt):
    """Convert an annual probability to the probability for timestep `dt`."""
    if pd.isna(probability):
        return probability

    dt_years = dt_to_years(dt)
    if dt_years <= 0:
        raise ValueError(f"dt must be positive; got {dt!r}")

    probability = float(probability)
    if not 0 <= probability <= 1:
        raise ValueError(f"Probability must be in [0, 1]; got {probability}")

    if probability in (0.0, 1.0) or dt_years == 1.0:
        return probability

    return 1.0 - (1.0 - probability) ** dt_years


def duration_years_to_timesteps(duration_years, dt):
    """Convert a duration in years into a number of timesteps."""
    if pd.isna(duration_years):
        return duration_years

    dt_years = dt_to_years(dt)
    if dt_years <= 0:
        raise ValueError(f"dt must be positive; got {dt!r}")

    return float(duration_years) / dt_years


def birth_mother_baby_pairs(sim):
    """
    Return aligned (mother_uids, baby_uids) for births tied to the current timestep.

    Starsim 3.x ``MaternalNet`` / ``PrenatalNet`` edges only store ``p1``, ``p2``,
    and ``beta`` (no ``start`` time). When a :class:`starsim.Pregnancy` module is
    present, mothers who deliver at ``sim.ti`` have ``ti_delivery == sim.ti``;
    their newborn(s) are taken as this mother's youngest child(ren) by age
    (handles twins).

    If no pregnancy module is found but maternal edges expose a legacy ``start``
    array, pairs are recovered the old way (``start == sim.ti``).
    """
    maternal = sim.networks.get("maternalnet", None) if hasattr(sim, "networks") else None
    edges = getattr(maternal, "edges", None) if maternal is not None else None

    pregnancy = None
    for mod in sim.demographics():
        if isinstance(mod, ss.Pregnancy):
            pregnancy = mod
            break

    if pregnancy is not None:
        mothers = (pregnancy.ti_delivery == sim.ti).uids
        mothers = np.asarray(mothers, dtype=int)
        if mothers.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        ppl = sim.people
        p1_out, p2_out = [], []
        for m in mothers:
            ch = ppl.find_children(ss.uids(np.array([m], dtype=int)))
            ch = np.asarray(ch, dtype=int)
            if ch.size == 0:
                continue
            ages = np.asarray(ppl.age[ch], dtype=float)
            min_age = ages.min()
            young = ch[ages == min_age]
            p1_out.append(np.full(young.shape, m, dtype=int))
            p2_out.append(young)
        if not p1_out:
            return np.array([], dtype=int), np.array([], dtype=int)
        return np.concatenate(p1_out), np.concatenate(p2_out)

    if edges is not None and hasattr(edges, "start") and hasattr(edges, "p1") and hasattr(edges, "p2"):
        try:
            birth_inds = np.where(np.asarray(edges.start) == sim.ti)[0]
        except Exception:
            return np.array([], dtype=int), np.array([], dtype=int)
        if birth_inds.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        return np.asarray(edges.p1, dtype=int)[birth_inds], np.asarray(edges.p2, dtype=int)[birth_inds]

    return np.array([], dtype=int), np.array([], dtype=int)


def convert_condition_parameters_to_dt(
    parameters,
    dt,
    *,
    probability_fields=("p_death", "remission_rate", "p_acquire", "p_acquire_male", "p_acquire_female"),
    duration_fields=(),
    copy=True,
):
    """Convert annual condition parameters to the specified timestep."""
    if isinstance(parameters, (str, Path)):
        data = pd.read_csv(parameters)
    elif isinstance(parameters, pd.DataFrame):
        data = parameters.copy(deep=True) if copy else parameters
    elif isinstance(parameters, dict):
        data = dict(parameters) if copy else parameters
    else:
        raise TypeError(
            "parameters must be a DataFrame, dict-like row, or CSV path; "
            f"got {type(parameters)!r}"
        )

    def _convert_mapping(mapping):
        for field in probability_fields:
            if field in mapping and not pd.isna(mapping[field]):
                mapping[field] = annual_probability_to_timestep(mapping[field], dt)
        for field in duration_fields:
            if field in mapping and not pd.isna(mapping[field]):
                mapping[field] = duration_years_to_timesteps(mapping[field], dt)
        return mapping

    if isinstance(data, pd.DataFrame):
        for field in probability_fields:
            if field in data.columns:
                numeric = pd.to_numeric(data[field], errors="coerce")
                converted = numeric.apply(
                    lambda value: annual_probability_to_timestep(value, dt) if not pd.isna(value) else value
                )
                data.loc[numeric.notna(), field] = converted.loc[numeric.notna()]
        for field in duration_fields:
            if field in data.columns:
                numeric = pd.to_numeric(data[field], errors="coerce")
                converted = numeric.apply(
                    lambda value: duration_years_to_timesteps(value, dt) if not pd.isna(value) else value
                )
                data.loc[numeric.notna(), field] = converted.loc[numeric.notna()]
        return data

    return _convert_mapping(data)


def make_p_death_fn(name=None, sim=None, uids=None):
    """Read in condition prognoses and return probability of instantaneous death."""

    ppl = sim.people  # Shorten
    death_prob = pd.Series(0.0, index=uids)  # placeholder for storing probabilities by age
    raw_progs = pd.read_csv("../mighti/data/condition_prognoses.csv")  # Read in the data
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
    raw_progs = pd.read_csv("../mighti/data/condition_prognoses.csv")  # Read in the data
    df = raw_progs.loc[raw_progs.condition == name]
    abins = np.append(df.age.unique(), 120)  # Add 120 as the upper age bin limit
    for sex in ["male", "female"]:
        for ai, lower_age in enumerate(abins[:-1]):
            upper_age = abins[ai + 1]
            meets_criteria = (ppl.age[uids] >= lower_age) & (ppl.age[uids] < upper_age) & (ppl[sex][uids])
            mean[uids[meets_criteria]] = df.dur_mean[(df.age == lower_age) & (df.sex == sex)].values[0]
            scale[uids[meets_criteria]] = df.dur_var[(df.age == lower_age) & (df.sex == sex)].values[0]
    return mean, scale

