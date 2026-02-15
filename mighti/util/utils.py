"""
Misc utilities.
"""

import pandas as pd
import numpy as np

__all__ = ["make_p_death_fn", "make_dur_inf_fn"]


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

