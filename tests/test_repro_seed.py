"""
Reproducibility smoke test.

Goal: ensure a fixed rand_seed yields identical outcomes across repeated runs.
This guards against accidental use of NumPy's global RNG inside MIGHTI modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import starsim as ss

import mighti as mi


def _run_sdoh_once(seed: int) -> np.ndarray:
    thisdir = Path(__file__).resolve().parent
    data_dir = thisdir / "test_data"

    param_path = data_dir / "sdoh.csv"
    age_path = data_dir / "eswatini_age_distribution_2007.csv"
    fertility_path = data_dir / "eswatini_asfr.csv"

    fertility_rate = {"fertility_rate": pd.read_csv(fertility_path)}
    pregnancy = ss.Pregnancy(pars=fertility_rate)

    extra_states = [ss.BoolArr("neighbourhood_situation")]
    people = ss.People(n_agents=800, age_data=pd.read_csv(age_path), extra_states=extra_states)
    maternal = ss.MaternalNet()
    ns = mi.NeighbourhoodSituation(csv_path=str(param_path))

    sim = ss.Sim(
        rand_seed=int(seed),
        n_agents=800,
        start=2007,
        stop=2010,
        people=people,
        networks=maternal,
        demographics=[pregnancy],
        connectors=[ns],
        copy_inputs=False,
    )
    sim.run()
    return np.asarray(sim.people.neighbourhood_situation, dtype=bool).copy()


def test_rand_seed_is_deterministic_for_sdoh():
    a = _run_sdoh_once(123)
    b = _run_sdoh_once(123)
    c = _run_sdoh_once(124)

    assert np.array_equal(a, b), "Same seed should reproduce identical SDoH states"
    assert not np.array_equal(a, c), "Different seed should change outcomes"

