"""
Test automatic annual-to-timestep parameter conversion in upstream MIGHTI diseases.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import starsim as ss

import mighti as mi
from mighti.util.utils import annual_probability_to_timestep


def test_main_disease_auto_converts_annual_probabilities_for_monthly_dt(tmp_path):
    param_path = tmp_path / "params.csv"
    pd.DataFrame(
        [
            {
                "condition": "Type2Diabetes",
                "p_death": 0.025,
                "dur_condition": 12.24,
                "rel_sus": 1.61,
                "remission_rate": 0.024,
                "max_disease_duration": 24.48,
                "p_acquire": 0.002681646,
                "p_acquire_male": 0.0030,
                "p_acquire_female": 0.0022,
                "affected_sex": "both",
            }
        ]
    ).to_csv(param_path, index=False)

    ppl = ss.People(10)
    ppl.hiv = np.zeros(10, dtype=bool)
    disease = mi.Type2Diabetes(csv_path=str(param_path), pars={"init_prev": ss.bernoulli(0.0)})

    sim = ss.Sim(
        people=ppl,
        diseases=[disease],
        start=2020,
        stop=2021,
        dt=1 / 12,
        copy_inputs=False,
    )
    sim.init()

    assert math.isclose(
        disease.pars.p_acquire,
        annual_probability_to_timestep(0.002681646, 1 / 12),
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        disease.pars.p_acquire_male,
        annual_probability_to_timestep(0.0030, 1 / 12),
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        disease.pars.p_acquire_female,
        annual_probability_to_timestep(0.0022, 1 / 12),
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        disease.pars.remission_rate,
        annual_probability_to_timestep(0.024, 1 / 12),
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        disease.pars.p_death.pars["p"],
        annual_probability_to_timestep(0.025, 1 / 12),
        rel_tol=0,
        abs_tol=1e-12,
    )
