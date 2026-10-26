"""
Test that annual probabilities give the same yearly risk whatever the timestep.

MIGHTI's parameter files give annual probabilities (of acquiring, remitting, and dying of a condition),
which Starsim converts to each timestep (`ss.probperyear`): at monthly steps, an annual probability p
is 1 - (1 - p)^(1/12) per step.
"""

import numpy as np
import pandas as pd
import starsim as ss

import mighti as mi


def test_annual_probability_conversion():
    """ An annual probability p is 1 - (1 - p)^(1/12) per month """
    p = 0.024
    assert np.isclose(ss.probperyear(p).to_prob(ss.months(1)), 1 - (1 - p)**(1/12))


def test_main_disease_auto_converts_annual_probabilities_for_monthly_dt(tmp_path):
    """ The share who acquire a condition in 5 years, with no remission or death, is 1 - (1 - p)^5 at yearly and monthly steps """
    param_path = tmp_path / "params.csv"
    p_f, p_m = 0.2, 0.1
    pd.DataFrame([dict(condition="Type2Diabetes", p_death=0.0, remission_rate=0.0, p_acquire_female=p_f, p_acquire_male=p_m, affected_sex="both")]).to_csv(param_path, index=False)

    for dt in [ss.years(1), ss.months(1)]:
        t2d = mi.diseases.Type2Diabetes(csv_path=str(param_path), pars={"init_prev": 0.0})
        sim = ss.Sim(n_agents=20_000, diseases=t2d, start=2020, stop=2025, dt=dt, use_aging=False)
        sim.run(verbose=0)
        dis = sim.diseases.type2diabetes
        female = sim.people.female
        for sex, p in [(female, p_f), (~female, p_m)]:
            share = dis.affected[sex.uids].mean()
            expected = 1 - (1 - p)**5
            assert abs(share - expected) < 0.015, f'At dt={dt}, {share=:.3f} acquired, but {expected=:.3f}'
