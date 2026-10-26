"""
Tests for reference life expectancy utilities and YLL analyzers.
"""

import numpy as np
import pandas as pd
import starsim as ss

import mighti as mi

def test_reference_ex_from_mx_and_yll_analyzers():
    max_age = 100

    # Simple synthetic period mortality: constant hazard by age/sex
    ages = np.arange(max_age + 1)
    df_mx = pd.concat(
        [
            pd.DataFrame({"age": ages, "sex": "Male", "mx": 0.02}),
            pd.DataFrame({"age": ages, "sex": "Female", "mx": 0.02}),
        ],
        ignore_index=True,
    )

    df_ex = mi.life_expectancy.reference_ex_from_mx_df(df_mx, max_age=max_age, radix=100_000)
    assert set(df_ex.columns) >= {"age", "sex", "ex"}
    assert len(df_ex) == 2 * (max_age + 1)

    ex_lookup = mi.life_expectancy.make_ex_lookup(df_ex)
    assert ex_lookup("Male", 0) > ex_lookup("Male", 50) >= 0

    # Run a tiny sim just to ensure analyzers accept a DataFrame reference ex table
    death = ss.Deaths({"death_rate": 0.5, "rate_units": 1})  # high death prob for test stability

    cond_an = mi.analyzers.ConditionAtDeathAnalyzer(conditions=[], ex_life_expectancy=df_ex)
    cod_an = mi.analyzers.CauseOfDeathYLLAnalyzer(df_ex)

    sim = ss.Sim(
        n_agents=200,
        start=2007,
        stop=2008,
        demographics=[death],
        analyzers=[cond_an, cod_an],
    )
    sim.run()

    cond_an = sim.analyzers.condition_at_death_analyzer
    cod_an = sim.analyzers.cause_of_death_yll_analyzer
    df_cond = cond_an.to_df()
    if len(df_cond):
        assert (df_cond["yll"] >= 0).all()

    df_cod = cod_an.to_df()
    if len(df_cod):
        assert (df_cod["yll"] >= 0).all()
        # The only cause of death is the background deaths of ss.Deaths
        assert set(df_cod["cause"].unique()) == {"deaths"}

