"""
MIGHTI simulation test.

This test verifies that a basic MIGHTI simulation can:
- Initialize with MIGHTI modules (e.g., deaths, analyzers)
- Run successfully through all time steps
- Exit cleanly without raising exceptions

"""

import starsim as ss
import mighti as mi
import numpy as np
import pandas as pd

def test_basic_mighti_run():
    # Dummy mortality rates
    death_rates = {'death_rate': pd.DataFrame({
        'Year': [2000]*5,
        'Age': list(range(5)),
        'Sex': ['Male']*5,
        'Rate': [0.01]*5
    }), 'rate_units': 1}

    death = ss.Deaths(death_rates)
    prevalence_analyzer = mi.PrevalenceAnalyzer()
    survivorship_analyzer = mi.SurvivorshipAnalyzer()
    deaths_analyzer = mi.DeathsByAgeSexAnalyzer()


    sim = ss.Sim(
        n_agents=50,
        start=2000,
        stop=2002,
        demographics=[death],
        analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer],
        diseases=[mi.AlcoholUseDisorder()],
        label='test_basic_mighti_run'
    )

    sim.run()

    # Minimal assertion: check that time advanced
    assert sim.t.ti == len(sim.t.tvec) - 1
    assert hasattr(prevalence_analyzer, 'results')
    assert prevalence_analyzer.results is not None