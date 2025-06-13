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
import os

n_agents = 500

def test_basic_mighti_run():
    
    # Dummy mortality rates
    death_rates = {
        'death_rate': pd.DataFrame({
            'Time': [2000]*10,
            'AgeGrpStart': list(range(5)) * 2,
            'Sex': ['Male']*5 + ['Female']*5,
            'mx': [0.01]*10
        }),
        'rate_units': 1
    }

    death = ss.Deaths(death_rates)
    prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data={}, diseases=['AlcoholUseDisorder'])
    survivorship_analyzer = mi.SurvivorshipAnalyzer()
    deaths_analyzer = mi.DeathsByAgeSexAnalyzer()

    thisdir = os.path.abspath(os.path.dirname(__file__))
    prevalence_path = os.path.join(thisdir, '..', 'mighti', 'data', 'eswatini_parameters_gbd.csv')

    sim = ss.Sim(
        n_agents=n_agents,
        start=2000,
        stop=2002,
        demographics=[death],
        analyzers=[deaths_analyzer, survivorship_analyzer],
        diseases=[mi.AlcoholUseDisorder(csv_path=prevalence_path, pars={'init_prev': ss.bernoulli(0.1)})],
        label='test_basic_mighti_run'
    )

    sim.run()

    # Minimal assertion: check that time advanced
    assert sim.t.ti == len(sim.t.tvec) - 1
    assert hasattr(prevalence_analyzer, 'results')
    assert prevalence_analyzer.results is not None
    
# Run as a script (optional)
if __name__ == '__main__':    
    test_basic_mighti_run()
    