"""
Test analyzers for MIGHTI demographic modules
"""

import mighti as mi
import starsim as ss
import sciris as sc
import pandas as pd
import numpy as np
import os

# Settings
n_agents = 500
do_plot = False
sc.options(interactive=do_plot)

# File paths
thisdir = os.path.abspath(os.path.dirname(__file__))
csv_path_death     = os.path.join(thisdir, '..', 'mighti', 'data', 'eswatini_mortality_rates_2007.csv')
csv_path_fertility = os.path.join(thisdir, '..', 'mighti', 'data', 'eswatini_asfr.csv')
csv_path_age       = os.path.join(thisdir, '..', 'mighti', 'data', 'eswatini_age_distribution_2007.csv')


def get_deaths_module(sim):
    for module in sim.modules:  # This includes analyzers
        if isinstance(module, mi.DeathsByAgeSexAnalyzer):
            return module
    raise ValueError("Deaths module not found in the simulation.")

def test_deaths_analyzer_counts():
    # Prepare input
    death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
    death = ss.Deaths(death_rates)
    fertility_rate = {'fertility_rate': pd.read_csv(csv_path_fertility)}
    pregnancy = ss.Pregnancy(pars=fertility_rate)

    # Create the analyzer instance
    deaths_analyzer = mi.DeathsByAgeSexAnalyzer()

    # Run simulation with analyzer
    sim = ss.Sim(
        n_agents=n_agents,
        start=2007,
        stop=2010,
        demographics=[pregnancy, death],
        analyzers=[deaths_analyzer]
    )
    sim.run()

    # Retrieve via module-like accessor
    deaths_module = get_deaths_module(sim)

    # Use it
    df = deaths_module.to_df()
    assert 'year' in df.columns
    assert 'age' in df.columns
    assert 'sex' in df.columns
    assert df['deaths'].sum() > 0
    
    
# def test_survivorship_output():
#     analyzer = mi.SurvivorshipAnalyzer()

#     sim = ss.Sim(n_agents=n_agents,
#                  start=2007,
#                  stop=2010)

#     sim.modules = [analyzer]

#     sim.run()

#     df = analyzer.to_df()
#     assert 'survivorship' in df.columns
#     assert df['survivorship'].notna().all()
    
    
if __name__ == '__main__':
    test_deaths_analyzer_counts()
    # test_survivorship_output()
    print("[✓] All demographic analyzer tests passed.")

    