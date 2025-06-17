"""
Test MIGHTI mortality rate and life expectancy
"""

import starsim as ss
import sciris as sc
import pandas as pd
import numpy as np
import mighti as mi
import os

# Settings
n_agents = 500
do_plot = False
sc.options(interactive=do_plot)

def get_deaths_module(sim):
    for module in sim.modules:
        if isinstance(module, mi.DeathsByAgeSexAnalyzer):
            return module
    raise ValueError("Deaths module not found in the simulation. Make sure you've added the DeathsByAgeSexAnalyzer to your simulation configuration")

def get_pregnancy_module(sim):
    for module in sim.modules:
        if isinstance(module, ss.Pregnancy):
            return module
    raise ValueError("Pregnancy module not found in the simulation.")


def test_life_expectancy():
    
    # File paths
    thisdir = os.path.dirname(__file__)
    csv_path_death = os.path.join(thisdir, 'test_data', 'eswatini_mortality_rates_2007.csv')
    csv_path_fertility = os.path.join(thisdir, 'test_data', 'eswatini_asfr.csv')

    # Prepare input
    death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
    death = ss.Deaths(death_rates)
    fertility_rate = {'fertility_rate': pd.read_csv(csv_path_fertility)}
    pregnancy = ss.Pregnancy(pars=fertility_rate)

    # Initialize the PrevalenceAnalyzer
    survivorship_analyzer = mi.SurvivorshipAnalyzer()
    deaths_analyzer = mi.DeathsByAgeSexAnalyzer()
    
    sim = ss.Sim(
        n_agents=n_agents,
        start=2007,
        stop=2008,
        demographics=[pregnancy, death],
        analyzers=[deaths_analyzer, survivorship_analyzer],
    )
    sim.run()

    deaths_module = get_deaths_module(sim)
    
    
    # Calculate mortality rates using `calculate_mortality_rates
    df_mx = mi.calculate_mortality_rates(sim, deaths_module, year=2008, max_age=100, radix=n_agents)

    # Basic checks
    assert not df_mx.empty
    assert 'mx' in df_mx.columns
    assert (df_mx['mx'] >= 0).all()

    df_mx_male = df_mx[df_mx['sex'] == 'Male']
    df_mx_female = df_mx[df_mx['sex'] == 'Female']

    life_table = mi.create_life_table(df_mx_male, df_mx_female, max_age=100)

    assert set(['Age', 'l(x)', 'd(x)', 'q(x)', 'm(x)', 'L(x)', 'T(x)', 'e(x)', 'sex']).issubset(life_table.columns)    
    assert len(life_table) == 2 * (100 + 1)  # 0–100 for Male and Female
    assert set(life_table['sex'].unique()) == {'Male', 'Female'}
    for sex in ['Male', 'Female']:
        lx = life_table.loc[life_table['sex'] == sex, 'l(x)'].values
    assert np.all(np.diff(lx) <= 0), f"l(x) not non-increasing for {sex}"
    assert life_table['e(x)'].min() >= 0
    
    
if __name__ == '__main__':
    test_life_expectancy()
    print("[✓] Mortality rate and life expectancy tests passed.")   
    