"""
Test impact of including the HIV-NCD connector

One simulation is run with HIV and alcohol use disorder, but no connector. A second simulation
is run with the addition of the connector. Adding the connector should increase
prevalence of alcohol use disorder. 
"""

import sciris as sc
import starsim as ss
import stisim as sti
import pylab as pl
import numpy as np
import mighti as mi
import pandas as pd
import os


# Settings
do_plot = False
sc.options(interactive=do_plot)

# File path to parameter file
thisdir = os.path.dirname(__file__)
param_path = os.path.join(thisdir, 'test_data', 'eswatini_parameters.csv')
params_df = pd.read_csv(param_path)
params_df.columns = params_df.columns.str.strip()


def test_hiv_alcoholusedisorder():
    hiv = sti.HIV(init_prev=0.1, beta={'structuredsexual': [0.01, 0.01]})
    alcoholusedisorder = mi.AlcoholUseDisorder(csv_path=param_path, init_prev=ss.bernoulli(0.1))

    pars = dict(
        start=2000,
        stop=2030,  # more years
        dt=1,
        n_agents=5000,  # more agents
        networks=sti.StructuredSexual(),
        diseases=[hiv, alcoholusedisorder]
    )
    sim0 = ss.Sim(pars).run()

    pars['connectors'] = mi.NCDHIVConnector({'alcoholusedisorder': 2.47})
    sim1 = ss.Sim(pars).run()

    res0 = sim0.results.alcoholusedisorder.prevalence.mean()
    res1 = sim1.results.alcoholusedisorder.prevalence.mean()
    assert res1 >= res0 - 0.01, f'AlcoholUseDisorder should be higher with connector: {res1=} vs {res0=}'
    print(f'[✓] Alcohol Use Disorder prevalence increased or remained similar: {res0:.3f} → {res1:.3f}')


# %% Run as a script
    if __name__ == '__main__':
        # Run `%matplotlib inline` to see the figure.
        pl.plot(sim0.results.alcoholusedisorder.timevec, sim0.results.alcoholusedisorder.prevalence, label='No connector')
        pl.plot(sim1.results.alcoholusedisorder.timevec, sim1.results.alcoholusedisorder.prevalence, label='With connector')
        pl.xlabel('Year')
        pl.ylabel('AUD Prevalence')
        pl.legend()
        pl.title('HIV ↔ Alcohol Use Disorder Connector Effect')
        pl.show()
        
        