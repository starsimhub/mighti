"""
Test that NeighbourhoodSituation initializes and inherits correctly.

Runs a simulation and checks:
- proportion of people initialized with `neighbourhood_situation=True`
- that newborns inherit from their mothers
"""

import sciris as sc
import starsim as ss
import mighti as mi
import pandas as pd
import numpy as np
import os

# Settings
do_plot = False
sc.options(interactive=do_plot)

# File path to parameter file
thisdir = os.path.dirname(__file__)
param_path = os.path.join(thisdir, 'test_data', 'sdoh.csv')  # must contain state, state_prob, inherit_prob
age_path = os.path.join(thisdir, 'test_data', 'eswatini_age_distribution_2007.csv')  # simple flat age distribution
fertility_path = os.path.join(thisdir,'test_data', 'eswatini_asfr.csv')
def test_neighbourhood_situation_inheritance():
    
    fertility_rate = {'fertility_rate': pd.read_csv(fertility_path)}
    pregnancy = ss.Pregnancy(pars=fertility_rate)
    
    # SDoH states
    extra_sdoh_states = [
        ss.BoolArr('neighbourhood_situation')
    ]

    people = ss.People(1000, age_data=pd.read_csv(age_path), extra_states=extra_sdoh_states)

    maternal = ss.MaternalNet()    
    
    # Setup module
    ns = mi.NeighbourhoodSituation(csv_path=param_path)

    sim = ss.Sim(
        n_agents=10000,
        networks=maternal,
        start=2007,
        stop=2010,
        demographics=[pregnancy],
        people=people,
        connectors=ns,
        copy_inputs=False,
    )
    
    sim.run()

    ppl = sim.people
    n_true = ppl.neighbourhood_situation.sum()
    n_false = (~ppl.neighbourhood_situation).sum()

    print(f" NeighbourhoodSituation Init:\n  True:  {n_true:,}\n  False: {n_false:,}")
    assert 0 < n_true < len(ppl), "Expected a mix of True and False in initialized values."

    # Inheritance 
    maternal = sim.networks['maternalnet']
    edges = maternal.edges
    
    mothers_vals = edges.p1.to_numpy()
    newborns_vals = edges.p2.to_numpy()
    
    valid = (~np.isnan(mothers_vals)) & (~np.isnan(newborns_vals))
    
    m = mothers_vals[valid].astype(int)
    b = newborns_vals[valid].astype(int)
    
    sdoh = sim.people.neighbourhood_situation
    matches = sdoh.values[b] == sdoh.values[m]
    n_matches = matches.sum()
    n_total = len(matches)
    
    print(f"✅ Inherited housing: {n_matches} / {n_total} match")

if __name__ == '__main__':
    test_neighbourhood_situation_inheritance()
    
    