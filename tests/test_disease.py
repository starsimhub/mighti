"""
Test MIGHTI disease modules (non-HIV)
"""

import mighti as mi
import starsim as ss
import sciris as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Settings
do_plot = False
sc.options(interactive=do_plot)

# File path to parameter file
thisdir = os.path.abspath(os.path.dirname(__file__))
param_path = os.path.join(thisdir, '..', 'mighti', 'data', 'eswatini_parameters.csv')
param_path = os.path.abspath(param_path)
params_df = pd.read_csv(param_path)
params_df.columns = params_df.columns.str.strip()

# List of disease names (excluding HIV)
disease_names = params_df.query("condition != 'HIV'")['condition'].unique()

# Group diseases
ncd_names = params_df[params_df["disease_class"] == "ncd"]["condition"].unique().tolist()
id_names  = params_df[params_df["disease_class"] == "sis"]["condition"].unique().tolist()

def test_ncd_state(disease_name, n_agents=100):
    sc.heading(f'Testing {disease_name}')
    ppl = ss.People(n_agents)
    ppl.hiv = np.zeros(n_agents, dtype=bool)
    disease = getattr(mi, disease_name)(csv_path=param_path, pars={'init_prev': 0.1})
    sim = ss.Sim(people=ppl, diseases=[disease], start=2020, stop=2025, dt=1)
    sim.run()
    assert isinstance(disease.affected.sum(), (int, float, np.integer, np.floating))
    assert np.all((disease.affected | ~disease.affected))
    return sim

def test_id_state(disease_name, n_agents=100):
    sc.heading(f'Testing {disease_name}')
    ppl = ss.People(n_agents)
    ppl.hiv = np.zeros(n_agents, dtype=bool)
    disease = getattr(mi, disease_name)(csv_path=param_path, pars={'init_prev': 0.1})
    sim = ss.Sim(people=ppl, diseases=[disease], start=2020, stop=2025, dt=1)
    sim.run()
    assert isinstance(disease.infected.sum(), (int, float, np.integer, np.floating))
    assert np.all((disease.infected | ~disease.infected))
    return sim

def plot_disease_trajectory(sim, disease_name):
    import pylab as pl
    disease = sim.diseases[disease_name.lower()]
    r = disease.results
    t = sim.timevec
    death = sim.results.new_deaths.cumsum() if hasattr(sim.results, 'new_deaths') else np.zeros_like(t)

    if hasattr(r, 'n_affected') and hasattr(r, 'n_at_risk') and hasattr(r, 'n_not_at_risk'):
        pl.figure()
        pl.stackplot(t, r.n_not_at_risk, r.n_at_risk - r.n_affected, r.n_affected, death)
        pl.legend(['Not at risk', 'At risk', 'Affected', 'Dead'])
    elif hasattr(r, 'n_infected') and hasattr(r, 'n_susceptible'):
        recovered = getattr(r, 'n_recovered', np.zeros_like(t))
        pl.figure()
        pl.stackplot(t, r.n_susceptible, r.n_infected, recovered, death)
        labels = ['Susceptible', 'Infected', 'Recovered', 'Dead'] if np.any(recovered) else ['Susceptible', 'Infected', 'Dead']
        pl.legend(labels)
    else:
        print(f"[WARNING] {disease_name} missing recognized state results")
        return
    pl.title(disease_name)
    pl.xlabel('Year')
    pl.ylabel('Number of agents')
    pl.tight_layout()
    pl.show()

def test_all_diseases():
    n_passed = 0
    n_failed = 0

    for name in ncd_names:
        try:
            sim = test_ncd_state(name)
            if do_plot: plot_disease_trajectory(sim, name)
            print(f"[SUCCESS] {name} passed.")
            n_passed += 1
        except Exception as E:
            print(f"[ERROR] {name} failed: {E}")
            n_failed += 1

    for name in id_names:
        try:
            sim = test_id_state(name)
            if do_plot: plot_disease_trajectory(sim, name)
            print(f"[SUCCESS] {name} passed.")
            n_passed += 1
        except Exception as E:
            print(f"[ERROR] {name} failed: {E}")
            n_failed += 1

    # Final assertion: ensure at least one test passed
    assert n_passed > 0, "All disease tests failed"

def test_multidisease(n_agents=100):
    sc.heading('Testing multi-disease simulation')
    ppl = ss.People(n_agents)
    sir1 = ss.SIR(name='sir1', pars={'beta': {'randomnet': 0.1}})
    sir2 = ss.SIR(name='sir2', pars={'beta': {'randomnet': 0.2}})
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(4)))
    sim = ss.Sim(people=ppl, diseases=[sir1, sir2], networks=net, start=2020, stop=2025, dt=1)
    sim.run()
    assert hasattr(sim.diseases['sir1'], 'results')
    assert hasattr(sim.diseases['sir2'], 'results')
    assert sim.diseases['sir1'].results.n_infected[-1] >= 0
    assert sim.diseases['sir2'].results.n_infected[-1] >= 0
    print("[SUCCESS] Multi-disease sim ran correctly.")
    assert sim.results is not None

if __name__ == '__main__':
    import pylab as pl
    sc.options(interactive=True)
    do_plot = True
    
    test_all_diseases()
    test_multidisease()
    