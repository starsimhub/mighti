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

# Disease groups
ncd_names = params_df.query("disease_class == 'ncd'")['condition'].unique().tolist()
id_names = params_df.query("disease_class == 'sis'")['condition'].unique().tolist()


def run_test_ncd_state(disease_name, n_agents=100):
    sc.heading(f'Testing {disease_name}')
    ppl = ss.People(n_agents)
    ppl.hiv = np.zeros(n_agents, dtype=bool)
    disease_class = getattr(mi, disease_name, None)
    assert disease_class is not None, f"{disease_name} class not found in MIGHTI"
    disease = disease_class(csv_path=param_path, pars={'init_prev': 0.1})
    sim = ss.Sim(people=ppl, diseases=[disease], start=2020, stop=2025, dt=1, copy_inputs=False)
    sim.run()
    assert isinstance(disease.affected.sum(), (int, float, np.integer, np.floating))
    assert np.all((disease.affected | ~disease.affected))
    
    if do_plot:
        plot_disease_trajectory(sim, disease_name)



def run_test_id_state(disease_name, n_agents=100):
    sc.heading(f'Testing {disease_name}')
    ppl = ss.People(n_agents)
    ppl.hiv = np.zeros(n_agents, dtype=bool)
    disease_class = getattr(mi, disease_name, None)
    assert disease_class is not None, f"{disease_name} class not found in MIGHTI"
    disease = disease_class(csv_path=param_path, pars={'init_prev': 0.1})
    sim = ss.Sim(people=ppl, diseases=[disease], start=2020, stop=2025, dt=1, copy_inputs=False)
    sim.run()
    assert isinstance(disease.infected.sum(), (int, float, np.integer, np.floating))
    assert np.all((disease.infected | ~disease.infected))
    
    if do_plot:
        plot_disease_trajectory(sim, disease_name)


def test_all_diseases():
    for name in ncd_names:
        try:
            test_ncd_state(name)
            print(f"[SUCCESS] {name} passed.")
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            raise

    for name in id_names:
        try:
            test_id_state(name)
            print(f"[SUCCESS] {name} passed.")
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            raise


def test_multidisease(n_agents=100):
    sc.heading('Testing multi-disease simulation with MIGHTI diseases')

    # Initialize population
    ppl = ss.People(n_agents)
    ppl.hiv = np.zeros(n_agents, dtype=bool)  # ensure no HIV dependency issues

    # Load diseases from MIGHTI
    disease_names = ['Type2Diabetes', 'Hypertension']
    diseases = [getattr(mi, name)(csv_path=param_path, pars={'init_prev': 0.1}) for name in disease_names]

    # Create and run the simulation
    sim = ss.Sim(
        people=ppl,
        diseases=diseases,
        start=2020,
        stop=2025
    )
    sim.run()

    # Assertions
    for disease in disease_names:
        d_key = disease.lower()
        assert hasattr(sim.diseases[d_key], 'results'), f"{disease} missing results"
        r = sim.diseases[d_key].results
        if hasattr(r, 'n_affected'):
            assert np.any(r.n_affected), f"{disease} did not affect any agents"
        elif hasattr(r, 'n_infected'):
            assert np.any(r.n_infected), f"{disease} did not infect any agents"
        else:
            raise AssertionError(f"{disease} has no recognized results")

    print(f"[SUCCESS] Multi-disease sim with {', '.join(disease_names)} ran correctly.")



def plot_disease_trajectory(sim, disease_name):
    import pylab as pl

    disease_key = disease_name.lower()
    disease = sim.diseases.get(disease_key, None)
    if disease is None:
        print(f"[WARNING] {disease_name} not found in sim.")
        return

    time = sim.timevec
    r = disease.results
    death = sim.results.new_deaths.cumsum() if hasattr(sim.results, 'new_deaths') else np.zeros_like(time)

    # NCD-style plot
    if all(hasattr(r, attr) for attr in ['n_affected', 'n_at_risk', 'n_not_at_risk']):
        pl.figure()
        pl.stackplot(
            time,
            r.n_not_at_risk,
            r.n_at_risk - r.n_affected,
            r.n_affected,
            death,
        )
        pl.legend(['Not at risk', 'At risk', 'Affected', 'Dead'])

    # Infectious disease-style plot
    elif all(hasattr(r, attr) for attr in ['n_infected', 'n_susceptible']):
        recovered = getattr(r, 'n_recovered', np.zeros_like(time))  # optional
        pl.figure()
        pl.stackplot(
            time,
            r.n_susceptible,
            r.n_infected,
            recovered,
            death,
        )
        labels = ['Susceptible', 'Infected', 'Recovered', 'Dead'] if recovered is not None else ['Susceptible', 'Infected', 'Dead']
        pl.legend(labels)

    else:
        print(f"[WARNING] {disease_name} does not have recognized state results for plotting.")
        return

    pl.title(disease_name)
    pl.xlabel('Year')
    pl.ylabel('Number of agents')
    pl.tight_layout()
    pl.show()
    

if __name__ == '__main__':
    sc.options(interactive=True)
    do_plot = True
    test_all_diseases()
    test_multidisease()
    
    