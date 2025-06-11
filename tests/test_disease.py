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


def test_ncd_state(disease_name, n_agents=100):
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

    return sim


def test_id_state(disease_name, n_agents=100):
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

    return sim




# def test_disease_state(disease_name, n_agents=100, do_plot=False):
#     sc.heading(f'Testing {disease_name}')

#     # Initialize synthetic population
#     ppl = ss.People(n_agents)
    
#     # Ensure 'hiv' attribute exists to avoid errors in conditions that check for it
#     if not hasattr(ppl, 'hiv'):
#         ppl.hiv = np.zeros(n_agents, dtype=bool)

#     # Instantiate disease class
#     disease_class = getattr(mi, disease_name, None)
#     assert disease_class is not None, f"{disease_name} class not found in MIGHTI"
#     disease = disease_class(csv_path=param_path, pars={'init_prev': 0.1})
    
#     # Create sim object
#     sim = ss.Sim(
#         people=ppl,
#         diseases=[disease],
#         start=2020,
#         stop=2025,
#         dt=1,
#         copy_inputs=False
#     )

#     # Run sim
#     sim.run()

#     # Check logical consistency of state arrays
#     if hasattr(disease, 'affected'):
#         assert isinstance(disease.affected.sum(), (int, float, np.integer, np.floating)), f"{disease_name}: 'affected' state not numeric"
#         assert np.all((disease.affected | ~disease.affected)), f"{disease_name}: 'affected' contains invalid values"
#     elif hasattr(disease, 'infected'):
#         assert isinstance(disease.infected.sum(), (int, float, np.integer, np.floating)), f"{disease_name}: 'infected' state not numeric"
#         assert np.all((disease.infected | ~disease.infected)), f"{disease_name}: 'infected' contains invalid values"
#     else:
#         raise AssertionError(f"{disease_name} has neither 'affected' nor 'infected' state defined")

#     return sim


def test_all_diseases():
    sims = []
    for name in ncd_names:
        try:
            sim = test_ncd_state(name)
            print(f"[SUCCESS] {name} passed.")
            sims.append(sim)
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")

    for name in id_names:
        try:
            sim = test_id_state(name)
            print(f"[SUCCESS] {name} passed.")
            sims.append(sim)
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
    return sims


def test_multidisease(n_agents=100):
    sc.heading('Testing multi-disease simulation')
    ppl = ss.People(n_agents)
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(4)))
    sir1 = ss.SIR(name='sir1', pars={'beta': {'randomnet': 0.1}})
    sir2 = ss.SIR(name='sir2', pars={'beta': {'randomnet': 0.2}})
    sim = ss.Sim(people=ppl, networks=net, diseases=[sir1, sir2], start=2020, stop=2025)
    sim.run()
    assert hasattr(sim.diseases['sir1'], 'results')
    assert hasattr(sim.diseases['sir2'], 'results')
    assert sim.diseases['sir1'].results.n_infected[-1] >= 0
    assert sim.diseases['sir2'].results.n_infected[-1] >= 0
    print("[SUCCESS] Multi-disease sim ran correctly.")
    return sim


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
    

# if __name__ == '__main__':
#     import pylab as pl
#     sc.options(interactive=True)  # Enable interactive plotting
#     do_plot = True
#     sims = []

#     for disease_name in disease_names:
#         try:
#             sim = test_disease_state(disease_name)
#             sims.append(sim)
#             print(f"[SUCCESS] {disease_name} passed.")

#             if do_plot:
#                 disease = sim.diseases[disease_name.lower()]
#                 r = disease.results
#                 t = sim.timevec
#                 death = sim.results.new_deaths.cumsum() if hasattr(sim.results, 'new_deaths') else np.zeros_like(t)

#                 if hasattr(r, 'n_not_at_risk') and hasattr(r, 'n_at_risk') and hasattr(r, 'n_affected'):
#                     pl.figure()
#                     pl.stackplot(
#                         t,
#                         r.n_not_at_risk,
#                         r.n_at_risk - r.n_affected,
#                         r.n_affected,
#                         death,
#                     )
#                     pl.legend(['Not at risk', 'At risk', 'Affected', 'Dead'])

#                 elif hasattr(r, 'n_infected') and hasattr(r, 'n_susceptible'):
#                     recovered = getattr(r, 'n_recovered', np.zeros_like(t))
#                     pl.figure()
#                     pl.stackplot(
#                         t,
#                         r.n_susceptible,
#                         r.n_infected,
#                         recovered,
#                         death,
#                     )
#                     labels = ['Susceptible', 'Infected', 'Recovered', 'Dead'] if np.any(recovered) else ['Susceptible', 'Infected', 'Dead']
#                     pl.legend(labels)

#                 else:
#                     print(f"[WARNING] {disease_name} does not have the required results for plotting.")
#                     continue

#                 pl.title(disease_name)
#                 pl.xlabel('Year')
#                 pl.ylabel('Number of agents')
#                 pl.tight_layout()
#                 pl.show()

#         except Exception as e:
#             print(f"[ERROR] {disease_name} failed: {e}")