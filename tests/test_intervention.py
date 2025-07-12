"""
Test that ReduceMortalityTx reduces deaths due to Type 2 Diabetes.

Runs two simulations:
- one with T2D only
- one with T2D and the ReduceMortalityTx intervention

Asserts that total deaths among people with T2D are lower with the intervention.
"""


import sciris as sc
import starsim as ss
import mighti as mi
import pandas as pd
import os
from mighti.diseases.type2diabetes import ReduceMortalityTx


# Settings
do_plot = False
sc.options(interactive=do_plot)

# File path to parameter file
thisdir = os.path.dirname(__file__)
param_path = os.path.join(thisdir, 'test_data', 'eswatini_parameters.csv')
params_df = pd.read_csv(param_path)
params_df.columns = params_df.columns.str.strip()



def test_reduce_mortality_tx_runs():

    # Make disease
    t2d = mi.Type2Diabetes(
        csv_path=param_path,
        pars={
            'init_prev': ss.bernoulli(0.9),
            'p_death': ss.bernoulli(0.1),
            'dur_condition': ss.normal(5, 0.01),
            'max_disease_duration': 10,
        }
    )

    # Create a minimal treatment product for T2D
    tx_df = pd.DataFrame({
        'disease': ['type2diabetes'],
        'state': ['affected'],
        'post_state': ['on_treatment'],
        'efficacy': [1.0]  # Ensure treatment always succeeds
    })
    tx = ss.Tx(df=tx_df)

    # Intervention to reduce death
    t2d_tx = ReduceMortalityTx(
        product=tx,
        rel_death_reduction=0.5,
        eligibility=lambda sim: sim.diseases.type2diabetes.affected.uids,
        label='T2D Mortality Reduction'
    )

    pars = dict(
        start=2000,
        stop=2050,
        dt=1,
        n_agents=10000,
        networks=[],
        diseases=[t2d],
    )

    sim0 = ss.Sim(pars).run()

    # Add intervention to the second sim
    pars['interventions'] = [t2d_tx]
    sim1 = ss.Sim(pars).run()

    # Compare total T2D deaths
    d0 = sim0.results.type2diabetes.new_deaths.values.sum()
    d1 = sim1.results.type2diabetes.new_deaths.values.sum()
    assert d1 < d0, f'T2D deaths should be lower with intervention: {d1=} < {d0=}'
    print(f'✓ T2D deaths reduced from {d0:.0f} → {d1:.0f} by ReduceMortalityTx')
    
    
if __name__ == '__main__':        
    test_reduce_mortality_tx_runs()
    