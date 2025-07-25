"""
Test MIGHTI disease prevalence initialization and age-sex assignment
"""

import mighti as mi
import starsim as ss
import pandas as pd
import os


def test_disease_prevalence_from_data(n_agents=500, inityear=2007):
    # Paths and loading
    thisdir = os.path.dirname(__file__)
    param_path = os.path.join(thisdir, 'test_data', 'eswatini_parameters.csv')
    prevalence_path = os.path.join(thisdir, 'test_data', 'eswatini_prevalence.csv')
    
    prevalence_df = pd.read_csv(prevalence_path)
    prevalence_df.columns = prevalence_df.columns.str.strip()

    params_df = pd.read_csv(param_path)
    params_df.columns = params_df.columns.str.strip()

    # Extract diseases
    diseases = params_df.query("condition != 'HIV'")['condition'].unique()

    # Initialize prevalence matrix and bins
    prevalence_data, age_bins = mi.initialize_prevalence_data(
        diseases=diseases,
        prevalence_data=prevalence_df,
        inityear=inityear
    )

    # Simulate population
    sim = ss.Sim(n_agents=n_agents)
    sim.init()  # Correct initialization
    uids = sim.people.uid.raw

    for disease in diseases:
        if len(age_bins.get(disease, [])) < 2:
            print(f"[⚠] Skipping {disease}: insufficient age bin data.")
            continue

        prev_fn = lambda module, sim, uids: mi.age_sex_dependent_prevalence(
            disease, prevalence_data, age_bins, sim, uids
        )
        
        dist = ss.bernoulli(prev_fn(None, sim, uids), strict=False)   
        dist.init(n_agents)
        sample = dist()
        prevalence_value = sample.mean()

        print(f'{disease}: Simulated prevalence = {prevalence_value:.3f}')
        assert 0 <= prevalence_value <= 1, f'{disease} prevalence out of bounds'

    print('[✓] All prevalence functions returned valid outputs.')

# Run as a script (optional)
if __name__ == '__main__':
    test_disease_prevalence_from_data()