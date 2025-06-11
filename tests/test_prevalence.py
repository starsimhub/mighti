"""
Test MIGHTI disease prevalence initialization and age-sex assignment
"""

import mighti as mi
import starsim as ss
import sciris as sc
import pandas as pd
import numpy as np
import os

def test_disease_prevalence_from_data(n_agents=500, inityear=2007):
    # Paths and loading
    thisdir = os.path.abspath(os.path.dirname(__file__))
    prevalence_path = os.path.join(thisdir, '..', 'mighti', 'data', 'eswatini_prevalence.csv')
    prevalence_df = pd.read_csv(prevalence_path)
    prevalence_df.columns = prevalence_df.columns.str.strip()

    # Extract diseases
    diseases = prevalence_df['condition'].unique().tolist()

    # Initialize prevalence matrix and bins
    prevalence_data, age_bins = mi.initialize_prevalence_data(
        diseases=diseases,
        prevalence_data=prevalence_df,
        inityear=inityear
    )

    # Make fake sim to test prevalence sampling
    sim = ss.Sim(n_agents=n_agents)
    sim.initialize()

    for disease in diseases:
        prev_fn = lambda module, sim, size: mi.age_sex_dependent_prevalence(
            disease, prevalence_data, age_bins, sim, size
        )
        sample = ss.bernoulli(prev_fn(None, sim, n_agents))
        prevalence_value = sample.mean()

        print(f'{disease}: Simulated prevalence = {prevalence_value:.3f}')
        assert 0 <= prevalence_value <= 1, f'{disease} prevalence out of bounds'

    print(' All prevalence functions returned valid outputs.')

# Run as a script (optional)
if __name__ == '__main__':
    test_disease_prevalence_from_data()