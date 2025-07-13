"""
Specifies parameter sets and configurations for each modeled disease
"""


import logging
import numpy as np


def initialize_prevalence_data(diseases, prevalence_data, inityear):
    """
    Initialize the prevalence_data structure for each disease using data from a DataFrame.
    
    Args:
        diseases (list): List of diseases to initialize prevalence data for.
        prevalence_data (DataFrame): DataFrame containing prevalence data.
        inityear (int): Initial year for filtering the prevalence data.
        
    Returns:
        prevalence_data (dict): Dictionary containing prevalence data.
        age_bins (dict): Dictionary containing age bins for each disease.
    """
    df_init = prevalence_data[prevalence_data['Year'] == inityear]

    prevalence_dict = {}
    
    for disease in diseases:
        prevalence_dict[disease] = {'male': {}, 'female': {}}

        for index, row in df_init.iterrows():
            try:
                age = int(row['Age'])
                male_key = f'{disease}_male'
                female_key = f'{disease}_female'
                
                if male_key in row and female_key in row:
                    male_prev = float(row[male_key])
                    female_prev = float(row[female_key])

                    prevalence_dict[disease]['male'][age] = male_prev
                    prevalence_dict[disease]['female'][age] = female_prev
            except (ValueError, KeyError) as e:
                logging.warning(f"Error processing row {index} for {disease}: {e}")
                continue

    age_bins = {disease: sorted(prevalence_dict[disease]['male'].keys()) for disease in prevalence_dict.keys()}
    return prevalence_dict, age_bins


def age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size):
    ages = sim.people.age[size]
    females = sim.people.female[size]
    sex = np.where(females, 'female', 'male')

    bins = age_bins[disease]
    if len(bins) < 2:
        return np.zeros(len(size))  # or fallback
    
    out = np.zeros(len(size))
    for i, b in enumerate(bins[:-1]):
        mask = (ages >= b) & (ages < bins[i + 1])
        for s in ['female', 'male']:
            submask = mask & (sex == s)
            val = prevalence_data[disease][s].get(b, 0.0)
            out[submask] = val
    return out
