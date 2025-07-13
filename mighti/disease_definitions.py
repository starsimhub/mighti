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


# Function to compute age and sex-dependent prevalence
def age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size):
    """
    Return the age- and sex-dependent prevalence for a given disease.
    
    Args:
        disease (str): Disease name.
        prevalence_data (dict): Prevalence data for diseases.
        age_bins (dict): Age bins for diseases.
        sim (object): Simulation object with population data.
        size (int): Size of the population subset.
        
    Returns:
        np.array: Prevalence values for the subset of the population.
    """
    ages = sim.people.age[size]
    females = sim.people.female[size]
    prevalence = np.zeros(len(ages))
    disease_age_bins = age_bins[disease]  # Get age bins for the specific disease

    for i in range(len(ages)):
        sex = 'female' if females[i] else 'male'
        # Ensure bins are processed correctly
        for j in range(len(disease_age_bins) - 1):
            left = disease_age_bins[j]
            right = disease_age_bins[j + 1]
            if ages[i] >= left and ages[i] < right:
                prevalence[i] = prevalence_data[disease][sex][left]
                break
        if ages[i] >= 80:  # For ages 80+
            if 80 in prevalence_data[disease][sex]:  # Check if 80+ data is available
                prevalence[i] = prevalence_data[disease][sex][80]

    return prevalence
