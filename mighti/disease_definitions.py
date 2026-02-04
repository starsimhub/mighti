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
def age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, uids=None, size=None):
    """
    Return the age- and sex-dependent prevalence for a given disease.
    
    Args:
        disease (str): Disease name.
        prevalence_data (dict): Prevalence data for diseases.
        age_bins (dict): Age bins for diseases.
        sim (object): Simulation object with population data.
        uids (array-like): Agent indices to compute prevalence for (StarSim 3.x signature).
        size (any): Back-compat alias for `uids` (some older call sites used `size`).
        
    Returns:
        np.array: Prevalence values for the subset of the population.
    """
    # Prefer StarSim 3.x `uids`, but accept older `size` usage as an alias.
    if uids is None:
        uids = size

    # Default to "all agents" if nothing was provided.
    if uids is None:
        if hasattr(sim, "people") and hasattr(sim.people, "uids"):
            uids = sim.people.uids
        elif hasattr(sim, "people") and hasattr(sim.people, "uid") and hasattr(sim.people.uid, "raw"):
            uids = sim.people.uid.raw
        else:
            raise ValueError("Must provide `uids` (or legacy `size`) to compute prevalence.")

    if isinstance(uids, slice):
        uids = np.arange(len(sim.people))[uids]

    uids = np.asarray(uids, dtype=int)
    ages = np.asarray(sim.people.age[uids], dtype=float)
    females = np.asarray(sim.people.female[uids], dtype=bool)

    prevalence = np.zeros(len(uids), dtype=float)
    disease_age_bins = np.asarray(age_bins[disease], dtype=float)  # age-bin left edges

    if disease_age_bins.size == 0:
        return prevalence

    # Map each age to a left-edge bin (same behavior as "left <= age < right").
    bin_idx = np.searchsorted(disease_age_bins, ages, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, disease_age_bins.size - 1)
    left_edges = disease_age_bins[bin_idx].astype(int)

    # Fill prevalence by sex and age bin.
    for sex, mask in (("female", females), ("male", ~females)):
        if not np.any(mask):
            continue
        sex_prev = prevalence_data[disease].get(sex, {})
        # Default missing bins to 0.0 (robust to sparse tables).
        prevalence[mask] = np.array([float(sex_prev.get(int(a), 0.0)) for a in left_edges[mask]], dtype=float)

        # For ages 80+, override with 80+ bucket if available (matches legacy behavior).
        if 80 in sex_prev:
            over80 = mask & (ages >= 80)
            if np.any(over80):
                prevalence[over80] = float(sex_prev[80])

    return prevalence
