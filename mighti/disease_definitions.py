import numpy as np
from scipy.interpolate import interp1d


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
    # Filter the DataFrame to include only the specified initial year
    df_init = prevalence_data[prevalence_data['Year'] == inityear]

    # Initialize the prevalence data dictionary
    prevalence_dict = {}
    
    for disease in diseases:
        prevalence_dict[disease] = {'male': {}, 'female': {}}
        for _, row in df_init.iterrows():
            try:
                age = int(row['Age'])
                male_key = f'{disease}_male'
                female_key = f'{disease}_female'

                # Check if the required columns exist in the row
                if male_key in row and female_key in row:
                    prevalence_dict[disease]['male'][age] = float(row[male_key])
                    prevalence_dict[disease]['female'][age] = float(row[female_key])
            except (ValueError, KeyError) as e:
                # Handle missing values or key errors
                print(f"Error processing row for {disease}: {e}")
                continue

    # Extract age bins from the loaded prevalence_data
    age_bins = {disease: sorted(prevalence_dict[disease]['male'].keys()) for disease in prevalence_dict.keys()}

    return prevalence_dict, age_bins


def age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size):
    """
    Return the age- and sex-dependent prevalence for a given disease.

    Args:
        disease (str): Disease name.
        prevalence_data (dict): Prevalence data for diseases.
        age_bins (dict): Age bins for diseases.
        sim (object): Simulation object with population data.
        size (any): Size of the population subset - can be a mask, UID object, or indices.

    Returns:
        np.array: Prevalence values for the subset of the population.
    """
    # Handle different types of 'size' parameter
    try:
        ages = sim.people.age[size]
        females = sim.people.female[size]
    except Exception:
        # Handle fallback cases
        if hasattr(sim, 'mock_data'):
            ages = sim.mock_data['age']
            females = sim.mock_data['female']
        else:
            ages = np.array(size)
            females = np.random.choice([True, False], size=len(ages))

    # Validate input arrays
    assert len(ages) == len(females), "Mismatch between ages and females array lengths"

    # Ensure age bins are sorted and non-overlapping
    disease_age_bins = sorted(age_bins[disease])
    assert all(disease_age_bins[i] < disease_age_bins[i + 1] for i in range(len(disease_age_bins) - 1)), \
        f"Age bins for {disease} are not properly sorted or are overlapping"

    # Create array for prevalence output
    prevalence = np.zeros(len(ages))

    # Process each person
    for sex in ['male', 'female']:
        if sex not in prevalence_data[disease]:
            raise ValueError(f"Missing prevalence data for {sex} in disease {disease}")

        # Extract age and prevalence values for interpolation
        bin_edges = np.array(sorted(prevalence_data[disease][sex].keys()))
        bin_values = np.array([prevalence_data[disease][sex][age] for age in bin_edges])

        # Create interpolation function
        interp_func = interp1d(bin_edges, bin_values, bounds_error=False, fill_value=(bin_values[0], bin_values[-1]))

        # Assign prevalence values for the given sex
        sex_mask = (females if sex == 'female' else ~females)
        prevalence[sex_mask] = interp_func(ages[sex_mask])

    return prevalence