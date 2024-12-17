import numpy as np
import pandas as pd
import starsim as ss

# Function to load condition parameters from a CSV file
def load_disease_parameters(disease_name, csv_file_path):
    """
    Load disease-specific parameters from a CSV file.

    Args:
        disease_name (str): The name of the disease (case-insensitive).
        csv_file_path (str): Path to the CSV file containing disease parameters.

    Returns:
        dict: A dictionary containing the disease parameters.
    """
    df = pd.read_csv(csv_file_path)
    row = df[df['condition'].str.lower() == disease_name.lower()]
    if row.empty:
        raise ValueError(f"No parameters found for disease: {disease_name}")

    # Extract parameters and convert to appropriate types
    disease_params = {
        'dur_condition': eval(f"ss.{row.iloc[0]['dur_condition']}"),
        'incidence': ss.bernoulli(float(row.iloc[0]['incidence'])),
        'p_death': ss.bernoulli(float(row.iloc[0]['p_death'])),
        'init_prev': ss.bernoulli(float(row.iloc[0]['init_prev']))
    }
    return disease_params

# Function to initialize prevalence data and age bins
def initialize_prevalence_data(diseases, csv_file_path, inityear):
    """
    Initialize the prevalence_data structure for each disease using data from a CSV file.
    
    Args:
        diseases (list): List of diseases to initialize prevalence data for.
        csv_file_path (str): Path to the CSV file containing prevalence data.
        
    Returns:
        prevalence_data (dict): Dictionary containing prevalence data.
        age_bins (dict): Dictionary containing age bins for each disease.
    """
    # Load prevalence data from the CSV file
    df = pd.read_csv(csv_file_path)

    # Initialize an empty dictionary for storing the prevalence data
    prevalence_data = {}

    # Filter the DataFrame to include only the year 2007
    df_init = df[df['Year'] == inityear]

    # Populate the prevalence_data structure for each disease
    for disease in diseases:
        prevalence_data[disease] = {'male': {}, 'female': {}}

        # Loop over the rows in the filtered DataFrame
        for index, row in df_init.iterrows():
            try:
                age = int(row['Age'])
                male_key = f'{disease}_male'
                female_key = f'{disease}_female'
                
                # Check if the required columns exist in the row
                if male_key in row and female_key in row:
                    male_prev = float(row[male_key])
                    female_prev = float(row[female_key])

                    # Add the prevalence data to the dictionary
                    prevalence_data[disease]['male'][age] = male_prev
                    prevalence_data[disease]['female'][age] = female_prev
            except (ValueError, KeyError) as e:
                # Handle missing values or key errors
                print(f"Error processing row {index} for {disease}: {e}")
                continue

    # Extract age bins from the loaded prevalence_data
    age_bins = {disease: sorted(prevalence_data[disease]['male'].keys()) for disease in prevalence_data.keys()}

    return prevalence_data, age_bins


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