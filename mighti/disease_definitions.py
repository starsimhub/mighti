import numpy as np
import pandas as pd

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
                if male_key in df_init.columns and female_key in df_init.columns:
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


def age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size):
    """
    Compute age- and sex-dependent prevalence for a given disease.

    Args:
        disease (str): Disease name.
        prevalence_data (dict): Prevalence data for diseases.
        age_bins (dict): Age bins for diseases.
        sim (object): Simulation object with population data.
        size (array): Indices of the population subset.

    Returns:
        np.array: Prevalence values for the subset of the population.
    """
    # **Fix: Use .raw to extract values before indexing**
    ages = sim.people.age.raw[size]  
    is_male = sim.people.male.raw[size]  # Starsim only tracks male, not female
    
    prevalence = np.zeros(len(ages))
    disease_age_bins = age_bins[disease]  # Get age bins for the specific disease

    for i in range(len(ages)):
        sex = 'male' if is_male[i] else 'female'

        # Assign prevalence based on age bins
        for j in range(len(disease_age_bins) - 1):
            left = disease_age_bins[j]
            right = disease_age_bins[j + 1]
            if left <= ages[i] < right:
                prevalence[i] = prevalence_data[disease][sex][left]
                break

        # Special case for 80+ age group
        if ages[i] >= 80 and 80 in prevalence_data[disease][sex]:
            prevalence[i] = prevalence_data[disease][sex][80]

    return prevalence