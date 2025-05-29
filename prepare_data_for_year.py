import pandas as pd
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

def prepare_data_for_year(year):
    # Define file paths
    csv_path_death = os.path.join(script_dir, 'mighti','data', f'eswatini_mortality_rates_{year}.csv')
    csv_path_age = os.path.join(script_dir, 'mighti','data', f'eswatini_age_distribution_{year}.csv')
    
    # Check if the files already exist
    if not os.path.exists(csv_path_death):
        # Load the mortality rates data
        csv_path_mortality_rates = os.path.join(script_dir, 'demography', 'eswatini_mortality_rates.csv')
        mortality_rates = pd.read_csv(csv_path_mortality_rates)
        
        # Extract rows for the specified year
        mortality_rates_year = mortality_rates[mortality_rates['Time'] == year]
        
        # Rename column 'Age' to 'AgeGrpStart'
        mortality_rates_year = mortality_rates_year.rename(columns={'Age': 'AgeGrpStart'})
        
        # Save the extracted data to a new CSV file
        mortality_rates_year.to_csv(csv_path_death, index=False)
        print(f"Mortality rates for {year} saved to '{csv_path_death}'")
    else:
        print(f"File '{csv_path_death}' already exists.")

    if not os.path.exists(csv_path_age):
        # Load the age distribution data
        csv_path_age_distribution = os.path.join(script_dir, 'demography', 'eswatini_age_distribution.csv')
        age_distribution = pd.read_csv(csv_path_age_distribution)
        
        # Extract rows for the specified year
        age_distribution_year = age_distribution[['age', 'sex', str(year)]]
        
        # Calculate the total population for each age
        age_distribution_year = age_distribution_year.groupby('age')[str(year)].sum().reset_index()
        age_distribution_year.columns = ['age', 'value']
        
        # Save the extracted data to a new CSV file
        age_distribution_year.to_csv(csv_path_age, index=False)
        print(f"Age distribution for {year} saved to '{csv_path_age}'")
    else:
        print(f"File '{csv_path_age}' already exists.")