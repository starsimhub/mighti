import pandas as pd
import os

def prepare_data_for_year(year):
    # Define file paths
    csv_path_death = f'mighti/data/eswatini_mortality_rates_{year}.csv'
    csv_path_age = f'mighti/data/eswatini_age_distribution_{year}.csv'
    
    # Check if the files already exist
    if not os.path.exists(csv_path_death):
        # Load the mortality rates data
        mortality_rates = pd.read_csv('eswatini_mortality_rates.csv')
        
        # Extract rows for the specified year
        mortality_rates_year = mortality_rates[mortality_rates['Time'] == year]
        
        # Save the extracted data to a new CSV file
        mortality_rates_year.to_csv(csv_path_death, index=False)
        print(f"Mortality rates for {year} saved to '{csv_path_death}'")
    else:
        print(f"File '{csv_path_death}' already exists.")

    if not os.path.exists(csv_path_age):
        # Load the age distribution data
        age_distribution = pd.read_csv('eswatini_age_distribution.csv')
        
        # Extract rows for the specified year
        age_distribution_year = age_distribution[['age', 'sex', str(year)]]
        
        # Rename the year column to 'population'
        age_distribution_year.rename(columns={str(year): 'population'}, inplace=True)
        
        # Save the extracted data to a new CSV file
        age_distribution_year.to_csv(csv_path_age, index=False)
        print(f"Age distribution for {year} saved to '{csv_path_age}'")
    else:
        print(f"File '{csv_path_age}' already exists.")

if __name__ == "__main__":
    year = 2007
    prepare_data_for_year(year)