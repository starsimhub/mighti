import pandas as pd
import numpy as np

def map_to_age_groups(data, ages, age_groups):
    """
    Map population or mortality data to specific age groups.
    """
    # Assuming 'ages' and 'age_groups' are lists of the same length
    mapped_data = np.zeros(len(age_groups))
    for i, age_group in enumerate(age_groups):
        mask = (ages >= age_group) & (ages < age_group + 5)
        mapped_data[i] = data[mask].sum()
    return mapped_data

def calculate_mortality_rate(population_data, mortality_data, ages, age_groups):
    """
    Calculate mortality rates using L(x).
    """
    lx = population_data
    dx = mortality_data
    Lx = (lx - dx) + (dx / 2)
    mx = dx / Lx
    mx = np.nan_to_num(mx)  # Replace NaN with 0
    return mx

def read_and_calculate_mortality(csv_population_path, csv_mortality_path, country, year, output_file):
    """
    Read population and mortality data, calculate mortality rates, and store the results.
    """
    # Read population data
    pop_data = pd.read_csv(csv_population_path)
    pop_data = pop_data[(pop_data['region'] == country) & (pop_data['year'] == year)]
    
    # Read mortality data
    death_data = pd.read_csv(csv_mortality_path)
    death_data = death_data[(death_data['region'] == country) & (death_data['year'] == year)]
    
    # Extract age-specific data
    ages = np.arange(0, 101)
    age_groups = np.arange(0, 101, 5)
    
    population = pop_data.iloc[:, 11:].values.flatten().astype(np.float)
    mortality = death_data.iloc[:, 11:].values.flatten().astype(np.float)
    
    # Map data to age groups
    mapped_pop = map_to_age_groups(population, ages, age_groups)
    mapped_death = map_to_age_groups(mortality, ages, age_groups)
    
    # Calculate mortality rates
    mx = calculate_mortality_rate(mapped_pop, mapped_death, ages, age_groups)
    
    # Create DataFrame to store results
    result_df = pd.DataFrame({
        'Time': year,
        'Sex': 'Both',
        'AgeGrpStart': age_groups,
        'mx': mx
    })
    
    # Save results to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Mortality rates saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    csv_population_path = 'population_single_age_male.csv'
    csv_mortality_path = 'death_single_age_male.csv'
    country = 'Eswatini'
    year = 2020
    output_file = f'{country}_deaths.csv'
    
    read_and_calculate_mortality(csv_population_path, csv_mortality_path, country, year, output_file)