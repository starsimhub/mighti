import pandas as pd

def process_population_data(male_csv, female_csv, output_csv, country):
    # Read population data
    male_population = pd.read_csv(male_csv)
    female_population = pd.read_csv(female_csv)
    
    # Handle non-finite values in the year column
    male_population['year'] = pd.to_numeric(male_population['year'], errors='coerce').fillna(0).astype(int)
    female_population['year'] = pd.to_numeric(female_population['year'], errors='coerce').fillna(0).astype(int)
    
    # Filter data for the specified country and years
    male_population = male_population[(male_population['region'] == country) & (male_population['year'])]
    female_population = female_population[(female_population['region'] == country) & (female_population['year'])]
    
    # Extract age-specific data and reorganize
    age_range = range(0, 101)  # Age 0 to 100
    years = sorted(female_population['year'].unique())
    
    data = {
        'age': list(age_range) * 2,
        'sex': ['Male'] * len(age_range) + ['Female'] * len(age_range)
    }
    
    for year in years:
        data[str(year)] = list(male_population[male_population['year'] == year].iloc[:, 3:].values.flatten()) + list(female_population[female_population['year'] == year].iloc[:, 3:].values.flatten())
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Population data saved to {output_csv}")

def process_mortality_data(male_csv, female_csv, output_csv, country):
    # Read mortality data
    male_mortality = pd.read_csv(male_csv)
    female_mortality = pd.read_csv(female_csv)
    
    # Handle non-finite values in the year column
    male_mortality['year'] = pd.to_numeric(male_mortality['year'], errors='coerce').fillna(0).astype(int)
    female_mortality['year'] = pd.to_numeric(female_mortality['year'], errors='coerce').fillna(0).astype(int)
    
    # Filter data for the specified country and years
    male_mortality = male_mortality[male_mortality['region'] == country]
    female_mortality = female_mortality[female_mortality['region'] == country]
    
    # Extract age-specific data and reorganize
    age_range = range(0, 101)  # Age 0 to 100
    years = sorted(female_mortality['year'].unique())
    
    data = {
        'age': list(age_range) * 2,
        'sex': ['Male'] * len(age_range) + ['Female'] * len(age_range)
    }
    
    for year in years:
        male_year_data = male_mortality[male_mortality['year'] == year].iloc[:, 3:].astype(float).values.flatten()
        female_year_data = female_mortality[female_mortality['year'] == year].iloc[:, 3:].astype(float).values.flatten()
        data[str(year)] = list(male_year_data) + list(female_year_data)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Mortality data saved to {output_csv}")

def calculate_mortality_rates(age_distribution_csv, mortality_deaths_csv, output_csv):
    # Load the age distribution and mortality data
    age_distribution = pd.read_csv(age_distribution_csv)
    mortality_deaths = pd.read_csv(mortality_deaths_csv)
    
    # Define age groups (5-year bins)
    age_groups = range(0, 101, 5)
    
    # Get the range of ages from the data
    min_age = age_distribution['age'].min()
    max_age = age_distribution['age'].max()
    ages = range(min_age, max_age)  # Single-year ages (excluding max_age since we need l(x+1))
    
    # Initialize an empty list to store the results
    results = []
    
    # Iterate over each year and sex
    for year in age_distribution.columns[2:]:
        for sex in ['Male', 'Female']:
            # Filter data for the given year and sex
            age_data = age_distribution[(age_distribution['sex'] == sex)]
            death_data = mortality_deaths[(mortality_deaths['sex'] == sex)]
            
    #         # Iterate over each age group
    #         for age_start in age_groups:
    #             age_end = age_start + 5
    #             # Filter data for the given age group
    #             age_group_data = age_data[(age_data['age'] >= age_start) & (age_data['age'] < age_end)]
    #             death_group_data = death_data[(death_data['age'] >= age_start) & (death_data['age'] < age_end)]
                
    #             # Calculate l(x), d(x), L(x), and m(x)
    #             lx = age_group_data[year].sum()
    #             dx = death_group_data[year].sum()
    #             Lx = lx - dx + 0.5 * dx
    #             mx = dx / Lx if Lx > 0 else 0
                
    #             # Append the results
    #             results.append([year, sex, age_start, mx])
    
    # # Create a DataFrame to store the results
    # results_df = pd.DataFrame(results, columns=['Time', 'Sex', 'AgeGrpStart', 'mx'])
    
            # # Iterate over each age
            for age in ages:
                # Filter data for the current age and the next age
                age_row_data = age_data[age_data['age'] == age]
                next_age_row_data = age_data[age_data['age'] == age + 1]  # l(x+1)
                death_row_data = death_data[death_data['age'] == age]
                
                # Extract values for l(x), l(x+1), and d(x)
                lx = age_row_data[year].values[0] if not age_row_data.empty else 0  # Population at age x
                lx_next = next_age_row_data[year].values[0] if not next_age_row_data.empty else 0  # Population at age x+1
                dx = death_row_data[year].values[0] if not death_row_data.empty else 0  # Deaths at age x
                
                # Calculate L(x) and m(x)
                Lx = lx_next + 0.5 * dx  # Correct L(x) calculation
                mx = dx / Lx if Lx > 0 else 0  # Mortality rate
                
                # Append the results
                results.append([year, sex, age, mx])
    
    # Create a DataFrame to store the results
    results_df = pd.DataFrame(results, columns=['Time', 'Sex', 'Age', 'mx'])

    # Save the results to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Mortality rates saved to {output_csv}")
    
    
def process_life_expectancy_data(male_csv, female_csv, output_csv, country):
    # Read life expectancy data
    male_life_expectancy = pd.read_csv(male_csv)
    female_life_expectancy = pd.read_csv(female_csv)
    
    # Handle non-finite values in the year column
    male_life_expectancy['year'] = pd.to_numeric(male_life_expectancy['year'], errors='coerce').fillna(0).astype(int)
    female_life_expectancy['year'] = pd.to_numeric(female_life_expectancy['year'], errors='coerce').fillna(0).astype(int)
    
    # Filter data for the specified country
    male_life_expectancy = male_life_expectancy[(male_life_expectancy['region'] == country) & (male_life_expectancy['year'])]
    female_life_expectancy = female_life_expectancy[(female_life_expectancy['region'] == country) & (female_life_expectancy['year'])]
    
    # Extract age-specific data and reorganize
    age_range = range(0, 101)  # Age 0 to 100
    years = sorted(female_life_expectancy['year'].unique())
    
    data = {
        'age': list(age_range) * 2,
        'sex': ['Male'] * len(age_range) + ['Female'] * len(age_range)
    }
    
    for year in years:
        data[str(year)] = list(male_life_expectancy[male_life_expectancy['year'] == year].iloc[:, 3:].values.flatten()) + list(female_life_expectancy[female_life_expectancy['year'] == year].iloc[:, 3:].values.flatten())
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Life expectancy data for {country} saved to {output_csv}")


if __name__ == "__main__":
    # File paths
    male_csv = 'population_single_age_male.csv'
    female_csv = 'population_single_age_female.csv'
    output_csv = 'eswatini_age_distribution.csv'
    
    process_population_data(male_csv, female_csv, output_csv, country ='Eswatini')
    
    male_csv = 'death_single_age_male.csv'
    female_csv = 'death_single_age_female.csv'
    output_csv = 'eswatini_deaths.csv'
    
    process_mortality_data(male_csv, female_csv, output_csv, country ='Eswatini')
    
    # Load the age distribution data
    age_distribution_csv = 'eswatini_age_distribution.csv'
    
    # Load the mortality data
    mortality_deaths_csv = 'eswatini_deaths.csv'
    
    output_csv = 'eswatini_mortality_rates.csv'
    
    calculate_mortality_rates(age_distribution_csv, mortality_deaths_csv, output_csv)
    
    male_LE_csv = 'life_expectancy_by_age_male.csv'
    female_LE_csv = 'life_expectancy_by_age_female.csv'
    output_LE_csv = 'eswatini_life_expectancy_by_age.csv'
    country = 'Eswatini'
    process_life_expectancy_data(male_LE_csv, female_LE_csv, output_LE_csv, country)