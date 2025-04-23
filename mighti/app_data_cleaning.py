import pandas as pd

def data_extraction(country, inityear, endyear):
    

    def process_population_data(male_csv, female_csv, output_csv, country, inityear):

        # Read population data
        male_population = pd.read_csv(male_csv)
        female_population = pd.read_csv(female_csv)

        # Convert 'year' column to numeric, filling invalid values with 0
        male_population['year'] = pd.to_numeric(male_population['year'], errors='coerce').fillna(0).astype(int)
        female_population['year'] = pd.to_numeric(female_population['year'], errors='coerce').fillna(0).astype(int)

        # Filter data for the specified country and year
        male_population = male_population[(male_population['region'] == country) & (male_population['year'] == inityear)]
        female_population = female_population[(female_population['region'] == country) & (female_population['year'] == inityear)]

        # Define the age range (0–100)
        age_range = range(0, 101)

        # Create a DataFrame for the male population
        male_data = pd.DataFrame({
            'age': list(age_range),
            'value': [male_population[str(age)].sum() if str(age) in male_population.columns else 0 for age in age_range],
            'sex': 'Male'
        })

        # Create a DataFrame for the female population
        female_data = pd.DataFrame({
            'age': list(age_range),
            'value': [female_population[str(age)].sum() if str(age) in female_population.columns else 0 for age in age_range],
            'sex': 'Female'
        })

        # Combine male and female data
        combined_data = pd.concat([male_data, female_data], ignore_index=True)

        # Save the processed data to the output CSV file
        combined_data.to_csv(output_csv, index=False)
        print(f"Population data for year {inityear} saved to {output_csv}")

        return output_csv
        
    def process_asfr_data(female_csv, output_csv, country, inityear):

        # Read the CSV file
        asfr_df = pd.read_csv(female_csv, low_memory=False)

        # Filter the data for the specified country and year
        asfr_filtered = asfr_df[(asfr_df['region'] == country) & (asfr_df['year'] == inityear)]

        # if asfr_filtered.empty:
        #     print(f"No data found for region '{country}' and year '{inityear}'.")
        #     return

        # Melt the DataFrame to transpose age columns into rows
        asfr_long = asfr_filtered.melt(
            id_vars=['year'],  # Keep the year column as an identifier
            value_vars=[str(age) for age in range(15, 50)],  # Columns for ages 15–49
            var_name='AgeGrp',  # Name for the age group column
            value_name='ASFR'  # Name for the ASFR (value) column
        )

        # Rename columns to match the desired output format
        asfr_long.rename(columns={'year': 'Time'}, inplace=True)

        # Ensure correct data types
        asfr_long['Time'] = pd.to_numeric(asfr_long['Time'], errors='coerce').fillna(0).astype(int)
        asfr_long['AgeGrp'] = pd.to_numeric(asfr_long['AgeGrp'], errors='coerce').fillna(0).astype(int)
        asfr_long['ASFR'] = pd.to_numeric(asfr_long['ASFR'], errors='coerce')

        # Save the processed data to the output CSV file
        asfr_long[['Time', 'AgeGrp', 'ASFR']].to_csv(output_csv, index=False)
        print(f"ASFR data for year {inityear} successfully saved to {output_csv}")
        return output_csv
            
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
        return output_csv

    def calculate_mortality_rates(age_distribution_csv, mortality_deaths_csv, output_csv, inityear):
        # Load the age distribution and mortality data
        age_distribution = pd.read_csv(age_distribution_csv)
        mortality_deaths = pd.read_csv(mortality_deaths_csv)
        # Filter the data for the specified year
        filtered_data = mortality_deaths[['age', 'sex', str(inityear)]].rename(columns={str(inityear): 'deaths'})
        
        # Define age groups (5-year bins)
        age_groups = range(0, 101, 5)
        
        # Initialize an empty list to store the results
        results = []
        
 
        for sex in ['Male', 'Female']:
            # Filter data for the given year and sex
            age_data = age_distribution[(age_distribution['sex'] == sex)]
            death_data = filtered_data[(filtered_data['sex'] == sex)]
            
            # Iterate over each age group
            for age_start in age_groups:
                age_end = age_start + 5
                # Filter data for the given age group
                age_group_data = age_data[(age_data['age'] >= age_start) & (age_data['age'] < age_end)]
                death_group_data = death_data[(death_data['age'] >= age_start) & (death_data['age'] < age_end)]
                
                # Calculate l(x), d(x), L(x), and m(x)
                lx = age_group_data['value'].sum()
                dx = death_group_data['deaths'].sum()
                Lx = lx - dx + 0.5 * dx
                mx = dx / Lx if Lx > 0 else 0
                
                # Append the results
                results.append([sex, age_start, mx])
        
        results_with_time = [[inityear] + row for row in results]

        # Create a DataFrame with the desired column names
        results_df = pd.DataFrame(results_with_time, columns=['Time', 'Sex', 'AgeGrpStart', 'mx'])
        
        # Save the results to CSV
        results_df.to_csv(output_csv, index=False)
        print(f"Mortality rates saved to {output_csv}")
        return output_csv
    
        
    def process_life_expectancy_data(male_csv, female_csv, output_csv, country):
        # Read life expectancy data
        male_life_expectancy = pd.read_csv(male_csv)
        female_life_expectancy = pd.read_csv(female_csv)
        # male_life_expectancy = pd.read_csv('demography/life_expectancy_by_age_male.csv')
        # female_life_expectancy = pd.read_csv('demography/life_expectancy_by_age_female.csv')
        
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
        return output_csv
    
    def extract_life_expectancy_data(life_expectancy_csv, output_csv, endyear):
        
        # Read the life expectancy data
        # life_expectancy_data = pd.read_csv( f'app/{country}_life_expectancy_by_age.csv')
        life_expectancy_data = pd.read_csv(life_expectancy_csv)
    
        # Debug: Print available columns
        print(f"Available columns in life_expectancy_data: {life_expectancy_data.columns}")
    
        # Ensure the specified year column exists
        endyear_column = str(endyear)
        if endyear_column not in life_expectancy_data.columns:
            raise KeyError(f"Year {endyear} is not available in the life expectancy data. "
                           f"Available year columns: {list(life_expectancy_data.columns[2:])}")
    
        # Extract the data for the specified year
        try:
            selected_data = life_expectancy_data[['age', 'sex', endyear_column]]
        except KeyError as e:
            raise KeyError(f"Error extracting data for year {endyear}. Ensure the file has the correct format.") from e
            
        # Rename the year column to 'year'
        selected_data = selected_data.rename(columns={endyear_column: 'year'})
     
    
        # Save the extracted data to a new CSV file
        selected_data.to_csv(output_csv, index=False)
        print(f"Extracted life expectancy data for {endyear} saved to {output_csv}")
    
        return output_csv
        
    # File paths
    population_csv = process_population_data(
        'demography/population_single_age_male.csv', 
        'demography/population_single_age_female.csv', 
        f'app/{country}_age_distribution_{inityear}.csv', 
        country,
        inityear
    )
    fertility_csv = process_asfr_data(
        'demography/fertility_by_single_age_of_mother.csv', 
        f'app/{country}_asfr_{inityear}.csv', 
        country,
        inityear
    )
    mortality_csv = process_mortality_data(
        'demography/death_single_age_male.csv', 
        'demography/death_single_age_female.csv', 
        f'app/{country}_deaths.csv', 
        country
    )
    mortality_rates_csv = calculate_mortality_rates(
        f'app/{country}_age_distribution_{inityear}.csv',
        f'app/{country}_deaths.csv',
        f'app/{country}_mortality_rates_{inityear}.csv',
        inityear
    )
    life_expectancy_csv = process_life_expectancy_data(
        'demography/life_expectancy_by_age_male.csv',
        'demography/life_expectancy_by_age_female.csv',
        f'app/{country}_life_expectancy_by_age.csv',
        country
    )
    extracted_life_expectancy_csv = extract_life_expectancy_data(
        f'app/{country}_life_expectancy_by_age.csv',
        f'app/{country}_life_expectancy_{endyear}.csv',
        endyear
    )
    
    # Return all CSV paths
    return (
        population_csv,
        fertility_csv,
        mortality_csv,
        mortality_rates_csv,
        life_expectancy_csv,
        extracted_life_expectancy_csv
    )
