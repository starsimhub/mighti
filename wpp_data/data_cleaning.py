import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

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
    logging.info(f"Population data saved to {output_csv}")

def extract_life_table_by_country(male_csv1, male_csv2, male_csv3, male_csv4, female_csv1, female_csv2, female_csv3, female_csv4, country):
    def load_and_clean(filepath, sex):
        df = pd.read_csv(filepath, low_memory=False)
        df = df[df['region'] == country].copy()
        df['Sex'] = sex
        df = df.rename(columns={'year': 'Time', 'age': 'Age'})
        numeric_cols = ['Time', 'Age', 'mx', 'ex']  # Add more if needed
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    male = pd.concat([load_and_clean(male_csv1, 'Male'), load_and_clean(male_csv2, 'Male'), load_and_clean(male_csv3, 'Male'), load_and_clean(male_csv4, 'Male')])
    female = pd.concat([load_and_clean(female_csv1, 'Female'), load_and_clean(female_csv2, 'Female'), load_and_clean(female_csv3, 'Female'), load_and_clean(female_csv4, 'Female')])
    return pd.concat([male, female], ignore_index=True)

def extract_indicator_from_life_table(life_table_df, indicator, output_csv=None):
    df = life_table_df[['Time', 'Age', 'Sex', indicator]].dropna()
    df[indicator] = pd.to_numeric(df[indicator], errors='coerce')
    result = df.pivot_table(index=['Age', 'Sex'], columns='Time', values=indicator).reset_index()
    
    if output_csv:
        result.to_csv(output_csv, index=False)
        logging.info(f"{indicator} saved to {output_csv}")
    
    return result


if __name__ == "__main__":
    

    ### Age distribution data ###
    male_csv = 'population_single_age_male.csv'
    female_csv = 'population_single_age_female.csv'
    output_csv = '../demography/eswatini_age_distribution.csv'
    
    process_population_data(male_csv, female_csv, output_csv, country ='Eswatini')
    
    
    ### Life table ###
    life_table = extract_life_table_by_country(
        'life_table_male_1986_1995.csv', 'life_table_male_1996_2005.csv', 'life_table_male_2006_2015.csv', 'life_table_male_2016_2023.csv',
        'life_table_female_1986_1995.csv', 'life_table_female_1996_2005.csv', 'life_table_female_2006_2015.csv', 'life_table_female_2016_2023.csv',
        country='Eswatini'
    )
    
    # Extract and save mx and ex
    mx_df = extract_indicator_from_life_table(life_table, 'mx', '../demography/eswatini_mx.csv')
    ex_df = extract_indicator_from_life_table(life_table, 'ex', '../demography/eswatini_ex.csv')
    
   