import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

def prepare_data_for_year(region, year):
    # Define output paths
    csv_path_death = os.path.join(script_dir, 'mighti', 'data', f'{region}_mortality_rates_{year}.csv')
    csv_path_age = os.path.join(script_dir, 'mighti', 'data', f'{region}_age_distribution_{year}.csv')

    # Load wide-format mx file
    csv_path_mortality_rates = os.path.join(script_dir, 'mighti', 'data', f'{region}_mx.csv')
    mortality_rates = pd.read_csv(csv_path_mortality_rates)

    # Melt the data to long format
    melted = mortality_rates.melt(id_vars=['Age', 'Sex'], var_name='Time', value_name='mx')

    # Coerce year and mx to numeric
    melted['Time'] = pd.to_numeric(melted['Time'], errors='coerce')
    melted['mx'] = pd.to_numeric(melted['mx'], errors='coerce')

    # Filter for the specified year
    mortality_rates_year = melted[melted['Time'] == year].dropna(subset=['mx'])

    # Rename Age → AgeGrpStart
    mortality_rates_year = mortality_rates_year.rename(columns={'Age': 'AgeGrpStart'})

    # Save
    mortality_rates_year.to_csv(csv_path_death, index=False)
    print(f"Mortality rates for {year} saved to '{csv_path_death}'")


    # Load the age distribution data
    csv_path_age_distribution = os.path.join(script_dir, 'mighti', 'data', f'{region}_age_distribution.csv')
    age_distribution = pd.read_csv(csv_path_age_distribution)
    
    # Extract rows for the specified year
    age_distribution_year = age_distribution[['age', 'sex', str(year)]]
    
    # Calculate the total population for each age
    age_distribution_year = age_distribution_year.groupby('age')[str(year)].sum().reset_index()
    age_distribution_year.columns = ['age', 'value']
    
    # Save the extracted data to a new CSV file
    age_distribution_year.to_csv(csv_path_age, index=False)
    print(f"Age distribution for {year} saved to '{csv_path_age}'")

        
        
def extract_indicator_for_plot(csv_path, year, value_column_name='mx'):
    """
    Convert a wide-format indicator file (e.g., mx or ex) into long-format for a single year.

    Args:
        csv_path: Path to the wide-format CSV file (e.g., region_mx.csv or region_ex.csv)
        year: Target year to extract
        value_column_name: Name to assign to the melted value column (e.g., 'mx', 'ex')

    Returns:
        DataFrame with columns: ['Age', 'Sex', 'Time', value_column_name]
    """
    df = pd.read_csv(csv_path)
    df = df.melt(id_vars=['Age', 'Sex'], var_name='Time', value_name=value_column_name)
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df[value_column_name] = pd.to_numeric(df[value_column_name], errors='coerce')
    df = df[df['Time'] == year].dropna(subset=[value_column_name])
    return df