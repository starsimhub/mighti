"""
Data preparation utility for extracting year-specific mortality rates and age distributions
for use in life table and simulation analyses.
"""


import logging
import os
import pandas as pd


logger = logging.getLogger(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))


def prepare_data_for_year(region, year):
    """
    Prepare year-specific mortality rates and age distribution files from wide-format input.

    Args:
        region (str): Region or country identifier (used in file naming).
        year (int): Year to extract data for.

    Outputs:
        - {region}_mortality_rates_{year}.csv
        - {region}_age_distribution_{year}.csv
        (Both saved in mighti/data/)
    """
    # ------------------------------------------------------------------
    # Extract mortality rates
    # ------------------------------------------------------------------
    input_mx_path = os.path.join(script_dir, 'mighti', 'data', f'{region}_mx.csv')
    output_mx_path = os.path.join(script_dir, 'mighti', 'data', f'{region}_mortality_rates_{year}.csv')

    df_mx = pd.read_csv(input_mx_path)
    df_mx = df_mx.melt(id_vars=['Age', 'Sex'], var_name='Time', value_name='mx')
    df_mx['Time'] = pd.to_numeric(df_mx['Time'], errors='coerce')
    df_mx['mx'] = pd.to_numeric(df_mx['mx'], errors='coerce')
    df_mx_year = df_mx[df_mx['Time'] == year].dropna(subset=['mx'])

    df_mx_year = df_mx_year.rename(columns={'Age': 'AgeGrpStart'})
    # df_mx_year.to_csv(output_mx_path, index=False)
    logger.info(f"Mortality rates for {year} saved to '{output_mx_path}'")

    # ------------------------------------------------------------------
    # Extract age distribution
    # ------------------------------------------------------------------
    input_age_path = os.path.join(script_dir, 'mighti', 'data', f'{region}_age_distribution.csv')
    output_age_path = os.path.join(script_dir, 'mighti', 'data', f'{region}_age_distribution_{year}.csv')

    df_age = pd.read_csv(input_age_path)
    if str(year) not in df_age.columns:
        raise ValueError(f"Year {year} not found in age distribution file: {input_age_path}")

    df_age_year = df_age[['age', 'sex', str(year)]]
    df_age_year = df_age_year.groupby('age')[str(year)].sum().reset_index()
    df_age_year.columns = ['age', 'value']
    df_age_year.to_csv(output_age_path, index=False)
    logger.info(f"Age distribution for {year} saved to '{output_age_path}'")


def prepare_data(region):
    """
    Prepare year-specific mortality rates and age distribution files from wide-format input.

    Args:
        region (str): Region or country identifier (used in file naming).
        year (int): Year to extract data for.

    Outputs:
        - {region}_mortality_rates_{year}.csv
        - {region}_age_distribution_{year}.csv
        (Both saved in mighti/data/)
    """
    # ------------------------------------------------------------------
    # Extract mortality rates
    # ------------------------------------------------------------------
    input_mx_path = os.path.join(script_dir, 'mighti', 'data', f'{region}_mx.csv')
    output_mx_path = os.path.join(script_dir, 'mighti', 'data', f'{region}_mortality_rates.csv')

    df_mx = pd.read_csv(input_mx_path)
    df_mx = df_mx.melt(id_vars=['Age', 'Sex'], var_name='Time', value_name='mx')
    df_mx['Time'] = pd.to_numeric(df_mx['Time'], errors='coerce')
    df_mx['mx'] = pd.to_numeric(df_mx['mx'], errors='coerce')

    df_mx_year = df_mx.rename(columns={'Age': 'AgeGrpStart'})
    df_mx_year.to_csv(output_mx_path, index=False)
    logger.info(f"Mortality rates saved to '{output_mx_path}'")
    
    
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
