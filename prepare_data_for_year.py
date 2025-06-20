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
    df_mx_year.to_csv(output_mx_path, index=False)
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
