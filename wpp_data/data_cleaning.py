import os
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
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
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
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        result.to_csv(output_csv, index=False)
        logging.info(f"{indicator} saved to {output_csv}")
    
    return result


def reshape_mx_to_mortality_rates(mx_df, output_csv):
    """
    Reshape the wide-format mx DataFrame into long format with columns:
    AgeGrpStart, Sex, Time, mx
    and save it to output_csv
    """
    import os

    # Rename columns for clarity
    mx_df = mx_df.rename(columns={mx_df.columns[0]: "AgeGrpStart", mx_df.columns[1]: "Sex"})

    # Melt from wide to long format
    df_long = mx_df.melt(id_vars=["AgeGrpStart", "Sex"], var_name="Time", value_name="mx")

    # Ensure correct data types
    df_long["Time"] = pd.to_numeric(df_long["Time"], errors="coerce")
    df_long["mx"] = pd.to_numeric(df_long["mx"], errors="coerce")

    # Drop any rows with missing time or mx
    df_long = df_long.dropna(subset=["Time", "mx"])

    # Sort for neatness
    df_long = df_long.sort_values(by=["Time", "Sex", "AgeGrpStart"])

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_long.to_csv(output_csv, index=False)
    logging.info(f"Mortality rates saved to {output_csv}")
    
    return df_long


def reshape_fertility_to_asfr(input_csv, region_name, output_csv):
    """
    Convert wide-format fertility data to long-format ASFR for a specific region.
    
    Output format: Time, AgeGrp, ASFR
    """
    import os

    # Read and filter
    df = pd.read_csv(input_csv)
    df = df[df['region'] == region_name]

    if df.empty:
        raise ValueError(f"No rows found for region: {region_name}")

    # Keep only year and age columns
    id_vars = ['year']
    value_vars = [str(age) for age in range(15, 50)]  # ASFR is usually defined for 15–49
    df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='AgeGrp', value_name='ASFR')

    # Clean and rename
    df_long = df_long.rename(columns={'year': 'Time'})
    df_long['AgeGrp'] = df_long['AgeGrp'].astype(int)
    df_long['ASFR'] = pd.to_numeric(df_long['ASFR'], errors='coerce')
    df_long = df_long.dropna()

    # Sort
    df_long = df_long.sort_values(by=['Time', 'AgeGrp'])

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_long.to_csv(output_csv, index=False)
    logging.info(f"Fertility data saved to {output_csv}")

    return df_long


def create_empty_prevalence_template(output_csv, start_year=1987, end_year=2021, age_range=range(0, 101, 5)):
    """
    Create a fully empty prevalence template (all NaNs) with Age x Year x Condition structure.
    No input data required. Columns include all modeled conditions by sex.
    """
    import os

    # Define modeled conditions
    MODELED_CONDITIONS = [
        "HIV", "Obesity", "Hypertension", "AlcoholUseDisorder", "AlzheimersDisease",
        "Asthma", "BreastCancer", "COPD", "CervicalCancer", "ChronicKidneyDisease",
        "ChronicLiverDisease", "ColorectalCancer", "MajorDepressiveDisorder", "CardiovascularDiseases",
        "LungCancer", "ParkinsonsDisease", "ProstateCancer", "RoadInjuries", "Transportinjuries",
        "Type1Diabetes", "Type2Diabetes", "Hyperlipidemia", "TobaccoUse", "Dementia", "PTSD", "HPV", "Flu",
        "ViralHepatitis", "InterpersonalViolence"
    ]

    # Generate condition columns for both sexes
    EXPECTED_COLUMNS = [f"{cond}_{sex}" for cond in MODELED_CONDITIONS for sex in ['male', 'female']]

    # Create full Age x Year grid
    import pandas as pd
    grid = pd.MultiIndex.from_product(
        [age_range, range(start_year, end_year + 1)],
        names=["Age", "Year"]
    ).to_frame(index=False)

    # Add empty columns
    for col in EXPECTED_COLUMNS:
        grid[col] = float('nan')

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    grid.to_csv(output_csv, index=False)
    logging.info(f"Empty prevalence template saved to {output_csv}")

    return grid



def create_condition_metadata_table(output_csv):
    import pandas as pd
    import os

    # Define list of conditions and metadata
    data = [
        # condition, disease_type, disease_class, affected_sex
        ("Type1Diabetes", "chronic", "ncd", "both"),
        ("Type2Diabetes", "remitting", "ncd", "both"),
        ("Hypertension", "chronic", "ncd", "both"),
        ("Obesity", "chronic", "ncd", "both"),
        ("CardiovascularDiseases", "chronic", "ncd", "both"),
        ("ChronicKidneyDisease", "chronic", "ncd", "both"),
        ("Hyperlipidemia", "chronic", "ncd", "both"),
        ("CervicalCancer", "chronic", "ncd", "female"),
        ("ColorectalCancer", "chronic", "ncd", "both"),
        ("BreastCancer", "chronic", "ncd", "both"),
        ("LungCancer", "chronic", "ncd", "both"),
        ("ProstateCancer", "chronic", "ncd", "male"),
        ("AlcoholUseDisorder", "remitting", "ncd", "both"),
        ("TobaccoUse", "remitting", "ncd", "both"),
        ("Dementia", "chronic", "ncd", "both"),
        ("PTSD", "chronic", "ncd", "both"),
        ("MajorDepressiveDisorder", "remitting", "ncd", "both"),
        ("HPV", "remitting", "sis", "both"),
        ("Flu", "chronic", "sis", "both"),
        ("ViralHepatitis", "chronic", "ncd", "both"),
        ("InterpersonalViolence", "acute", "ncd", "both"),
        ("RoadInjuries", "acute", "ncd", "both"),
        ("ChronicLiverDisease", "chronic", "ncd", "both"),
        ("Asthma", "chronic", "ncd", "both"),
        ("COPD", "chronic", "ncd", "both"),
        ("AlzheimersDisease", "chronic", "ncd", "both"),
        ("ParkinsonsDisease", "chronic", "ncd", "both"),
    ]

    # Create DataFrame
    df = pd.DataFrame(data, columns=["condition", "disease_type", "disease_class", "affected_sex"])

    # Add empty columns for parameters to be filled in later
    df["p_death"] = ""
    df["dur_condition"] = ""
    df["rel_sus"] = ""
    df["remission_rate"] = ""
    df["max_disease_duration"] = ""

    # Reorder columns
    df = df[[
        "condition", "p_death", "dur_condition", "rel_sus", "remission_rate",
        "max_disease_duration", "disease_type", "disease_class", "affected_sex"
    ]]

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logging.info(f"Condition metadata table saved to {output_csv}")

    return df


if __name__ == "__main__":
    region = 'eswatini'
    Region = 'Eswatini'

    ### Age distribution data ###
    male_csv = 'population_single_age_male.csv'
    female_csv = 'population_single_age_female.csv'
    output_csv = f"../mighti/data/{region}_age_distribution.csv"
    
    process_population_data(male_csv, female_csv, output_csv, country = Region)
    
    
    ### Life table ###
    life_table = extract_life_table_by_country(
        'life_table_male_1986_1995.csv', 'life_table_male_1996_2005.csv', 'life_table_male_2006_2015.csv', 'life_table_male_2016_2023.csv',
        'life_table_female_1986_1995.csv', 'life_table_female_1996_2005.csv', 'life_table_female_2006_2015.csv', 'life_table_female_2016_2023.csv',
        country=Region
    )
    
    # Extract and save mx and ex
    mx_df = extract_indicator_from_life_table(life_table, 'mx', f"../mighti/data/{region}_mx.csv")
    ex_df = extract_indicator_from_life_table(life_table, 'ex', f"../mighti/data/{region}_ex.csv")
    
    output_csv_mortality = f"../mighti/data/{region}_mortality_rates.csv"
    reshape_mx_to_mortality_rates(mx_df, output_csv_mortality)
    
    csv_path_fertility = f"../mighti/data/{region}_asfr.csv"
    reshape_fertility_to_asfr(
        input_csv="fertility_by_single_age_of_mother.csv",
        region_name=Region,
        output_csv=csv_path_fertility
    )

    csv_prevalence = f"../mighti/data/{region}_prevalence.csv"
    create_empty_prevalence_template(csv_prevalence)
    
    output_csv = f"../mighti/data/{region}_parameters.csv"
    create_condition_metadata_table(output_csv)
    
    
