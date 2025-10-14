import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)


cause_map = {
    "Diabetes mellitus type 1": "Type1Diabetes",
    "Diabetes mellitus type 2": "Type2Diabetes",
    "Hypertension": "Hypertension",
    "High body-mass index": "Obesity",
    "Cardiovascular diseases": "CardiovascularDiseases",
    "Chronic kidney disease": "ChronicKidneyDisease",
    "High LDL cholesterol": "Hyperlipidemia",
    "Cervical cancer": "CervicalCancer",
    "Colon and rectum cancer": "ColorectalCancer",
    "Breast cancer": "BreastCancer",
    "Tracheal, bronchus, and lung cancer": "LungCancer",
    "Prostate cancer": "ProstateCancer",
    "Alcohol use disorders": "AlcoholUseDisorder",
    "Tobacco use": "TobaccoUse",
    "Dementia": "Dementia",
    "Post-traumatic stress disorder": "PTSD",
    "Major depressive disorder": "MajorDepressiveDisorder",
    "Human papillomavirus infection": "HPV",
    "Influenza and pneumonia": "Flu",
    "Hepatitis B": "ViralHepatitis",
    "Hepatitis C": "ViralHepatitis",
    "Interpersonal violence": "InterpersonalViolence",
    "Road injuries": "RoadInjuries",
    "Cirrhosis and other chronic liver diseases": "ChronicLiverDisease",
    "Asthma": "Asthma",
    "Chronic obstructive pulmonary disease": "COPD",
    "Alzheimer’s disease and other dementias": "AlzheimersDisease",
    "Parkinson’s disease": "ParkinsonsDisease"
}

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


import re

def create_and_fill_prevalence_template_from_long_format(
    raw_csv,
    output_csv,
    start_year=1987,
    end_year=2021,
    age_range=range(0, 101, 5)
):
    """
    Create and fill a prevalence template using a long-format prevalence file.

    Args:
        raw_csv (str): Path to the raw prevalence data in long format (age, year, sex, cause, val).
        output_csv (str): Path to save the filled prevalence CSV.
        start_year (int): Start year for the template.
        end_year (int): End year for the template.
        age_range (range): Age range (e.g., range(0, 101, 5)).
    """


    def map_age_group(age_str):
        """
        Map age group strings like '80-84 years', '<5 years', '85+ years' to integer start age.
        """
        if isinstance(age_str, str):
            age_str = age_str.strip()
            if re.match(r'^\<\s*5', age_str):
                return 0
            elif re.match(r'^\d+\s*\-\s*\d+', age_str):
                return int(re.match(r'^(\d+)', age_str).group(1))
            elif re.match(r'^\d+\+?', age_str):
                return int(re.match(r'^(\d+)', age_str).group(1))
        return None

    # Step 1: Create empty template
    modeled_conditions = list(set(cause_map.values()))
    expected_columns = [f"{cond}_{sex}" for cond in modeled_conditions for sex in ['male', 'female']]
    grid = pd.MultiIndex.from_product(
        [age_range, range(start_year, end_year + 1)],
        names=["Age", "Year"]
    ).to_frame(index=False)
    for col in expected_columns:
        grid[col] = float('nan')

    # Step 2: Load and preprocess raw data
    raw_df = pd.read_csv(raw_csv)
    raw_df = raw_df.rename(columns={'cause': 'condition', 'val': 'prevalence'})

    # Map age group strings → int
    raw_df['age_group'] = raw_df['age'].apply(map_age_group)
    raw_df = raw_df.dropna(subset=['age_group'])  # drop unmatched ages

    # Extract year (start year if in range)
    raw_df['year'] = raw_df['year'].astype(str).str.extract(r'^(\d{4})').astype(float).astype('Int64')

    # Standardize sex
    raw_df['sex'] = raw_df['sex'].str.lower().map({'male': 'male', 'm': 'male', 'female': 'female', 'f': 'female'})

    # Map cause to MIGHTI condition
    raw_df['condition'] = raw_df['condition'].map(cause_map)
    raw_df = raw_df.dropna(subset=['condition', 'year', 'sex', 'prevalence'])

    # Normalize prevalence if needed (assume percentage)
    if raw_df['prevalence'].max() > 1.5:
        raw_df['prevalence'] /= 100.0

    # Step 3: Fill template
    for (cond, sex), group in raw_df.groupby(['condition', 'sex']):
        col = f"{cond}_{sex}"
        if col in grid.columns:
            for _, row in group.iterrows():
                mask = (grid['Age'] == row['age_group']) & (grid['Year'] == row['year'])
                grid.loc[mask, col] = row['prevalence']

    # Step 4: Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    grid.to_csv(output_csv, index=False)
    logging.info(f"✅ Filled prevalence template saved to {output_csv}")

    return grid



def create_condition_metadata_table(long_csv_path, output_csv):


    # Load raw data
    raw = pd.read_csv(long_csv_path)

    # Filter for relevant rows
    raw = raw[(raw["sex"] == "Both") & (raw["age"] == "All ages") & (raw["year"] == 1999)]

    # Pivot the data (use mean in case of duplicates)
    pivot = raw.pivot_table(index="cause", columns="measure", values="val", aggfunc="mean").rename_axis(None, axis=1).reset_index()

    # Confirm required columns exist
    required = ["Deaths", "Prevalence", "Incidence"]
    for col in required:
        if col not in pivot.columns:
            raise ValueError(f"Missing required column: '{col}' in pivoted data.")

    # Compute derived parameters
    pivot["p_death"] = pivot["Deaths"] / 100_000 / (pivot["Prevalence"] / 100)
    pivot["dur_condition"] = pivot["Prevalence"] / pivot["Incidence"]


    pivot["condition"] = pivot["cause"].map(cause_map)
    pivot = pivot.dropna(subset=["condition"])

    # Define fixed metadata for all conditions
    metadata = [
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

    df_meta = pd.DataFrame(metadata, columns=["condition", "disease_type", "disease_class", "affected_sex"])

    # Merge estimates into metadata
    merged = df_meta.merge(pivot[["condition", "p_death", "dur_condition"]], on="condition", how="left")

    # Add empty columns for other parameters
    merged["rel_sus"] = ""
    merged["remission_rate"] = ""
    merged["max_disease_duration"] = ""

    # Reorder columns
    merged = merged[[
        "condition", "p_death", "dur_condition", "rel_sus", "remission_rate",
        "max_disease_duration", "disease_type", "disease_class", "affected_sex"
    ]]

    # Round numeric columns
    merged["p_death"] = pd.to_numeric(merged["p_death"], errors="coerce").round(5)
    merged["dur_condition"] = pd.to_numeric(merged["dur_condition"], errors="coerce").round(2)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged.to_csv(output_csv, index=False)
    logging.info(f"✅ Condition metadata table saved to {output_csv}")

    return merged



def extract_prevalence_timeseries_by_sex(long_csv_path, output_csv):
    import pandas as pd
    import os

    # Load raw file
    df = pd.read_csv(long_csv_path)

    # Filter
    df = df[(df["measure"] == "Prevalence") & (df["age"] == "All ages")]

    df["condition"] = df["cause"].map(cause_map)
    df = df.dropna(subset=["condition"])

    # Pivot: rows = (year, sex), columns = conditions
    wide = df.pivot_table(
        index=["year", "sex"],
        columns="condition",
        values="val",
        aggfunc="mean"
    ).reset_index()

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    wide.to_csv(output_csv, index=False)
    print(f"✅ Prevalence time series saved to {output_csv}")
    return wide

if __name__ == "__main__":
    region = 'eswatini'
    Region = 'Eswatini'

    # ### Age distribution data ###
    # male_csv = 'population_single_age_male.csv'
    # female_csv = 'population_single_age_female.csv'
    # output_csv = f"../mighti/data/{region}_age_distribution.csv"
    
    # process_population_data(male_csv, female_csv, output_csv, country = Region)
    
    
    # ### Life table ###
    # life_table = extract_life_table_by_country(
    #     'life_table_male_1986_1995.csv', 'life_table_male_1996_2005.csv', 'life_table_male_2006_2015.csv', 'life_table_male_2016_2023.csv',
    #     'life_table_female_1986_1995.csv', 'life_table_female_1996_2005.csv', 'life_table_female_2006_2015.csv', 'life_table_female_2016_2023.csv',
    #     country=Region
    # )
    
    # # Extract and save mx and ex
    # mx_df = extract_indicator_from_life_table(life_table, 'mx', f"../mighti/data/{region}_mx.csv")
    # ex_df = extract_indicator_from_life_table(life_table, 'ex', f"../mighti/data/{region}_ex.csv")
    
    # output_csv_mortality = f"../mighti/data/{region}_mortality_rates.csv"
    # reshape_mx_to_mortality_rates(mx_df, output_csv_mortality)
    
    # csv_path_fertility = f"../mighti/data/{region}_asfr.csv"
    # reshape_fertility_to_asfr(
    #     input_csv="fertility_by_single_age_of_mother.csv",
    #     region_name=Region,
    #     output_csv=csv_path_fertility
    # )

    
    
    prevalence_raw_csv=f"{region}_prevalence_all_ncds.csv"   
    # prevalence_output_csv=f"../mighti/data/{region}_prevalence.csv"
    # create_and_fill_prevalence_template_from_long_format(prevalence_raw_csv,prevalence_output_csv)
    
    # long_csv_path = f"{region}_p_death_estimation_parameters.csv" 
    # output_csv_parameters = f"../mighti/data/{region}_parameters.csv"
    # create_condition_metadata_table(long_csv_path,output_csv_parameters)
    
    output_prevalence_csv = f"../mighti/data/{region}_postprocess_check_prevalence.csv"
    extract_prevalence_timeseries_by_sex(prevalence_raw_csv, output_prevalence_csv)
    
    
