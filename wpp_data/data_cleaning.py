import os
import mighti  # ensures we know the installed package path
import pandas as pd
import logging
import re
import numpy as np

logging.basicConfig(level=logging.INFO)


# -------------------------------------------------------------------------
# Define base path for data output — always points to active MIGHTI install
# -------------------------------------------------------------------------
MIGHTI_BASE = os.path.dirname(mighti.__file__)
DATA_DIR = os.path.join(MIGHTI_BASE, "data")

# Create the folder if it doesn’t exist
os.makedirs(DATA_DIR, exist_ok=True)

def data_path(filename):
    """Return full path for a file inside mighti/data/"""
    return os.path.join(DATA_DIR, filename)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WPP_DATA = os.path.join(PROJECT_ROOT, "wpp_data")

def wpp_path(filename):
    """Return full path for files in wpp_data directory."""
    return os.path.join(WPP_DATA, filename)

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
    "Parkinson’s disease": "ParkinsonsDisease",
    "Neonatal encephalopathy due to birth asphyxia and trauma": "NeonatalEncephalopathy",
    "Neonatal preterm birth": "NeonatalPretermBirth",
    "Neonatal sepsis and other neonatal infections": "NeonatalSepsis",
    "Neural tube defects": "NeuralTubeDefects",
    "Congenital heart anomalies": "CongenitalHeartAnomalies",
    "Congenital musculoskeletal and limb anomalies": "CongenitalMusculoskeletal",
    "Digestive congenital anomalies": "DigestiveCongenitalAnomalies",
    "Down syndrome": "DownSyndrome",
    "Other chromosomal abnormalities": "ChromosomalAbnormalities",
    "Diarrheal disease": "DiarrhealDisease",
    "Esophageal cancer": "EsophagealCancer",
    "Protein-energy malnutrition": "ProteinEnergyMalnutrition",
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


def create_and_fill_prevalence_template_from_long_format(
    raw_csv,
    output_csv,
    start_year=1987,
    end_year=2021,
    age_starts=None
):
    """
    Create and fill a prevalence template with numeric Age values (0, 5, 10, 15...).
    """

    if age_starts is None:
        # lower bounds for age groups, consistent with GBD
        age_starts = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

    # Step 1 — make the base grid
    modeled_conditions = list(set(cause_map.values()))
    expected_cols = [f"{cond}_{sex}" for cond in modeled_conditions for sex in ["male", "female"]]
    grid = pd.MultiIndex.from_product(
        [age_starts, range(start_year, end_year + 1)],
        names=["Age", "Year"]
    ).to_frame(index=False)
    for c in expected_cols:
        grid[c] = np.nan

    # Step 2 — load raw prevalence file
    df = pd.read_csv(raw_csv).rename(columns={"cause": "condition", "val": "prevalence"})
    df["condition"] = df["condition"].map(cause_map)
    df = df.dropna(subset=["condition"])

    # normalize sex labels
    df["sex"] = df["sex"].str.lower().map(
        {"male": "male", "m": "male", "female": "female", "f": "female"}
    )

    # extract numeric year
    df["year"] = df["year"].astype(str).str.extract(r"(\d{4})").astype(float).astype("Int64")

    # normalize prevalence (% → fraction)
    if df["prevalence"].max() > 1.5:
        df["prevalence"] /= 100.0

    # Step 3 — map GBD-style age labels to numeric start
    def get_age_start(label):
        if isinstance(label, str):
            label = label.strip()
            if re.match(r"^<\s*5", label):
                return 0
            m = re.match(r"^(\d+)", label)
            if m:
                return int(m.group(1))
        return np.nan

    df["Age"] = df["age"].apply(get_age_start)
    df = df.dropna(subset=["Age"])

    # Step 4 — fill grid
    for (cond, sex), group in df.groupby(["condition", "sex"]):
        col = f"{cond}_{sex}"
        if col in grid.columns:
            for _, r in group.iterrows():
                mask = (grid["Age"] == r["Age"]) & (grid["Year"] == r["year"])
                grid.loc[mask, col] = r["prevalence"]

    # Step 5 — save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    grid.to_csv(output_csv, index=False)
    logging.info(f"Filled prevalence template saved to {output_csv}")
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
    logging.info(f" Condition metadata table saved to {output_csv}")

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
    print(f" Prevalence time series saved to {output_csv}")
    return wide

if __name__ == "__main__":
    region = "eswatini"
    Region = "Eswatini"

    print(f"Running WPP + GBD data processing for: {Region}")
    print(f"Saving all outputs under: {DATA_DIR}")

    # -------------------------------------------------------------
    # 1. Age distribution
    # -------------------------------------------------------------
    male_csv = wpp_path("population_single_age_male.csv")
    female_csv = wpp_path("population_single_age_female.csv")
    output_csv_age = data_path(f"{region}_age_distribution.csv")
    # process_population_data(male_csv, female_csv, output_csv_age, country=Region)

    # -------------------------------------------------------------
    # 2. Life table (mortality and life expectancy)
    # -------------------------------------------------------------
    male_life_files = [
        wpp_path("life_table_male_1986_1995.csv"),
        wpp_path("life_table_male_1996_2005.csv"),
        wpp_path("life_table_male_2006_2015.csv"),
        wpp_path("life_table_male_2016_2023.csv"),
    ]
    female_life_files = [
        wpp_path("life_table_female_1986_1995.csv"),
        wpp_path("life_table_female_1996_2005.csv"),
        wpp_path("life_table_female_2006_2015.csv"),
        wpp_path("life_table_female_2016_2023.csv"),
    ]

    # life_table = extract_life_table_by_country(*male_life_files, *female_life_files, country=Region)

    # mx/ex extraction
    # mx_df = extract_indicator_from_life_table(life_table, "mx", data_path(f"{region}_mx.csv"))
    # ex_df = extract_indicator_from_life_table(life_table, "ex", data_path(f"{region}_ex.csv"))

    # mortality rates
    # output_csv_mortality = data_path(f"{region}_mortality_rates.csv")
    # reshape_mx_to_mortality_rates(mx_df, output_csv_mortality)

    # -------------------------------------------------------------
    # 3. Fertility (ASFR)
    # -------------------------------------------------------------
    # fertility_input_csv = wpp_path("fertility_by_single_age_of_mother.csv")
    # fertility_output_csv = data_path(f"{region}_asfr.csv")
    # reshape_fertility_to_asfr(fertility_input_csv, Region, fertility_output_csv)

    # -------------------------------------------------------------
    # 4. Prevalence (GBD long-format to MIGHTI template)
    # -------------------------------------------------------------
    prevalence_raw_csv = wpp_path("eswatini_all_disease.csv")
    prevalence_output_csv = data_path(f"{region}_prevalence.csv")
    create_and_fill_prevalence_template_from_long_format(prevalence_raw_csv, prevalence_output_csv)

    # -------------------------------------------------------------
    # 5. Parameter table (derived from GBD long format)
    # -------------------------------------------------------------
    # long_csv_path = wpp_path(f"{region}_p_death_estimation_parameters.csv")
    # output_csv_parameters = data_path(f"{region}_parameters.csv")
    # create_condition_metadata_table(long_csv_path, output_csv_parameters)

    # -------------------------------------------------------------
    # 6. Post-processing check: prevalence by sex
    # -------------------------------------------------------------
    # output_prevalence_check_csv = data_path(f"{region}_postprocess_check_prevalence.csv")
    # extract_prevalence_timeseries_by_sex(prevalence_raw_csv, output_prevalence_check_csv)
    