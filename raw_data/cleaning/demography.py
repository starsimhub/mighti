import os
import logging

import pandas as pd


logger = logging.getLogger(__name__)


def process_population_data(male_csv, female_csv, output_csv, country):
    # Read population data
    male_population = pd.read_csv(male_csv)
    female_population = pd.read_csv(female_csv)

    # Handle non-finite values in the year column
    male_population["year"] = pd.to_numeric(male_population["year"], errors="coerce").fillna(0).astype(int)
    female_population["year"] = pd.to_numeric(female_population["year"], errors="coerce").fillna(0).astype(int)

    # Filter data for the specified country and years
    male_population = male_population[(male_population["region"] == country) & (male_population["year"])]
    female_population = female_population[(female_population["region"] == country) & (female_population["year"])]

    # Extract age-specific data and reorganize
    age_range = range(0, 101)  # Age 0 to 100
    years = sorted(female_population["year"].unique())

    data = {"age": list(age_range) * 2, "sex": ["Male"] * len(age_range) + ["Female"] * len(age_range)}

    for year in years:
        data[str(year)] = (
            list(male_population[male_population["year"] == year].iloc[:, 3:].values.flatten())
            + list(female_population[female_population["year"] == year].iloc[:, 3:].values.flatten())
        )

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Population data saved to {output_csv}")


def extract_life_table_by_country(
    male_csv1,
    male_csv2,
    male_csv3,
    male_csv4,
    female_csv1,
    female_csv2,
    female_csv3,
    female_csv4,
    country,
):
    def load_and_clean(filepath, sex):
        df = pd.read_csv(filepath, low_memory=False)
        df = df[df["region"] == country].copy()
        df["Sex"] = sex
        df = df.rename(columns={"year": "Time", "age": "Age"})
        numeric_cols = ["Time", "Age", "mx", "ex"]  # Add more if needed
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    male = pd.concat(
        [
            load_and_clean(male_csv1, "Male"),
            load_and_clean(male_csv2, "Male"),
            load_and_clean(male_csv3, "Male"),
            load_and_clean(male_csv4, "Male"),
        ]
    )
    female = pd.concat(
        [
            load_and_clean(female_csv1, "Female"),
            load_and_clean(female_csv2, "Female"),
            load_and_clean(female_csv3, "Female"),
            load_and_clean(female_csv4, "Female"),
        ]
    )
    return pd.concat([male, female], ignore_index=True)


def extract_indicator_from_life_table(life_table_df, indicator, output_csv=None):
    df = life_table_df[["Time", "Age", "Sex", indicator]].dropna()
    df[indicator] = pd.to_numeric(df[indicator], errors="coerce")
    result = df.pivot_table(index=["Age", "Sex"], columns="Time", values=indicator).reset_index()

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        result.to_csv(output_csv, index=False)
        logger.info(f"{indicator} saved to {output_csv}")

    return result


def reshape_mx_to_mortality_rates(mx_df, output_csv):
    """
    Reshape the wide-format mx DataFrame into long format with columns:
    AgeGrpStart, Sex, Time, mx
    and save it to output_csv
    """
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
    logger.info(f"Mortality rates saved to {output_csv}")

    return df_long


def reshape_fertility_to_asfr(input_csv, region_name, output_csv):
    """
    Convert wide-format fertility data to long-format ASFR for a specific region.

    Output format: Time, AgeGrp, ASFR
    """
    # Read and filter
    df = pd.read_csv(input_csv)
    df = df[df["region"] == region_name]

    if df.empty:
        raise ValueError(f"No rows found for region: {region_name}")

    # Keep only year and age columns
    id_vars = ["year"]
    value_vars = [str(age) for age in range(15, 50)]  # ASFR is usually defined for 15–49
    df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="AgeGrp", value_name="ASFR")

    # Clean and rename
    df_long = df_long.rename(columns={"year": "Time"})
    df_long["AgeGrp"] = df_long["AgeGrp"].astype(int)
    df_long["ASFR"] = pd.to_numeric(df_long["ASFR"], errors="coerce")
    df_long = df_long.dropna()

    # Sort
    df_long = df_long.sort_values(by=["Time", "AgeGrp"])

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_long.to_csv(output_csv, index=False)
    logger.info(f"Fertility data saved to {output_csv}")

    return df_long

