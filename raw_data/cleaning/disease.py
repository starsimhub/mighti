import os
import logging
import re

import numpy as np
import pandas as pd

from .paths import cause_map


logger = logging.getLogger(__name__)


def create_and_fill_prevalence_template_from_long_format(
    raw_csv,
    output_csv,
    start_year=1987,
    end_year=2021,
    age_starts=None,
    *,
    overwrite: bool = False,
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
    grid = pd.MultiIndex.from_product([age_starts, range(start_year, end_year + 1)], names=["Age", "Year"]).to_frame(
        index=False
    )
    for c in expected_cols:
        grid[c] = np.nan

    # Step 2 — load raw prevalence file
    df = pd.read_csv(raw_csv).rename(columns={"cause": "condition", "val": "prevalence"})
    df["condition"] = df["condition"].map(cause_map)
    df = df.dropna(subset=["condition"])

    # normalize sex labels
    df["sex"] = df["sex"].str.lower().map({"male": "male", "m": "male", "female": "female", "f": "female"})

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

    # Step 5 — save (only if missing unless overwrite=True)
    if (not overwrite) and os.path.exists(output_csv):
        logger.info(f"Prevalence file already exists; keeping existing file: {output_csv}")
        return grid

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    grid.to_csv(output_csv, index=False)
    logger.info(f"Filled prevalence template saved to {output_csv}")
    return grid


def create_condition_metadata_table(long_csv_path, output_csv):
    # Load raw data
    raw = pd.read_csv(long_csv_path)

    # Filter for relevant rows
    raw = raw[(raw["sex"] == "Both") & (raw["age"] == "All ages") & (raw["year"] == 1999)]

    # Pivot the data (use mean in case of duplicates)
    pivot = (
        raw.pivot_table(index="cause", columns="measure", values="val", aggfunc="mean")
        .rename_axis(None, axis=1)
        .reset_index()
    )

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
        ("BipolarDisorder", "remitting", "ncd", "both"),
        ("HPV", "remitting", "sis", "both"),
        ("Flu", "chronic", "sis", "both"),
        ("ViralHepatitis", "chronic", "ncd", "both"),
        ("InterpersonalViolence", "acute", "ncd", "both"),
        ("RoadInjuries", "acute", "ncd", "both"),
        ("SelfHarm", "acute", "ncd", "both"),
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
    merged = merged[
        [
            "condition",
            "p_death",
            "dur_condition",
            "rel_sus",
            "remission_rate",
            "max_disease_duration",
            "disease_type",
            "disease_class",
            "affected_sex",
        ]
    ]

    # Round numeric columns
    merged["p_death"] = pd.to_numeric(merged["p_death"], errors="coerce").round(5)
    merged["dur_condition"] = pd.to_numeric(merged["dur_condition"], errors="coerce").round(2)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged.to_csv(output_csv, index=False)
    logger.info(f"Condition metadata table saved to {output_csv}")

    return merged


def extract_prevalence_timeseries_by_sex(long_csv_path, output_csv):
    # Load raw file
    df = pd.read_csv(long_csv_path)

    # Filter
    df = df[(df["measure"] == "Prevalence") & (df["age"] == "All ages")]

    df["condition"] = df["cause"].map(cause_map)
    df = df.dropna(subset=["condition"])

    # Pivot: rows = (year, sex), columns = conditions
    wide = df.pivot_table(index=["year", "sex"], columns="condition", values="val", aggfunc="mean").reset_index()

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    wide.to_csv(output_csv, index=False)
    logger.info(f"Prevalence time series saved to {output_csv}")
    return wide

