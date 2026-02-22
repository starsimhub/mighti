import os
import logging
import re

import numpy as np
import pandas as pd

from .paths import cause_map


logger = logging.getLogger(__name__)


MIGHTI_PARAMETERS_COLUMNS = [
    "condition",
    "p_death",
    "dur_condition",
    "rel_sus",
    "remission_rate",
    "max_disease_duration",
    "p_acquire",
    "disease_type",
    "affected_sex",
]


MIGHTI_CONDITION_METADATA = [
    # Core NCDs + risks
    ("Type1Diabetes", "chronic", "both"),
    ("Type2Diabetes", "remitting", "both"),
    ("Hypertension", "chronic", "both"),
    ("Obesity", "chronic", "both"),
    ("CardiovascularDiseases", "chronic", "both"),
    ("ChronicKidneyDisease", "chronic", "both"),
    ("Hyperlipidemia", "chronic", "both"),
    ("CervicalCancer", "chronic", "female"),
    ("ColorectalCancer", "chronic", "both"),
    ("BreastCancer", "chronic", "both"),
    ("LungCancer", "chronic", "both"),
    ("ProstateCancer", "chronic", "male"),
    ("AlcoholUseDisorder", "remitting", "both"),
    ("TobaccoUse", "remitting", "both"),
    ("Dementia", "chronic", "both"),
    ("PTSD", "chronic", "both"),
    ("MajorDepressiveDisorder", "remitting", "both"),
    ("BipolarDisorder", "remitting", "both"),
    ("ViralHepatitis", "chronic", "both"),
    ("ChronicLiverDisease", "chronic", "both"),
    ("Asthma", "chronic", "both"),
    ("COPD", "chronic", "both"),
    ("AlzheimersDisease", "chronic", "both"),
    ("ParkinsonsDisease", "chronic", "both"),
    ("DrugUseDisorder", "remitting", "both"),
    ("OpioidUseDisorder", "remitting", "both"),
    ("AnxietyDisorder", "remitting", "both"),
    ("ChronicPain", "remitting", "both"),

    # Infectious / SIS-SIR placeholders used by MIGHTI
    ("HPV", "genericsis", "both"),
    ("Flu", "genericsir", "both"),
    ("DiarrhealDisease", "genericsir", "both"),

    # Injuries / violence
    ("InterpersonalViolence", "surgical", "both"),
    ("RoadInjuries", "surgical", "both"),
    ("SelfHarm", "acute", "both"),

    # Neonatal + congenital
    ("NeonatalEncephalopathy", "acute", "both"),
    ("NeonatalPretermBirth", "acute", "both"),
    ("NeonatalSepsis", "acute", "both"),
    ("NeonatalJaundice", "acute", "both"),
    ("CongenitalHeartAnomalies", "surgical", "both"),
    ("CongenitalMusculoskeletal", "surgical", "both"),
    ("DigestiveCongenitalAnomalies", "surgical", "both"),
    ("DownSyndrome", "static", "both"),
    ("NeuralTubeDefects", "surgical", "both"),
    ("ChromosomalAbnormalities", "static", "both"),
    ("EsophagealCancer", "chronic", "both"),
    ("ProteinEnergyMalnutrition", "acute", "both"),

    # New Eswatini mortality-driven additions
    ("COVID19", "genericsis", "both"),
    ("LowerRespiratoryInfections", "genericsis", "both"),
    ("Tuberculosis", "genericsir", "both"),
    ("MaternalConditions", "acute", "female"),
]


def create_mighti_parameters_template(output_csv: str, *, overwrite: bool = False) -> pd.DataFrame:
    """
    Create a template parameter CSV in the same schema as `mighti/data/eswatini_parameters.csv`.

    This is useful when you *do not* have a long-format file containing Deaths/Incidence
    to estimate `p_death` and `dur_condition`, but still want a complete parameter table
    for all modeled conditions (with blanks that can be filled later).
    """
    if (not overwrite) and os.path.exists(output_csv):
        logger.info(f"Parameter file already exists; keeping existing file: {output_csv}")
        return pd.read_csv(output_csv)

    df = pd.DataFrame(MIGHTI_CONDITION_METADATA, columns=["condition", "disease_type", "affected_sex"])

    # Fill required columns (leave numeric ones blank)
    for col in ["p_death", "dur_condition", "rel_sus", "remission_rate", "max_disease_duration", "p_acquire"]:
        df[col] = ""

    df = df[MIGHTI_PARAMETERS_COLUMNS]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"MIGHTI parameter template saved to {output_csv}")
    return df


def ensure_conditions_in_parameters_csv(
    parameters_csv_path: str,
    *,
    output_csv_path: str | None = None,
    overwrite: bool = True,
    create_if_missing: bool = True,
) -> pd.DataFrame:
    """
    Ensure `parameters_csv_path` contains at least the conditions in `MIGHTI_CONDITION_METADATA`.

    - Adds missing columns needed by MIGHTI
    - Appends missing conditions with blank numeric parameters
    - Preserves existing values for existing conditions
    """
    out = output_csv_path or parameters_csv_path

    if not os.path.exists(parameters_csv_path):
        if not create_if_missing:
            raise FileNotFoundError(parameters_csv_path)
        return create_mighti_parameters_template(out, overwrite=overwrite)

    df = pd.read_csv(parameters_csv_path)

    # Ensure required columns exist
    for col in MIGHTI_PARAMETERS_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Standardize condition key
    df["condition"] = df["condition"].astype(str).str.strip()

    # Append any missing conditions
    existing = set(df["condition"].dropna())
    rows = []
    for condition, disease_type, affected_sex in MIGHTI_CONDITION_METADATA:
        if condition in existing:
            continue
        row = {c: "" for c in MIGHTI_PARAMETERS_COLUMNS}
        row["condition"] = condition
        row["disease_type"] = disease_type
        row["affected_sex"] = affected_sex
        rows.append(row)

    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    # Reorder and write
    df = df[MIGHTI_PARAMETERS_COLUMNS]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if overwrite or (not os.path.exists(out)):
        df.to_csv(out, index=False)
        logger.info(f"Updated parameter table saved to {out}")
    else:
        logger.info(f"Not overwriting existing file: {out}")
    return df


def fill_p_death_and_duration_from_allcause_files(
    parameters_csv_path: str,
    *,
    allcause_percent_csv_path: str,
    allcause_rate_csv_path: str,
    year: int = 2007,
    output_csv_path: str | None = None,
    overwrite_existing: bool = False,
) -> pd.DataFrame:
    """
    Fill `p_death` and `dur_condition` in a MIGHTI parameters CSV using GBD-style
    all-cause exports.

    - `allcause_percent_csv_path`: prevalence in "Percent" metric (GBD export);
      values are typically already in fractional form (e.g. 0.012 = 1.2%).
    - `allcause_rate_csv_path`: Deaths/Prevalence/Incidence in "Rate" metric
      (per 100k).

    Computations (matching `data_cleaning_rev`):
      - p_death = deaths_per_100k / 100_000 / prevalence_fraction
      - dur_condition = prevalence_rate_per_100k / incidence_rate_per_100k

    Notes:
    - Only conditions present in the allcause files (after `cause_map`) can be filled.
    - If multiple raw causes map to the same modeled condition (many→one), we SUM them
      before computing.
    """
    # Keep a single implementation of the computation in `fill_p_death_and_duration_from_gbd_long_files()`
    # to avoid drift between multiple versions of the same logic.
    return fill_p_death_and_duration_from_gbd_long_files(
        parameters_csv_path=parameters_csv_path,
        gbd_long_csv_paths=[allcause_rate_csv_path, allcause_percent_csv_path],
        year=year,
        output_csv_path=output_csv_path,
        overwrite_existing=overwrite_existing,
    )


def fill_p_death_and_duration_from_gbd_long_files(
    parameters_csv_path: str,
    *,
    gbd_long_csv_paths: list[str],
    year: int = 2007,
    location: str = "Eswatini",
    output_csv_path: str | None = None,
    overwrite_existing: bool = False,
    acute_duration_years_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Fill `p_death` and `dur_condition` using one or more GBD-style long-format exports
    that contain rows with columns like:

      measure, location, sex, age, cause, metric, year, val

    Some exports may include extra leading columns like `population_group`; these are ignored.

    Requirements for calculations:
    - Prevalence fraction: measure='Prevalence', metric='Percent' (typically already fractional)
    - Rates per 100k: metric='Rate' for measures Deaths/Prevalence/Incidence

    Computations:
      - dur_condition = prevalence_rate_per_100k / incidence_rate_per_100k

    p_death computation depends on disease type:
      - For acute/event-like conditions we prefer a CFR-style proxy:
          p_death = deaths_rate_per_100k / incidence_rate_per_100k
        This is typically much more stable than using point-prevalence, which can be tiny.
        We apply this when the condition's `disease_type` in the parameters table is one of:
          {"acute", "surgical", "genericsis", "genericsir", "sis", "sir"}
        (and when incidence is available and >0).

      - Otherwise (default, prevalence-based proxy):
          p_death = deaths_rate_per_100k / 100_000 / prevalence_fraction

    Note: `acute_duration_years_threshold` is kept for backward-compatibility/experimentation,
    but the primary switch is now `disease_type`.
    """
    out = output_csv_path or parameters_csv_path

    params = pd.read_csv(parameters_csv_path)
    if "condition" not in params.columns:
        raise ValueError(f"Missing 'condition' column in {parameters_csv_path}")

    for col in ["p_death", "dur_condition"]:
        if col not in params.columns:
            params[col] = np.nan

    params["condition"] = params["condition"].astype(str).str.strip()

    frames = []
    for p in gbd_long_csv_paths:
        df = pd.read_csv(p)
        # Allow extra columns; keep only what we need if present
        needed = ["measure", "location", "sex", "age", "cause", "metric", "year", "val"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{p} is missing required columns: {missing}")
        frames.append(df[needed].copy())
    raw = pd.concat(frames, ignore_index=True)

    # Basic filters
    raw = raw[
        (raw["location"] == location)
        & (raw["sex"] == "Both")
        & (raw["age"] == "All ages")
        & (raw["year"] == year)
    ].copy()

    raw["condition"] = raw["cause"].map(cause_map)
    raw = raw.dropna(subset=["condition"])

    # Determine which modeled conditions should use CFR for p_death (Deaths/Incidence).
    # We infer this from the parameter table's disease_type column.
    cfr_types = {"acute", "surgical", "genericsis", "genericsir", "sis", "sir"}
    cfr_conditions: set[str] = set()
    if "disease_type" in params.columns:
        cfr_conditions = set(
            params.loc[params["disease_type"].astype(str).str.lower().isin(cfr_types), "condition"]
            .astype(str)
            .str.strip()
        )

    # Prevalence fraction (Percent metric) and rates (Rate metric) — sum many→one mappings
    prev_frac = (
        raw[(raw["measure"] == "Prevalence") & (raw["metric"] == "Percent")]
        .groupby("condition", as_index=True)["val"]
        .sum()
    )

    rates = raw[raw["metric"] == "Rate"].copy()
    deaths_rate = rates[rates["measure"] == "Deaths"].groupby("condition", as_index=True)["val"].sum()
    prev_rate = rates[rates["measure"] == "Prevalence"].groupby("condition", as_index=True)["val"].sum()
    inc_rate = rates[rates["measure"] == "Incidence"].groupby("condition", as_index=True)["val"].sum()

    idx = sorted(set(prev_frac.index) | set(deaths_rate.index) | set(prev_rate.index) | set(inc_rate.index))
    computed = pd.DataFrame(index=idx)
    computed.index.name = "condition"
    computed["prevalence_fraction"] = prev_frac.reindex(idx)
    computed["deaths_per_100k"] = deaths_rate.reindex(idx)
    computed["prevalence_rate_per_100k"] = prev_rate.reindex(idx)
    computed["incidence_rate_per_100k"] = inc_rate.reindex(idx)

    computed["dur_condition"] = np.nan
    ok2 = (
        computed["prevalence_rate_per_100k"].notna()
        & computed["incidence_rate_per_100k"].notna()
        & (computed["incidence_rate_per_100k"] > 0)
    )
    computed.loc[ok2, "dur_condition"] = computed.loc[ok2, "prevalence_rate_per_100k"] / computed.loc[ok2, "incidence_rate_per_100k"]

    # Default p_death (prevalence-based proxy)
    computed["p_death"] = np.nan
    ok_prev = (
        computed["deaths_per_100k"].notna()
        & computed["prevalence_fraction"].notna()
        & (computed["prevalence_fraction"] > 0)
    )
    computed.loc[ok_prev, "p_death"] = (
        computed.loc[ok_prev, "deaths_per_100k"] / 100_000.0 / computed.loc[ok_prev, "prevalence_fraction"]
    )

    # For CFR-eligible conditions, overwrite p_death with CFR proxy (deaths/incidence)
    cfr_applied_conditions: set[str] = set()
    if cfr_conditions:
        is_cfr = computed.index.to_series().isin(cfr_conditions)
        ok_cfr = (
            is_cfr
            & computed["deaths_per_100k"].notna()
            & computed["incidence_rate_per_100k"].notna()
            & (computed["incidence_rate_per_100k"] > 0)
        )
        computed.loc[ok_cfr, "p_death"] = (
            computed.loc[ok_cfr, "deaths_per_100k"] / computed.loc[ok_cfr, "incidence_rate_per_100k"]
        )
        cfr_applied_conditions = set(computed.index[ok_cfr])

    # Keep p_death in [0, 1] when interpreted as a probability
    computed["p_death"] = computed["p_death"].clip(lower=0, upper=1)

    comp_short = computed[["p_death", "dur_condition"]].reset_index()
    merged = params.merge(comp_short, on="condition", how="left", suffixes=("", "_calc"))

    for col in ["p_death", "dur_condition"]:
        calc = f"{col}_calc"
        if col == "p_death" and cfr_applied_conditions:
            # Always overwrite p_death for CFR-eligible conditions if we computed it
            force = merged["condition"].isin(cfr_applied_conditions) & merged[calc].notna()
            merged.loc[force, col] = merged.loc[force, calc]
        if overwrite_existing:
            merged[col] = merged[calc].where(merged[calc].notna(), merged[col])
        else:
            missing = merged[col].isna()
            merged.loc[missing, col] = merged.loc[missing, calc]
        merged = merged.drop(columns=[calc])

    os.makedirs(os.path.dirname(out), exist_ok=True)
    merged.to_csv(out, index=False)
    logger.info(f"Filled p_death/dur_condition saved to {out} (year={year})")
    return merged


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

    # Step 3.5 — aggregate collisions from cause_map many→one mappings
    # e.g. Hepatitis B/C → ViralHepatitis, multiple maternal sub-causes → MaternalConditions
    df = (
        df.groupby(["condition", "sex", "Age", "year"], as_index=False)["prevalence"]
        .sum()
    )

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

    # Define fixed metadata for all conditions (schema aligned with mighti/data/*_parameters.csv)
    df_meta = pd.DataFrame(MIGHTI_CONDITION_METADATA, columns=["condition", "disease_type", "affected_sex"])

    # Merge estimates into metadata
    merged = df_meta.merge(pivot[["condition", "p_death", "dur_condition"]], on="condition", how="left")

    # Add empty columns for other parameters
    merged["rel_sus"] = ""
    merged["remission_rate"] = ""
    merged["max_disease_duration"] = ""
    merged["p_acquire"] = ""

    # Reorder columns
    merged = merged[MIGHTI_PARAMETERS_COLUMNS]

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

