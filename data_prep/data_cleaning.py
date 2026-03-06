"""
Single entrypoint for generating `data/processed/` inputs from `data/raw/`.

Keep the actual transformation logic in `data_prep/cleaning/*` and keep this file
as a thin runner so there is one canonical data cleaning script.
"""

import os
import logging
import argparse
import pandas as pd

from cleaning.paths import data_path, disease_data_path, ensure_data_dir, wpp_path
from cleaning.disease import (
    create_and_fill_prevalence_template_from_long_format,
    ensure_conditions_in_parameters_csv,
    fill_p_death_and_duration_from_gbd_long_files,
    merge_gbd_long_csvs,
)
from cleaning.demography import (
    extract_life_table_by_country,
    extract_indicator_from_life_table,
    process_population_data,
    reshape_fertility_to_asfr,
    reshape_mx_to_mortality_rates,
)


logging.basicConfig(level=logging.INFO)


def run_eswatini_pipeline(
    *,
    year_for_parameters=2007,
    overwrite_outputs=False,
    overwrite_existing_parameters=False,
    merge_split_prevalence_exports=False,
):
    """Backward-compatible wrapper for Eswatini."""
    return run_pipeline(
        region="eswatini",
        location_name="Eswatini",
        baseline_year=year_for_parameters,
        overwrite_outputs=overwrite_outputs,
        overwrite_existing_parameters=overwrite_existing_parameters,
        merge_split_prevalence_exports=merge_split_prevalence_exports,
    )


def run_pipeline(
    *,
    region,
    location_name,
    baseline_year=2007,
    overwrite_outputs=False,
    overwrite_existing_parameters=False,
    merge_split_prevalence_exports=False,
):
    data_dir = ensure_data_dir()

    print(f"Running GBD/WPP data processing for: {location_name}")
    print(f"Saving outputs under: {data_dir}")

    # -------------------------------------------------------------
    # 0) Demography (WPP → MIGHTI)
    # -------------------------------------------------------------
    mx_csv = data_path(f"{region}_mx.csv")
    ex_csv = data_path(f"{region}_ex.csv")
    mortality_rates_csv = data_path(f"{region}_mortality_rates.csv")
    mortality_rates_year_csv = data_path(f"{region}_mortality_rates_{baseline_year}.csv")
    asfr_csv = data_path(f"{region}_asfr.csv")
    age_dist_csv = data_path(f"{region}_age_distribution.csv")
    age_dist_year_csv = data_path(f"{region}_age_distribution_{baseline_year}.csv")

    # Life table → mx/ex (wide)
    if overwrite_outputs or (not os.path.exists(mx_csv)) or (not os.path.exists(ex_csv)):
        life_table = extract_life_table_by_country(
            male_csv1=wpp_path("life_table_male_1986_1995.csv"),
            male_csv2=wpp_path("life_table_male_1996_2005.csv"),
            male_csv3=wpp_path("life_table_male_2006_2015.csv"),
            male_csv4=wpp_path("life_table_male_2016_2023.csv"),
            female_csv1=wpp_path("life_table_female_1986_1995.csv"),
            female_csv2=wpp_path("life_table_female_1996_2005.csv"),
            female_csv3=wpp_path("life_table_female_2006_2015.csv"),
            female_csv4=wpp_path("life_table_female_2016_2023.csv"),
            country=location_name,
        )
        extract_indicator_from_life_table(life_table, "mx", output_csv=mx_csv)
        extract_indicator_from_life_table(life_table, "ex", output_csv=ex_csv)

    # mx (wide) → mortality_rates (long)
    if overwrite_outputs or (not os.path.exists(mortality_rates_csv)):
        mx_df = pd.read_csv(mx_csv)
        reshape_mx_to_mortality_rates(mx_df, mortality_rates_csv)

    # Extract year-specific mortality rates (long)
    if overwrite_outputs or (not os.path.exists(mortality_rates_year_csv)):
        mort = pd.read_csv(mortality_rates_csv)
        mort["Time"] = pd.to_numeric(mort["Time"], errors="coerce")
        mort_year = mort[mort["Time"] == baseline_year].dropna(subset=["mx"]).copy()
        mort_year.to_csv(mortality_rates_year_csv, index=False)

    # Fertility → ASFR (long)
    if overwrite_outputs or (not os.path.exists(asfr_csv)):
        reshape_fertility_to_asfr(
            input_csv=wpp_path("fertility_by_single_age_of_mother.csv"),
            region_name=location_name,
            output_csv=asfr_csv,
        )

    # Population by single age → age distribution (wide; by sex)
    if overwrite_outputs or (not os.path.exists(age_dist_csv)):
        process_population_data(
            male_csv=wpp_path("population_single_age_male.csv"),
            female_csv=wpp_path("population_single_age_female.csv"),
            output_csv=age_dist_csv,
            country=location_name,
        )

    # Extract year-specific age distribution (collapsed across sex)
    if overwrite_outputs or (not os.path.exists(age_dist_year_csv)):
        df_age = pd.read_csv(age_dist_csv)
        ycol = str(baseline_year)
        if ycol not in df_age.columns:
            raise ValueError(f"Year {baseline_year} not found in age distribution file: {age_dist_csv}")
        df_age_year = df_age[["age", "sex", ycol]].groupby("age")[ycol].sum().reset_index()
        df_age_year.columns = ["age", "value"]
        df_age_year.to_csv(age_dist_year_csv, index=False)

    # -------------------------------------------------------------
    # 1) Prevalence (GBD long-format by age/sex → MIGHTI prevalence grid)
    # -------------------------------------------------------------
    prevalence_raw_csv = disease_data_path(f"{region}_disease_prevalence_agesex.csv")
    disease_dir = os.path.dirname(prevalence_raw_csv)
    if merge_split_prevalence_exports:
        extra_prevalence_parts = sorted(
            [
                os.path.join(disease_dir, f)
                for f in os.listdir(disease_dir)
                if f.startswith(f"{region}_alldiseases_") and f.endswith(".csv")
            ]
        )
        # One-off back-compat: include disease-specific prevalence exports (same long schema, sometimes with extra columns).
        disease_specific_prevalence = sorted(
            [
                os.path.join(disease_dir, f)
                for f in os.listdir(disease_dir)
                if f.startswith(f"{region}_")
                and f.endswith(".csv")
                and ("prevalence" in f.lower())
                and (f not in {os.path.basename(prevalence_raw_csv), f"{region}_prevalence_all_ncds.csv"})
            ]
        )
        for fname in [f"{region}_opioidusedisorder.csv"]:
            p = os.path.join(disease_dir, fname)
            if os.path.exists(p) and p not in disease_specific_prevalence:
                disease_specific_prevalence.append(p)

        if extra_prevalence_parts or disease_specific_prevalence:
            merged_raw_csv = os.path.join(disease_dir, f"{region}_disease_prevalence_agesex_multiyear.csv")
            merge_gbd_long_csvs(
                [prevalence_raw_csv, *extra_prevalence_parts, *disease_specific_prevalence], merged_raw_csv
            )
            prevalence_raw_csv = merged_raw_csv

    prevalence_output_csv = data_path(f"{region}_prevalence.csv")
    if overwrite_outputs or (not os.path.exists(prevalence_output_csv)):
        create_and_fill_prevalence_template_from_long_format(
            prevalence_raw_csv,
            prevalence_output_csv,
            start_year=None,
            end_year=None,
            overwrite=True,
        )

    # -------------------------------------------------------------
    # 2) Disease parameter table (p_death, dur_condition) from all-cause exports
    # -------------------------------------------------------------
    parameters_csv = data_path(f"{region}_parameters.csv")

    if (not os.path.exists(parameters_csv)) or overwrite_outputs or overwrite_existing_parameters:
        # Ensure the parameter table has all expected conditions/columns
        ensure_conditions_in_parameters_csv(parameters_csv, overwrite=True, prune_extras=True)

        # Fill p_death and dur_condition for any causes present in these all-cause exports
        allcause_rate = disease_data_path("allcause_rate.csv")

        filled = fill_p_death_and_duration_from_gbd_long_files(
            parameters_csv_path=parameters_csv,
            gbd_long_csv_paths=[allcause_rate],
            year=baseline_year,
            location=location_name,
            output_csv_path=parameters_csv,
            overwrite_existing=overwrite_existing_parameters,
        )

        # Quick summary
        p_death_n = pd.to_numeric(filled["p_death"], errors="coerce").notna().sum()
        dur_n = pd.to_numeric(filled["dur_condition"], errors="coerce").notna().sum()
        print(
            f"Filled p_death for {p_death_n} conditions; dur_condition for {dur_n} conditions (year={baseline_year})"
        )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data/processed inputs from data/raw for a region.")
    parser.add_argument("--region", default="eswatini", help="Region slug used in filenames (e.g., eswatini)")
    parser.add_argument("--location-name", default="Eswatini", help="GBD/WPP location name (e.g., Eswatini)")
    parser.add_argument(
        "--baseline-year",
        type=int,
        default=2007,
        help="Baseline year for parameter derivation and year-specific demography outputs",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data/processed outputs")
    parser.add_argument(
        "--overwrite-existing-parameters",
        action="store_true",
        help="Overwrite existing p_death/dur_condition values in parameters CSV",
    )
    parser.add_argument(
        "--merge-split-prevalence-exports",
        action="store_true",
        help="One-off: merge split prevalence exports into a single multiyear input",
    )
    args = parser.parse_args()

    run_pipeline(
        region=args.region,
        location_name=args.location_name,
        baseline_year=args.baseline_year,
        overwrite_outputs=args.overwrite,
        overwrite_existing_parameters=args.overwrite_existing_parameters,
        merge_split_prevalence_exports=args.merge_split_prevalence_exports,
    )