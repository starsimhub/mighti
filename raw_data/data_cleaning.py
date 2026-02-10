"""
Single entrypoint for generating `mighti/data/` inputs from `raw_data/`.

Keep the actual transformation logic in `raw_data/cleaning/*` and keep this file
as a thin runner so there is one canonical data cleaning script.
"""

import logging
import pandas as pd

from cleaning.paths import data_path, disease_data_path, ensure_data_dir
from cleaning.disease import (
    create_and_fill_prevalence_template_from_long_format,
    ensure_conditions_in_parameters_csv,
    fill_p_death_and_duration_from_gbd_long_files,
)


logging.basicConfig(level=logging.INFO)


def run_eswatini_pipeline(*, year_for_parameters: int = 2007, overwrite_outputs: bool = True) -> None:
    region = "eswatini"
    Region = "Eswatini"
    data_dir = ensure_data_dir()

    print(f"Running GBD data processing for: {Region}")
    print(f"Saving outputs under: {data_dir}")

    # -------------------------------------------------------------
    # 1) Prevalence (GBD long-format by age/sex → MIGHTI prevalence grid)
    # -------------------------------------------------------------
    prevalence_raw_csv = disease_data_path(f"{region}_disease_prevalence_agesex.csv")
    prevalence_output_csv = data_path(f"{region}_prevalence.csv")
    create_and_fill_prevalence_template_from_long_format(
        prevalence_raw_csv,
        prevalence_output_csv,
        start_year=2007,
        end_year=2021,
        overwrite=overwrite_outputs,
    )

    # -------------------------------------------------------------
    # 2) Disease parameter table (p_death, dur_condition) from all-cause exports
    # -------------------------------------------------------------
    parameters_csv = data_path(f"{region}_parameters.csv")

    # Ensure the parameter table has all expected conditions/columns
    ensure_conditions_in_parameters_csv(parameters_csv, overwrite=True)

    # Fill p_death and dur_condition for any causes present in these all-cause exports
    allcause_percent = disease_data_path("allcause_percent.csv")
    allcause_rate = disease_data_path("allcause_rate.csv")

    filled = fill_p_death_and_duration_from_gbd_long_files(
        parameters_csv_path=parameters_csv,
        gbd_long_csv_paths=[allcause_rate, allcause_percent],
        year=year_for_parameters,
        location=Region,
        output_csv_path=parameters_csv,
        overwrite_existing=False,
    )

    # Quick summary
    p_death_n = pd.to_numeric(filled["p_death"], errors="coerce").notna().sum()
    dur_n = pd.to_numeric(filled["dur_condition"], errors="coerce").notna().sum()
    print(f"Filled p_death for {p_death_n} conditions; dur_condition for {dur_n} conditions (year={year_for_parameters})")
    return


if __name__ == "__main__":
    run_eswatini_pipeline()