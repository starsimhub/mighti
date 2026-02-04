import logging

from cleaning.paths import data_path, wpp_path, disease_data_path, ensure_data_dir
from cleaning.demography import (
    process_population_data,
    extract_life_table_by_country,
    extract_indicator_from_life_table,
    reshape_mx_to_mortality_rates,
    reshape_fertility_to_asfr,
)
from cleaning.disease import (
    create_and_fill_prevalence_template_from_long_format,
    create_condition_metadata_table,
    extract_prevalence_timeseries_by_sex,
)


logging.basicConfig(level=logging.INFO)
DATA_DIR = ensure_data_dir()

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
    # GBD-style long-format prevalence lives under raw_data/disease_data/
    prevalence_raw_csv = disease_data_path(f"{region}_disease_prevalence_agesex.csv")
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
    