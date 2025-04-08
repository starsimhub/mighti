import pandas as pd
import starsim as ss
import mighti as mi
import matplotlib.pyplot as plt
import streamlit as st

def run_demography(region, init_year, end_year, population_size):
    beta = 0.001

# 1. call prepare_data and extract date 
# 2. Read in relevant data

    # Parameters
    csv_path_params = 'mighti/data/eswatini_parameters.csv'

    # Load the mortality rates and ensure correct format
    mortality_rates_year = mortality_data

    # Load the age distribution data for the specified year
    age_distribution_year = demographics_data

    # Load parameters
    df = pd.read_csv(csv_path_params)
    df.columns = df.columns.str.strip()

    # Extract all conditions except HIV
    healthconditions = [condition for condition in df.condition if condition != "HIV"]
    diseases = ["HIV"] + healthconditions

    # Filter the DataFrame for disease_class being 'ncd'
    ncd_df = df[df["disease_class"] == "ncd"]

    # Extract disease categories from the filtered DataFrame
    chronic = ncd_df[ncd_df["disease_type"] == "chronic"]["condition"].tolist()
    acute = ncd_df[ncd_df["disease_type"] == "acute"]["condition"].tolist()
    remitting = ncd_df[ncd_df["disease_type"] == "remitting"]["condition"].tolist()

    # Extract communicable diseases with disease_class as 'sis'
    communicable_diseases = df[df["disease_class"] == "sis"]["condition"].tolist()

    # Initialize prevalence data
    prevalence_data_df = prevalence_data
    prevalence_data, age_bins = mi.initialize_prevalence_data(diseases, prevalence_data_df, init_year)

    # Define a function for disease-specific prevalence
    def get_prevalence_function(disease):
        return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

    def get_deaths_module(sim):
        for module in sim.modules:
            if isinstance(module, mi.Deaths):
                return module
        raise ValueError("Deaths module not found in the simulation.")
    
    def get_pregnancy_module(sim):
        for module in sim.modules:
            if isinstance(module, mi.Pregnancy):
                return module
        raise ValueError("Pregnancy module not found in the simulation.")

    # Initialize the PrevalenceAnalyzer
    prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=healthconditions)

    death_rates = {'death_rate': mortality_data, 'rate_units': 1}
    death = mi.Deaths(death_rates)  # Use Demographics class implemented in mighti
    fertility_rate = {'fertility_rate': fertility_data}
    pregnancy = mi.Pregnancy(pars=fertility_rate)

    ppl = ss.People(population_size, age_data=demographics_data)

    # Initialize networks
    mf = ss.MFNet(duration=1/24, acts=80)
    maternal = ss.MaternalNet()
    networks = [mf, maternal]

    # Initialize disease conditions
    hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
    disease_objects = []
    for disease in healthconditions:
        init_prev = ss.bernoulli(get_prevalence_function(disease))
        disease_class = getattr(mi, disease, None)
        if disease_class:
            disease_obj = disease_class(csv_path=csv_path_params, pars={"init_prev": init_prev})
            disease_objects.append(disease_obj)

    disease_objects.append(hiv_disease)

    # Initialize interaction objects for HIV-NCD interactions
    ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
    ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
    interactions = [ncd_hiv_connector]

    # Load NCD-NCD interactions
    ncd_interactions = mi.read_interactions("mighti/data/rel_sus.csv")
    connectors = mi.create_connectors(ncd_interactions)

    # Add NCD-NCD connectors to interactions
    interactions.extend(connectors)

    # Initialize the simulation with connectors
    sim = ss.Sim(
        n_agents=population_size,
        networks=networks,
        diseases=disease_objects,
        analyzers=[prevalence_analyzer],
        start=init_year,
        stop=end_year,
        people=ppl,
        demographics=[pregnancy, death],
        connectors=interactions,
        copy_inputs=False,
        label='Connector'
    )

    # Run the simulation
    sim.run()

    # Get the modules
    deaths_module = get_deaths_module(sim)
    pregnancy_module = get_pregnancy_module(sim)

    # Initialize lists to store yearly data
    years = list(range(init_year+1, end_year))

       # Create results DataFrame
    df_results = mi.create_results_dataframe(sim, init_year, end_year, deaths_module)

    # Calculate metrics
    df_metrics = mi.calculate_metrics(df_results)

    # Plot the mortality rates comparison
    mi.plot_mortality_rates_comparison(df_metrics, mortality_data, observed_year=end_year, year=end_year)

    # Create life table
    life_table = mi.create_life_table(df_metrics, year=end_year, max_age=100)

    # Load observed life expectancy data
    observed_LE = demographics_data

    # Plot life expectancy
    mi.plot_life_expectancy(life_table, observed_LE, year=end_year, max_age=100, figsize=(14, 10), title=None)

    return sim, df_metrics, life_table

def plot_demography(outcome, df_metrics, life_table, df):
    if outcome == "Mortality Rates":
        fig = mi.plot_mortality_rates_comparison(df_metrics, df, observed_year=2011, year=2011)
        st.pyplot(fig)
    elif outcome == "Life Expectancy":
        fig = mi.plot_life_expectancy(life_table, df, year=2023, max_age=100, figsize=(14, 10), title=None)   
        st.pyplot(fig)

   