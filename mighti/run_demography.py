import pandas as pd
import starsim as ss
import mighti as mi
import streamlit as st

def run_demography(population_csv, fertility_csv, mortality_rates_csv, life_expectancy_csv, extracted_life_expectancy_csv, region, init_year, end_year, population_size):
    beta = 0.001

    population_csv =  f'app/{region}_age_distribution_{init_year}.csv'
    fertility_csv = f'app/{region}_asfr_{init_year}.csv'
    mortality_rates_csv = f'app/{region}_mortality_rates_{init_year}.csv'



    # Load parameters
    csv_path_params = "mighti/data/eswatini_parameters.csv"
    df = pd.read_csv(csv_path_params)
    df.columns = df.columns.str.strip()
    
    
    # Load data from CSVs
    demographics_data = pd.read_csv(population_csv)
    fertility_data = pd.read_csv(fertility_csv)
    mortality_rates_data = pd.read_csv(mortality_rates_csv)
    # life_expectancy_data = pd.read_csv(life_expectancy_csv)

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
    prevalence_data_df = pd.read_csv("mighti/data/prevalence_data_eswatini.csv")
    prevalence_data, age_bins = mi.initialize_prevalence_data(
        diseases, prevalence_data_df, init_year
    )

    # Define a function for disease-specific prevalence
    def get_prevalence_function(disease):
        return lambda module, sim, size: mi.age_sex_dependent_prevalence(
            disease, prevalence_data, age_bins, sim, size
        )

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
    prevalence_analyzer = mi.PrevalenceAnalyzer(
        prevalence_data=prevalence_data, diseases=healthconditions
    )

     # Extract rows corresponding to the init_year
    print(f'here is mortality: {mortality_rates_csv}')
    mortality_rates = mortality_rates_data[mortality_rates_data['Time'] == init_year]
    
    # Check if the data for init_year exists
    if mortality_rates.empty:
        raise ValueError(f"No mortality rates data found for the initial year {init_year}.")

    # Initialize Death and Fertility modules
    death_rates = {"death_rate": mortality_rates, "rate_units": 1}
    death = mi.Deaths(death_rates)
    fertility_rate = {"fertility_rate": fertility_data}
    pregnancy = mi.Pregnancy(pars=fertility_rate)
    
    # Load the processed population data
    
    # Sum male and female values for each age
    age_data = demographics_data.groupby('age')['value'].sum().reset_index()
    
    # Use the summed age_data for the ss.People initialization
    ppl = ss.People(population_size, age_data=age_data)

    # Create the networks - sexual and maternal
    mf = ss.MFNet(duration=1 / 24, acts=80)
    maternal = ss.MaternalNet()
    networks = [mf, maternal]

    # Initialize disease conditions
    hiv_disease = ss.HIV(
        init_prev=ss.bernoulli(get_prevalence_function("HIV")), beta=beta
    )
    disease_objects = []
    for disease in healthconditions:
        init_prev = ss.bernoulli(get_prevalence_function(disease))
        disease_class = getattr(mi, disease, None)
        if disease_class:
            disease_obj = disease_class(
                csv_path=csv_path_params, pars={"init_prev": init_prev}
            )
            disease_objects.append(disease_obj)

    disease_objects.append(hiv_disease)

    # Initialize interaction objects for HIV-NCD interactions
    ncd_hiv_rel_sus = df.set_index("condition")["rel_sus"].to_dict()
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
        label="Connector Simulation",
    )

    # Run the simulation
    sim.run()

    # Get the modules
    deaths_module = get_deaths_module(sim)
    pregnancy_module = get_pregnancy_module(sim)

    # Extract data for each year
    years = list(range(init_year + 1, end_year))
    simulated_imr = []

    for year in years:
        # Retrieve the number of births and deaths for the year
        births = pregnancy_module.get_births(year)
        infant_deaths = deaths_module.infant_deaths

        # Calculate the IMR for males and females
        imr = (infant_deaths / births) if births > 0 else 0

        # Append the IMR values to the lists
        simulated_imr.append(imr)

    # Store the data in a DataFrame
    simulated_data = pd.DataFrame(
        {
            "Year": years,
            "IMR": simulated_imr,
        }
    )

    # Create results DataFrame
    df_results = mi.create_results_dataframe(sim, init_year, end_year, deaths_module)

    # Calculate metrics
    df_metrics = mi.calculate_metrics(df_results)

    # Create life table
    life_table = mi.create_life_table(df_metrics, year=end_year, n_agents = population_size, max_age=100)

    return sim, df_metrics, life_table

def plot_demography(outcome, df_metrics, mortality_csv, life_table, extracted_life_expectancy_csv, year):
    
    if outcome == "Mortality Rates":
        # Plot mortality rates
        observed_data = pd.read_csv(mortality_csv)
        fig = mi.plot_mortality_rates_comparison_app(df_metrics, observed_data, observed_year=year, year=year)
        st.pyplot(fig)
        
    elif outcome == "Life Expectancy":
        # Plot life expectancy
        observed_data = pd.read_csv(extracted_life_expectancy_csv)
        fig = mi.plot_life_expectancy_app(life_table, observed_data, year=year, max_age=100)
        st.pyplot(fig)
