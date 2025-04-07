import pandas as pd
import starsim as ss
import mighti as mi
import matplotlib.pyplot as plt
import streamlit as st

def run_simulation(prevalence_data, demographics_data, fertility_data, mortality_data, init_year, end_year, population_size):
    beta = 0.001


    # # Parameters
    csv_path_params = 'mighti/data/eswatini_parameters.csv'
    
    # # Relative Risks
    # csv_path_interactions = "mighti/data/rel_sus.csv"
    
    # # Prevalence data
    # csv_prevalence = 'mighti/data/prevalence_data_eswatini.csv'
    
    # # Fertility data 
    # csv_path_fertility = 'mighti/data/eswatini_asfr.csv'
    
    # # Death data
    # csv_path_death = f'mighti/data/eswatini_mortality_rates_{init_year}.csv'
    
    # # Age distribution data
    # csv_path_age = f'mighti/data/eswatini_age_distribution_{init_year}.csv'
    

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

    # Initialize conditions
    # mi.initialize_conditions(df, chronic, acute, remitting, communicable_diseases)

    # Initialize prevalence data
    prevalence_data_df = prevalence_data
    prevalence_data, age_bins = mi.initialize_prevalence_data(diseases, prevalence_data_df, init_year)

    # Define a function for disease-specific prevalence
    def get_prevalence_function(disease):
        return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

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
    return sim, prevalence_analyzer

def plot_results(sim, prevalence_analyzer, outcome, disease, age_bins):
    if outcome == "Mean Prevalence":
        fig = mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, disease)
        st.pyplot(fig)
    elif outcome == "Age-dependent Prevalence":
        st.write("Age-dependent Prevalence is not implemented yet.")
    elif outcome == "Life Expectancy":
        st.write("Life Expectancy is not implemented yet.")