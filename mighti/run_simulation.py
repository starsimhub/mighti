import pandas as pd
import starsim as ss
import mighti as mi
import matplotlib.pyplot as plt
import streamlit as st

def run_simulation(prevalence_data, demographics_data, fertility_data, mortality_data, init_year, end_year, population_size):
    beta = 0.001

    # Specify data file paths
    csv_path_params = 'mighti/data/eswatini_parameters.csv'
    csv_path_interactions = "mighti/data/rel_sus.csv"
    
    # Read disease parameter file and interactions file
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
    mi.initialize_conditions(df, chronic, acute, remitting, communicable_diseases)

    # Initialize prevalence data
    prevalence_data, age_bins = mi.initialize_prevalence_data(diseases, prevalence_data, init_year)

    # Define a function for disease-specific prevalence
    def get_prevalence_function(disease):
        return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

    # Initialize the PrevalenceAnalyzer
    prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=healthconditions, communicable_diseases=communicable_diseases)

    # Demographics
    fertility_rates = {'fertility_rate': fertility_data}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': mortality_data, 'rate_units': 1}
    death = ss.Deaths(death_rates)
    ppl = ss.People(population_size, age_data=demographics_data)

    # Networks
    mf = ss.MFNet(duration=1/24, acts=80)
    maternal = ss.MaternalNet()
    networks = [mf, maternal]

    # Disease Conditions
    hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)

    # Automatically create disease objects
    disease_objects = []
    for disease in healthconditions:
        init_prev = ss.bernoulli(get_prevalence_function(disease))
        
        # Dynamically get the disease class from `mi` module
        disease_class = getattr(mi, disease, None)
        
        if disease_class:
            disease_obj = disease_class(init_prev=init_prev)  # Instantiate dynamically
            disease_objects.append(disease_obj)
        else:
            print(f"[WARNING] {disease} is not found in `mighti` module. Skipping.")
            
    # Combine all disease objects including HIV
    disease_objects.append(hiv_disease)

    # Disease Interactions
    # Initialize interaction objects for HIV-NCD interactions
    ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
    ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
    interactions = [ncd_hiv_connector]

    # Load NCD-NCD interactions
    ncd_interactions = mi.read_interactions(csv_path_interactions)  # Reads rel_sus.csv
    connectors = mi.create_connectors(ncd_interactions, communicable_diseases)

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
        fig = mi.plot_age_dependent_prevalence(sim, prevalence_analyzer, disease, age_bins)
        st.pyplot(fig)
    elif outcome == "Sex-Dependent Prevalence":
        st.write("Sex-dependent prevalence outcome is not implemented yet.", key="example_key", label_visibility="collapsed")
