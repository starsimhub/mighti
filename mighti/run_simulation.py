import pandas as pd
import starsim as ss
import mighti as mi
import matplotlib.pyplot as plt
import streamlit as st

def run_simulation(prevalence_data, demographics_data, fertility_data, mortality_data, init_year, end_year, population_size):
    # Initialize prevalence data
    prevalence_data, age_bins = mi.initialize_prevalence_data(['HIV', 'Type2Diabetes', 'ChronicKidneyDisease'], prevalence_data, init_year)

    # Define a function for disease-specific prevalence
    def get_prevalence_function(disease):
        return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

    # Initialize the PrevalenceAnalyzer
    prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=['HIV', 'Type2Diabetes', 'ChronicKidneyDisease'])

    # Create demographics
    fertility_rates = {'fertility_rate': fertility_data}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': mortality_data, 'rate_units': 1}
    death = ss.Deaths(death_rates)
    ppl = ss.People(population_size, age_data=demographics_data)

    # Create the networks - sexual and maternal
    mf = ss.MFNet(duration=1/24, acts=80)
    maternal = ss.MaternalNet()
    networks = [mf, maternal]

    # Define diseases
    hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=0.0001)  # Example beta value
    ncd = ['Type2Diabetes', 'ChronicKidneyDisease'] 

    # Automatically create disease objects for NCDs
    disease_objects = []
    for disease in ncd:
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
    
    # Initialize interaction objects for HIV-NCD interactions
    interactions = [mi.Type2DiabetesHIVConnector(),
                    mi.CKDHIVConnector()]
    
    # Load NCD-NCD interactions
    ncd_interactions = mi.read_interactions("mighti/data/rel_sus.csv")  # Reads rel_sus.csv
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

def plot_results(sim, prevalence_analyzer, outcome):
    if outcome == "Mean Prevalence":
        fig1 = mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'Type2Diabetes')
        fig2 = mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'ChronicKidneyDisease')
        st.pyplot(fig1)
        st.pyplot(fig2)
    elif outcome == "Population":
        st.write("Population outcome is not implemented yet.")
    elif outcome == "Sex-Dependent Prevalence":
        st.write("Sex-dependent prevalence outcome is not implemented yet.")