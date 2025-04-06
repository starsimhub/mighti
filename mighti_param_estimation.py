import starsim as ss
import sciris as sc
import mighti as mi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
# Define population size and simulation timeline
# ---------------------------------------------------------------------
beta = 0.001
n_agents = 5000 # Number of agents in the simulation
inityear = 2007  # Simulation start year
endyear = 2012

# ---------------------------------------------------------------------
# Specify data file paths
# ---------------------------------------------------------------------

# Parameters
csv_path_params = 'mighti/data/eswatini_parameters.csv'

# Relative Risks
csv_path_interactions = "mighti/data/rel_sus_0.csv"

# Prevalence data
csv_prevalence = 'mighti/data/prevalence_data_eswatini.csv'

# Fertility data 
csv_path_fertility = 'mighti/data/eswatini_asfr.csv'

# Death data
csv_path_death = f'mighti/data/eswatini_mortality_rates_{inityear}.csv'

# Age distribution data
csv_path_age = f'mighti/data/eswatini_age_distribution_{inityear}.csv'

import prepare_data_for_year
prepare_data_for_year.prepare_data_for_year(inityear)

# Load the mortality rates and ensure correct format
mortality_rates_year = pd.read_csv(csv_path_death)

# Load the age distribution data for the specified year
age_distribution_year = pd.read_csv(csv_path_age)

# Load parameters
df = pd.read_csv(csv_path_params)
df.columns = df.columns.str.strip()


# Define diseases
# conditions = ncds + communicable_diseases  # List of all diseases

ncd = ['Type2Diabetes', 'ChronicKidneyDisease','CervicalCancer','ProstateCancer'] 
diseases = ['HIV'] + ncd #+conditions # List of diseases including HIV


# Load prevalence data from the CSV file
prevalence_data_csv_path = sc.thispath() / 'mighti/data/prevalence_data_eswatini.csv'
prevalence_data_df = pd.read_csv(prevalence_data_csv_path)

#Check that there are non-zero values for CCa, T2D, CKD
# only prostate cancer data is being plotted for observed scatter point plots
print(prevalence_data_df[['ProstateCancer_male', 'CervicalCancer_male', 'Type2Diabetes_male', 'ChronicKidneyDisease_male']].tail())

# Initialize prevalence data from the DataFrame
prevalence_data, age_bins = mi.initialize_prevalence_data(diseases, prevalence_data=prevalence_data_df, inityear=inityear)

# Define a function for disease-specific prevalence
def get_prevalence_function(disease):
    return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)


death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
death = mi.Deaths(death_rates)  # Use normal StarSim Deaths implementation


fertility_rate = {'fertility_rate': pd.read_csv(csv_path_fertility)}
pregnancy = mi.Pregnancy(pars=fertility_rate)  # Use the correct parameter name

ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))

# Create the networks - sexual and maternal
mf = ss.MFNet(duration=1/24, acts=80)
maternal = ss.MaternalNet()
networks = [mf, maternal]

hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)

# Initialize interaction objects for HIV-NCD interactions
ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
interactions = [ncd_hiv_connector]

if __name__ == '__main__':
    
    hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)

    # Automatically create disease objects for all diseases
    disease_objects = []
    for disease in ncd:
        init_prev = ss.bernoulli(get_prevalence_function(disease))
        
        # Dynamically get the disease class from `mi` module
        disease_class = getattr(mi, disease, None)
        
        if disease_class:
            disease_obj = disease_class(disease_name=disease, csv_path=csv_path_params, pars={"init_prev": init_prev})  # Instantiate dynamically and pass csv_path
            disease_objects.append(disease_obj)
        else:
            print(f"[WARNING] {disease} is not found in `mighti` module. Skipping.")


    # Combine all disease objects including HIV
    disease_objects.append(hiv_disease)
    
    # Initialize interaction objects for HIV-NCD interactions
    ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
    ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
    interactions = [ncd_hiv_connector]
    
    # Load NCD-NCD interactions
    ncd_interactions = mi.read_interactions("mighti/data/rel_sus_0.csv")
    connectors = mi.create_connectors(ncd_interactions)
    
    # Add NCD-NCD connectors to interactions
    interactions.extend(connectors)
     
    # Initialize the simulation with connectors
    sim = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        diseases=disease_objects,
        analyzers=[prevalence_analyzer],
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        connectors=interactions,
        copy_inputs=False,
        label='Connector'
    )

    # Run the simulation
    sim.run()

    
    # Plot the results for each simulation
    mi.plot_mean_prevalence(sim, prevalence_analyzer, 'Type2Diabetes', prevalence_data_df, init_year = inityear, end_year = endyear)  
    mi.plot_mean_prevalence(sim, prevalence_analyzer, 'ChronicKidneyDisease', prevalence_data_df, init_year = inityear, end_year = endyear)
    mi.plot_mean_prevalence(sim, prevalence_analyzer, 'CervicalCancer', prevalence_data_df, init_year = inityear, end_year = endyear)
    mi.plot_mean_prevalence(sim, prevalence_analyzer, 'ProstateCancer', prevalence_data_df, init_year = inityear, end_year = endyear)
   
    # Example usage:
    mi.plot_age_group_prevalence(sim, prevalence_analyzer, 'Type2Diabetes', prevalence_data_df, init_year = inityear, end_year = endyear)    