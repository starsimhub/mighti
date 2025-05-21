import starsim as ss
import sciris as sc
import mighti as mi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

##### TO DO #####
# Use HIV in stisim 
# Incidence rate estimation 
# Relative Risk implementation 


# ---------------------------------------------------------------------
# Define population size and simulation timeline
# ---------------------------------------------------------------------
beta = 0.001
n_agents = 10_000 # Number of agents in the simulation
inityear = 2007  # Simulation start year
endyear = 2030

# ---------------------------------------------------------------------
# Specify data file paths
# ---------------------------------------------------------------------


# Parameters
csv_path_params = 'mighti/data/eswatini_parameters_gbd.csv'

# Relative Risks
csv_path_interactions = "mighti/data/rel_sus.csv"

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

# Extract all conditions except HIV
# healthconditions = [condition for condition in df.condition if condition != "HIV"]
healthconditions = [condition for condition in df.condition if condition not in ["HIV", "TB", "HPV", "Flu", "ViralHepatitis"]]
# healthconditions = ['Type2Diabetes', 'ChronicKidneyDisease', 'CervicalCancer', 'ProstateCancer', 'RoadInjuries', 'DomesticViolence']
# 
# Combine with HIV
diseases = ["HIV"] + healthconditions

# Filter the DataFrame for disease_class being 'ncd'
ncd_df = df[df["disease_class"] == "ncd"]

# Extract disease categories from the filtered DataFrame
chronic = ncd_df[ncd_df["disease_type"] == "chronic"]["condition"].tolist()
acute = ncd_df[ncd_df["disease_type"] == "acute"]["condition"].tolist()
remitting = ncd_df[ncd_df["disease_type"] == "remitting"]["condition"].tolist()

# ncd = chronic + acute + remitting

# Extract communicable diseases with disease_class as 'sis'
communicable_diseases = df[df["disease_class"] == "sis"]["condition"].tolist()

# Initialize disease models with preloaded data
# mi.initialize_conditions(df, chronic, acute, remitting, communicable_diseases)

# ---------------------------------------------------------------------
# Initialize conditions, prevalence analyzer, and interactions
# ---------------------------------------------------------------------
prevalence_data_df = pd.read_csv(csv_prevalence)

prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases, prevalence_data=prevalence_data_df, inityear=inityear
)

# Define a function for disease-specific prevalence
def get_prevalence_function(disease):
    return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)
survivorship_analyzer = mi.SurvivorshipAnalyzer()
deaths_analyzer = mi.DeathsByAgeSexAnalyzer()

# -------------------------
# Demographics
# -------------------------

death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
death = ss.Deaths(death_rates)  # Use Demographics class implemented in mighti
fertility_rate = {'fertility_rate': pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rate)

ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))

# Initialize networks
mf = ss.MFNet(duration=1/24, acts=80)
maternal = ss.MaternalNet()
networks = [mf, maternal]

# -------------------------
# Diseases
# -------------------------

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
ncd_interactions = mi.read_interactions(csv_path_interactions) 
connectors = mi.create_connectors(ncd_interactions)

# Add NCD-NCD connectors to interactions
interactions.extend(connectors)


def get_deaths_module(sim):
    for module in sim.modules:
        if isinstance(module, mi.DeathsByAgeSexAnalyzer):
            return module
    raise ValueError("Deaths module not found in the simulation. Make sure you've added the DeathsByAgeSexAnalyzer to your simulation configuration")

def get_pregnancy_module(sim):
    for module in sim.modules:
        if isinstance(module, ss.Pregnancy):
            return module
    raise ValueError("Pregnancy module not found in the simulation.")


if __name__ == '__main__':
    # Initialize the simulation with connectors and force=True
    sim = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer],
        diseases=disease_objects,
        connectors=interactions,
        copy_inputs=False,
        label='Connector'
    )
 
    # Run the simulation
    sim.run()
    
    # # Plot the results for each simulation
    mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'Type2Diabetes')  
    mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'ChronicKidneyDisease')
    mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'CervicalCancer')
    mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'ProstateCancer')
    
    # Plot the results for each simulation
    # mi.plot_mean_prevalence(sim, prevalence_analyzer, 'Type2Diabetes', prevalence_data_df, init_year = inityear, end_year = endyear)  
    # mi.plot_mean_prevalence(sim, prevalence_analyzer, 'ChronicKidneyDisease', prevalence_data_df, init_year = inityear, end_year = endyear)
    # mi.plot_mean_prevalence(sim, prevalence_analyzer, 'CervicalCancer', prevalence_data_df, init_year = inityear, end_year = endyear)
    # mi.plot_mean_prevalence(sim, prevalence_analyzer, 'ProstateCancer', prevalence_data_df, init_year = inityear, end_year = endyear)
   
    # # Example usage:
    # mi.plot_age_group_prevalence(sim, prevalence_analyzer, 'Type2Diabetes', prevalence_data_df, init_year = inityear, end_year = endyear) 
    # mi.plot_age_group_prevalence(sim, prevalence_analyzer, 'ChronicKidneyDisease', prevalence_data_df, init_year = inityear, end_year = endyear) 
    # mi.plot_age_group_prevalence(sim, prevalence_analyzer, 'CervicalCancer', prevalence_data_df, init_year = inityear, end_year = endyear) 
    # mi.plot_age_group_prevalence(sim, prevalence_analyzer, 'ProstateCancer', prevalence_data_df, init_year = inityear, end_year = endyear) 
    
    
    