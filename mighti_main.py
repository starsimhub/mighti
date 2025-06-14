import starsim as ss
import stisim as sti
import sciris as sc
import mighti as mi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

##### TO DO #####
# Calibration 
# Relative Risk implementation 


# ---------------------------------------------------------------------
# Define population size and simulation timeline
# ---------------------------------------------------------------------
n_agents = 100_000 # Number of agents in the simulation
inityear = 2000  # Simulation start year
endyear = 2001
region = 'nyc'

# ---------------------------------------------------------------------
# Specify data file paths
# ---------------------------------------------------------------------


# Parameters
csv_path_params = f'mighti/data/{region}_parameters.csv'

# Relative Risks
csv_path_interactions = "mighti/data/rel_sus.csv"

# Prevalence data
csv_prevalence = f'mighti/data/{region}_prevalence.csv'

# Fertility data 
csv_path_fertility = f'mighti/data/{region}_asfr.csv'

# Death data
csv_path_death = f'mighti/data/{region}_mortality_rates_{inityear}.csv'

# Age distribution data
csv_path_age = f'mighti/data/{region}_age_distribution_{inityear}.csv'

# Load the mortality rates and ensure correct format
mortality_rates_year = pd.read_csv(csv_path_death)

# Load the age distribution data for the specified year
age_distribution_year = pd.read_csv(csv_path_age)

# Load parameters
df = pd.read_csv(csv_path_params)
df.columns = df.columns.str.strip()

# Combine with HIV
healthconditions = ['AlcoholUseDisorder', 'SubstanceUseDisorder', 'Depression']
diseases = ["HIV"] + healthconditions

# Filter the DataFrame for disease_class being 'ncd'
ncd_df = df[df["disease_class"] == "ncd"]

# Extract disease categories from the filtered DataFrame
chronic = ncd_df[ncd_df["disease_type"] == "chronic"]["condition"].tolist()
acute = ncd_df[ncd_df["disease_type"] == "acute"]["condition"].tolist()
remitting = ncd_df[ncd_df["disease_type"] == "remitting"]["condition"].tolist()

# Extract communicable diseases with disease_class as 'sis'
communicable_diseases = df[df["disease_class"] == "sis"]["condition"].tolist()


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
maternal = ss.MaternalNet()
structuredsexual = sti.StructuredSexual()
networks = [maternal, structuredsexual]

# -------------------------
# Social Determinant of Health
# -------------------------

housing_module = mi.HousingSituation()


# -------------------------
# Diseases
# -------------------------

# Initialize disease conditions
hiv_disease = sti.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')),
                      init_prev_data=None,   
                      p_hiv_death=None, 
                      include_aids_deaths=True, 
                      beta={'structuredsexual': [0.01, 0.01], 'maternal': [0.01, 0.01]})

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

# -------------------------
# Interventions
# -------------------------

interventions = [
    # Universal, high-probability annual HIV testing (ramping up in early 2010s)
    sti.HIVTest(test_prob_data=[0.6, 0.7, 0.95], years=[2000, 2007, 2016]),
    # Test and treat: ART for nearly all diagnosed from 2010 onward
    sti.ART(pars={'future_coverage': {'year': 2005, 'prop': 0.95}}),
    # VMMC scale-up: reach 30% by 2015
    sti.VMMC(pars={'future_coverage': {'year': 2015, 'prop': 0.30}}),
    # PrEP for high-risk (starts low, ramps up)
    sti.Prep(pars={'coverage': [0, 0.05, 0.25], 'years': [2007, 2015, 2020]})
]

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
    # Initialize the simulation
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
        interventions=interventions,
        copy_inputs=False,
        label='Connector'
    )
    sim.init()  # ← this populates sim.people, sim.tvec, etc.
    housing_module.initialize(sim)
    sim.run()  # Run only after both are initialized
    
