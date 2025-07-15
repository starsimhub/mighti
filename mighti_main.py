"""
MIGHTI Simulation Script for a selected region: HIV and Health Conditions Interaction Modeling

This script initializes and runs an agent-based simulation using the MIGHTI framework
(built on StarSim and STI-Sim) to analyze the interplay between HIV and
other health conditions (HCs) in selected country. 
It loads demographic data, initializes diseases and networks, 
applies interventions, and analyzes prevalence and mortality outcomes for the selected period.

Key components:
- Loads parameters and prevalence data from CSV files.
- Initializes networks: maternal and structured sexual.
- Initializes HIV and HC modules.
- Sets up demographic modules (deaths, pregnancy).
- Applies HIV interventions (e.g., ART, VMMC).
- Computes and plots prevalence, mortality rates, and life expectancy.

To run: `python mighti_main.py`
"""


import logging
import mighti as mi
import numpy as np
import pandas as pd
import prepare_data_for_year
import starsim as ss
import stisim as sti
from mighti.sdoh import HousingSituation 


# Set up logging and random seeds for reproducibility
logger = logging.getLogger('MIGHTI')
logger.setLevel(logging.INFO) 


# ---------------------------------------------------------------------
# Define population size and simulation timeline
# ---------------------------------------------------------------------
n_agents = 10_000 # Number of agents in the simulation
inityear = 2000  # Simulation start year
endyear = 2010
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
csv_path_death = f'mighti/data/{region}_deaths.csv'

# Age distribution data
csv_path_age = f'mighti/data/{region}_age_2023.csv'


# Load the mortality rates and ensure correct format
mortality_rates_year = pd.read_csv(csv_path_death)

# Load the age distribution data for the specified year
age_distribution_year = pd.read_csv(csv_path_age)

# Load parameters
df = pd.read_csv(csv_path_params)
df.columns = df.columns.str.strip()

# Combine with HIV
healthconditions = ['AlcoholUseDisorder', 'SubstanceUseDisorder', 'Depression']
# healthconditions = ['Depression']
diseases = ["HIV"] + healthconditions

# Data paths for post process
mx_path = f'mighti/data/{region}_mx.csv'
ex_path = f'mighti/data/{region}_ex.csv'

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

maternal = ss.MaternalNet()
structuredsexual = sti.StructuredSexual()
networks = [maternal, structuredsexual]

# -------------------------
# SDoH
# -------------------------

housing_module = HousingSituation(prob=0.4)  # You can adjust this probability as needed


# -------------------------
# Diseases
# -------------------------
hiv_disease = sti.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')),
                      init_prev_data=None,   
                      p_hiv_death=None, 
                      include_aids_deaths=False, 
                      beta={'structuredsexual': [0.010754946814739815, 0.010754946814739815], 
                            'maternal': [0.011537993293074214, 0.011537993293074214]})
    # Best pars: {'hiv_beta_m2f': 0.010754946814739815, 'hiv_beta_m2c': 0.011537993293074214} seed: 12345

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
# Intervention
# -------------------------
intervention_hospital = [
    mi.ImproveHospitalDischarge(
        disease_name='depression',
        multiplier=10.0,
        start_day=0,
        end_day=10,
        label='FastDischarge'
    )
]

intervention_housing = [mi.GiveHousingToDepressed(coverage=1, start_day=0)]


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
    # # Initialize the simulation with connectors and force=True
    # sim_without = ss.Sim(
    #     n_agents=n_agents,
    #     networks=networks,
    #     start=inityear,
    #     stop=endyear,
    #     people=ppl,
    #     demographics=[pregnancy, death],
    #     analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer],
    #     diseases=disease_objects,
    #     connectors=interactions,
    #     # interventions = interventions,
    #     copy_inputs=False,
    #     label='No Intervention'
    # )
    
    # sim_with = ss.Sim(
    #     n_agents=n_agents,
    #     networks=networks,
    #     start=inityear,
    #     stop=endyear,
    #     people=ppl,
    #     demographics=[pregnancy, death],
    #     analyzers=[deaths_analyzer, survivorship_analyzer, prevalence_analyzer],
    #     diseases=disease_objects,
    #     connectors=interactions,
    #     interventions = intervention_hospital,
    #     copy_inputs=False,
    #     label='With Intervention'
    # )
     
    # # Run 
    # msim = ss.MultiSim(sims=[sim_with,sim_without])
    # msim.run(parallel=False)
    # housing_module.initialize(msim)     
    # msim.housing_module = housing_module
    # sim_without = msim.sims[0]
    # sim_with = msim.sims[1] 
    # print(np.count_nonzero(sim_without.diseases['depression'].hospitalized))
    # print(np.count_nonzero(sim_with.diseases['depression'].hospitalized))    


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
        interventions = intervention_housing,
        copy_inputs=False,
        label='Without Interventions'
    )

    sim.init()
    housing_module.initialize(sim)
    sim.housing_module = housing_module
        
    # Run the simulation
    sim.run()
    sim.housing_module = housing_module

    print(np.count_nonzero(housing_module.housing_unstable)) #s without intervention 



    # mi.plot_mean_prevalence(sim, prevalence_analyzer, 'HIV', prevalence_data_df, init_year = inityear, end_year = endyear)  
    # mi.plot_mean_prevalence(sim, prevalence_analyzer, 'Depression', prevalence_data_df, init_year = inityear, end_year = endyear)  
    # mi.plot_mean_prevalence(sim, prevalence_analyzer, 'AlcoholUseDisorder', prevalence_data_df, init_year = inityear, end_year = endyear)  
    # mi.plot_mean_prevalence(sim, prevalence_analyzer, 'SubstanceUseDisorder', prevalence_data_df, init_year = inityear, end_year = endyear)  

    # # Mortality rates and life table
    # target_year = endyear - 1
    
    # # Get the modules
    # deaths_module = get_deaths_module(sim)
    # pregnancy_module = get_pregnancy_module(sim)
    
    # df_mx = mi.calculate_mortality_rates(sim, deaths_module, year=target_year, max_age=100, radix=n_agents)

    # df_mx_male = df_mx[df_mx['sex'] == 'Male']
    # df_mx_female = df_mx[df_mx['sex'] == 'Female']
    
    
    # life_table = mi.calculate_life_table_from_mx(sim, df_mx_male, df_mx_female, max_age=100)
        
    # # Plot life expectancy comparison
    # mi.plot_life_expectancy(life_table, pd.read_csv(ex_path), year = target_year, max_age=100, figsize=(14, 10), title=None)

