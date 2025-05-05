import starsim as ss
import stisim as sti
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
n_agents = 100000 # Number of agents in the simulation
inityear = 2007  # Simulation start year
endyear = 2010

# ---------------------------------------------------------------------
# Specify data file paths
# ---------------------------------------------------------------------

# Parameters
csv_path_params = 'mighti/data/eswatini_parameters.csv'

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

# Define diseases
healthconditions = [condition for condition in df.condition if condition != "HIV"]
# healthconditions = ['Type2Diabetes']
diseases = ['HIV'] + healthconditions

# Load prevalence data from the CSV file
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


# -------------------------
# Demographics
# -------------------------

death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
death = mi.Deaths(death_rates)  # Use Demographics class implemented in mighti
fertility_rate = {'fertility_rate': pd.read_csv(csv_path_fertility)}
pregnancy = mi.Pregnancy(pars=fertility_rate)  

ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))

# Initialize networks
mf = ss.MFNet(duration=1/24, acts=80)
maternal = ss.MaternalNet()
networks = [mf, maternal]


# -------------------------
# Diseases
# -------------------------

# Initialize disease conditions
hiv_disease = sti.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)

# Automatically create disease objects for all diseases
disease_objects = []
for disease in healthconditions:
    init_prev = ss.bernoulli(get_prevalence_function(disease))
    
    # Dynamically get the disease class from `mi` module
    disease_class = getattr(mi, disease, None)
    
    if disease_class:
        disease_obj = disease_class(csv_path=csv_path_params, pars={"init_prev": init_prev})
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
ncd_interactions = mi.read_interactions(csv_path_interactions) 
connectors = mi.create_connectors(ncd_interactions)

# Add NCD-NCD connectors to interactions
interactions.extend(connectors)

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

    

if __name__ == '__main__':
    # Initialize the simulation with connectors and force=True
    sim = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        analyzers=[prevalence_analyzer, survivorship_analyzer],
        diseases=disease_objects,
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        connectors=interactions,
        copy_inputs=False,
        label='Connector'
    )
    
    # After initializing the simulation
    sim.run()
    
    # Get the modules
    deaths_module = get_deaths_module(sim)
    pregnancy_module = get_pregnancy_module(sim)
    

    year = 2009
    # Load observed mortality rate data
    observed_death_data = pd.read_csv('demography/eswatini_mortality_rates.csv')
    
    # Calculate mortality rates using `calculate_mortality_rates
    df_mortality_rates = mi.calculate_mortality_rates(sim, deaths_module, year=year, max_age=100, radix=n_agents)

    # Plot the mortality rates comparison
    # mi.plot_mortality_rates_comparison(df_mortality_rates, observed_death_data, observed_year=year, year=year)
    
    mi.plot_mortality_rates_comparison(
        df_metrics=df_mortality_rates, 
        observed_data=observed_death_data, 
        observed_year=year, 
        year=year, 
        log_scale=True, 
        title="Single-Age Mortality Rates Comparison"
    )
    # Create the life table
    life_table = mi.create_life_table(df_mortality_rates, year=year, n_agents=n_agents, max_age=100)
    print(life_table)
    
    # Load observed life expectancy data
    observed_LE = pd.read_csv('demography/eswatini_life_expectancy_by_age.csv')
    
    # Plot life expectancy comparison
    mi.plot_life_expectancy(life_table, observed_LE, year=year, max_age=100, figsize=(14, 10), title=None)
    
    # # Print life expectancy statement
    # mi.print_life_expectancy_statement(life_table)