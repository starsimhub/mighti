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
n_agents = 500000  # Number of agents in the simulation
inityear = 2007  # Simulation start year
endyear = 2009

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
ncd = ['Type2Diabetes']
diseases = ['HIV'] + ncd

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


# Before creating your Deaths object
mortality_rates = pd.read_csv(csv_path_death)

# Print original infant mortality rates
infant_rows = mortality_rates['AgeGrpStart'] == 0
male_rows = mortality_rates['Sex'] == 'Male'
female_rows = mortality_rates['Sex'] == 'Female'

# Get original rates by sex
male_infant_rate = mortality_rates.loc[infant_rows & male_rows, 'mx'].values[0]
female_infant_rate = mortality_rates.loc[infant_rows & female_rows, 'mx'].values[0]

print(f"Original infant mortality rates: Male={male_infant_rate:.6f}, Female={female_infant_rate:.6f}")

# Scale down infant mortality rates to 1/12 of their original value
# This assumes the issue is that annual rates are being applied monthly
mortality_rates.loc[infant_rows & male_rows, 'mx'] = mortality_rates.loc[infant_rows & male_rows, 'mx'] 
mortality_rates.loc[infant_rows & female_rows, 'mx'] = mortality_rates.loc[infant_rows & female_rows, 'mx'] 

# Print adjusted rates
adjusted_male_infant_rate = mortality_rates.loc[infant_rows & male_rows, 'mx'].values[0]
adjusted_female_infant_rate = mortality_rates.loc[infant_rows & female_rows, 'mx'].values[0]

print(f"Adjusted infant mortality rates: Male={adjusted_male_infant_rate:.6f}, Female={adjusted_female_infant_rate:.6f}")

# Now create standard death rates object with adjusted rates
death_rates = {'death_rate': mortality_rates, 'rate_units': 1}
death = ss.Deaths(death_rates)  # Use normal StarSim Deaths implementation

# Create demographics
fertility_rates = {'fertility_rate': pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rates)
# death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
# death = SaferDeaths(death_rates)
ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))

# Create the networks - sexual and maternal
mf = ss.MFNet(duration=1/24, acts=80)
maternal = ss.MaternalNet()
networks = [mf, maternal]

hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)

# Automatically create disease objects for all diseases
disease_objects = []
for disease in ncd:
    init_prev = ss.bernoulli(get_prevalence_function(disease))
    
    # Dynamically get the disease class from `mi` module
    disease_class = getattr(mi, disease, None)
    
    if disease_class:
        disease_obj = disease_class(disease_name=disease, csv_path=csv_path_params, pars={"init_prev": init_prev})
        print(f"init_prev: {init_prev}")
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

# Initialize the DeathTracker
death_tracker = mi.DeathTracker()

if __name__ == '__main__':
    # Initialize the simulation with connectors and force=True
    sim = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        analyzers=[death_tracker, prevalence_analyzer],
        diseases=disease_objects,
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        connectors=interactions,
        copy_inputs=False,
        label='Connector'
    )
    
    # Initialize the simulation
    sim.init()

    # Run the simulation
    sim.run()
    
    # After running the simulation
    mi.analyze_mortality_implementation(sim, death_tracker)
    
    # Step 3: Calculate mortality rates
    simulated_mortality_rates = mi.calculate_mortality_rates(
        death_tracker, 
        year=2009,
        age_groups=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    )
    
    observed_death_csv = 'demography/eswatini_deaths.csv'
    mi.plot_death_counts_comparison(death_tracker, observed_death_csv, 2009)

    
   # Calculate mortality rates from the simulation
    simulated_mortality_rates = mi.calculate_mortality_rates(
        death_tracker, 
        year=2009,
        age_groups=[0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    )
    
    # Load observed mortality rates
    observed_mortality_rates = pd.read_csv('mighti/data/eswatini_mortality_rates_2007.csv')
    observed_mortality_rates = observed_mortality_rates[observed_mortality_rates['AgeGrpStart'] <= 95]

    # Plot a comparison
    mi.plot_mortality_rates_comparison(
        simulated_mortality_rates, 
        observed_mortality_rates,
        observed_year=2009,  #
        year=2009,         
        log_scale=False,    
        title="Simulated (2009) vs Observed (2007) Mortality Rates"
    )