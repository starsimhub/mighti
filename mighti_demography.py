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
n_agents = 500000 # Number of agents in the simulation
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


death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
death = ss.Deaths(death_rates)  # Use normal StarSim Deaths implementation


fertility_rate = {'fertility_rate': pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rate)  # Use the correct parameter name

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

def get_deaths_module(sim):
    for module in sim.modules:
        if isinstance(module, ss.Deaths):
            return module
    raise ValueError("Deaths module not found in the simulation.")

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
        analyzers=[prevalence_analyzer],
        diseases=disease_objects,
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        # connectors=interactions,
        copy_inputs=False,
        label='Connector'
    )
    
    # After initializing the simulation
    sim.run()
    
    # Get the modules
    deaths_module = get_deaths_module(sim)
    pregnancy_module = get_pregnancy_module(sim)

    # Initialize lists to store yearly data
    years = list(range(inityear+1, endyear))
    simulated_imr = []
    
    # Extract data for each year
    for year in years:
        # Retrieve the number of births and deaths for the year
        births = pregnancy_module.get_births(year)
        infant_deaths = deaths_module.infant_deaths

        # Calculate the IMR for males and females
        imr= (infant_deaths / births) if births > 0 else 0

        # Append the IMR values to the lists
        simulated_imr.append(imr)

    # Store the data in a DataFrame
    simulated_data = pd.DataFrame({
        'Year': years,
        'IMR': simulated_imr,
    })

    # Print the simulated data
    print(simulated_data)
    
    mi.plot_imr('demography/eswatini_mortality_rates.csv', 'simulated_imr_data.csv', inityear, endyear)
    


    n_years = endyear - inityear + 1

    # Create results DataFrame
    df_results = mi.create_results_dataframe(sim, inityear, endyear, deaths_module)

    # Calculate metrics
    df_metrics = mi.calculate_metrics(df_results)

    # Plot the mortality rates comparison
    mi.plot_mortality_rates_comparison(df_metrics, 'demography/eswatini_mortality_rates.csv', observed_year=2011, year=2011)


    life_table = mi.create_life_table(df_metrics, year=2007, max_age=100)
    print(life_table)
    # mx_2011 = df_metrics[df_metrics['year']==2011]
    # life_table = mi.calculate_life_table(malemx,)
    life_table = pd.read_csv('path_to_life_table.csv')  # Load your life table DataFrame
    mi.plot_life_expectancy(life_table, year=2025, max_age=100, figsize=(14,10), title=None)
    # print(f"Infant deaths: {infant_deaths}, Total deaths: {total_deaths}")
    
    # # deaths_module = get_deaths_module(sim)
    # # print("Final Death Tracking Data:")
    # # print(deaths_module.death_tracking)
    
    # # Plot the death tracking data
    # male_deaths = deaths_module.death_tracking['Male']
    # female_deaths = deaths_module.death_tracking['Female']
    # ages = range(len(male_deaths))
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(ages, male_deaths, label='Male Deaths')
    # plt.plot(ages, female_deaths, label='Female Deaths')
    # plt.xlabel('Age')
    # plt.ylabel('Number of Deaths')
    # plt.title('Deaths by Age and Sex')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    #
    # mortality_rates = mi.calculate_mortality_rates(deaths_module)
    # life_table = mi.calculate_life_table(mortality_rates)
    # # print(life_table)
    
    # # Print the death tracking dictionary
    # print("Death tracking data:")
    # print(deaths_module.death_tracking)
    
    # # Plot the death tracking data
    # male_deaths = deaths_module.death_tracking['Male']
    # female_deaths = deaths_module.death_tracking['Female']
    # ages = range(len(male_deaths))
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(ages, male_deaths, label='Male Deaths')
    # plt.plot(ages, female_deaths, label='Female Deaths')
    # plt.xlabel('Age')
    # plt.ylabel('Number of Deaths')
    # plt.title('Deaths by Age and Sex')
    # plt.legend()
    # plt.grid(True)
    # plt.show()