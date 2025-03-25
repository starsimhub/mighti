import starsim as ss
import mighti as mi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define population size and simulation timeline
beta = 0.001
n_agents = 5000
inityear = 2017
endyear = 2050

# Specify data file paths
csv_path_params = 'mighti/data/eswatini_parameters.csv'
csv_path_interactions = "mighti/data/rel_sus.csv"
csv_prevalence = 'mighti/data/prevalence_data_eswatini.csv'
csv_path_fertility = 'mighti/data/eswatini_asfr.csv'
csv_path_death = 'mighti/data/eswatini_deaths.csv'
csv_path_age = 'mighti/data/eswatini_age_2023.csv'

# Load health conditions to include in the simulation
df = pd.read_csv(csv_path_params)
df.columns = df.columns.str.strip()
healthconditions = ['Type2Diabetes', 'ChronicKidneyDisease', 'CervicalCancer', 'ProstateCancer', 'RoadInjuries', 'DomesticViolence']
diseases = ["HIV"] + healthconditions

# Initialize prevalence data
prevalence_data_df = pd.read_csv(csv_prevalence)
prevalence_data, age_bins = mi.initialize_prevalence_data(diseases, prevalence_data=prevalence_data_df, inityear=inityear)

# Define a function for disease-specific prevalence
def get_prevalence_function(disease):
    return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=healthconditions)

# Initialize demographics
fertility_rates = {'fertility_rate': pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rates)
death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
death = mi.CustomDeaths(death_rates)
ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))

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

# Initialize interactions
ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
interactions = [ncd_hiv_connector]
ncd_interactions = mi.read_interactions("mighti/data/rel_sus.csv")
connectors = mi.create_connectors(ncd_interactions)
interactions.extend(connectors)

if __name__ == '__main__':
    # Ensure the 'people' key is registered in the simulation
    # ppl.name = 'people'
    # death.name = 'customdeaths'

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
    
    
    # Usage example
    # Calculate life expectancy using tracked deaths
    cumulative_deaths = death.get_cumulative_deaths()
    mortality_rates = mi.extract_mortality_rates(sim, cumulative_deaths)
    # life_table = mi.calculate_life_table(mortality_rates)
    # print(life_table)
    
    # Plot the mortality rates
    mi.plot_mortality_rates(csv_path_death,mortality_rates)

    # # Validate life expectancy
    # predicted_life_expectancy = mi.calculate_life_expectancy(sim, cumulative_deaths)
    # print(predicted_life_expectancy)

    # # Load actual mortality data
    # death_data = pd.read_csv(csv_path_death)
    # age_data = pd.read_csv(csv_path_age)
    
    # # Filter for the year 2020
    # death_data_2020 = death_data[death_data['Time'] == 2020]
    
    # # Calculate actual mortality rates
    # # Merge death data with age distribution data to get mortality rates per age group
    # merged_data = pd.merge(death_data_2020, age_data, left_on='AgeGrpStart', right_on='age')
    
    # # Calculate mortality rates
    # merged_data['mortality_rate'] = merged_data['mx'] / merged_data['value']
    
    # # Extract relevant columns
    # actual_death_data = merged_data[['age', 'mortality_rate']]
    # ages = np.arange(0, 101)

    # plt.figure(figsize=(10, 6))
    # plt.plot(ages, [mortality_rates.get(age, 0) for age in ages], label='Simulated Mortality Rates')
    # plt.plot(actual_death_data['age'],  merged_data['mx'], label='Actual Mortality Rates', linestyle='--')
    # plt.xlabel('Age')
    # plt.ylabel('Mortality Rate')
    # plt.title('Mortality Rates by Age')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Calculate SSE
    # sse = mi.compare_life_expectancy(predicted_life_expectancy, actual_life_expectancy)
    # print(f'Sum of Squared Errors (SSE): {sse}')

    # # Plot survival curves
    # mi.plot_survival_curves(predicted_life_expectancy, actual_life_expectancy)
