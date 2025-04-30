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
n_agents = 50000 # Number of agents in the simulation
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
ncd = ['Type2Diabetes','ChronicKidneyDisease','ProstateCancer','CervicalCancer']
diseases = ['HIV'] + ncd

# Load prevalence data from the CSV file
prevalence_data_df = pd.read_csv(csv_prevalence)

prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases, prevalence_data=prevalence_data_df, inityear=inityear
)

print("Age Bins for HIV in prevalence_data:", sorted(prevalence_data['HIV']['male'].keys()))
print("Age Bins for HIV in age_bins:", age_bins['HIV'])

# Define a function for disease-specific prevalence
# def get_prevalence_function(disease):
#     return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

def get_prevalence_function(disease):
    def prevalence_func(*args, **kwargs):
        interpolated_values = mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, slice(None))
        print(f"Interpolated values for {disease}: {interpolated_values}")

        # Apply relative susceptibility
        rel_sus = getattr(sim.people, disease.lower())['rel_sus']
        adjusted_values = interpolated_values * rel_sus
        print(f"Relative Susceptibility (rel_sus): {rel_sus}")
        print(f"Adjusted values after applying rel_sus for {disease}: {adjusted_values}")

        return adjusted_values
    return prevalence_func

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

# Automatically create disease objects for all diseases
disease_objects = []
for disease in ncd:
    init_prev = ss.bernoulli(get_prevalence_function(disease))
    print(f"Initial prevalence for {disease}: {init_prev}")

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

if __name__ == '__main__':     
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
    
    # # Load the reference population data
    reference_population_df = pd.read_csv(csv_path_age)
    
    # # Normalize the values to sum to 1 (convert to proportions)
    # reference_population_df['proportion'] = reference_population_df['value'] / reference_population_df['value'].sum()
    
    # reference_weights = reference_population_df['proportion']

    # # Specify the disease you want to process
    # disease_to_process = 'Type2Diabetes'  # Replace with the desired disease (e.g., 'HIV', 'ChronicKidneyDisease')
    
    # # Extract the age bins for the specified disease
    # bins = age_bins[disease_to_process]
    
    # # Group the reference population into the defined age bins
    # reference_population = {'male': {}, 'female': {}}
    # for i in range(len(bins) - 1):
    #     start = bins[i]
    #     end = bins[i + 1]
    
    #     if end == float('inf'):
    #         # For open-ended bins (e.g., age 80+)
    #         bin_population = reference_population_df[reference_population_df['age'] >= start]['proportion'].sum()
    #     else:
    #         # For defined age ranges
    #         bin_population = reference_population_df[
    #             (reference_population_df['age'] >= start) & (reference_population_df['age'] < end)
    #         ]['proportion'].sum()
        
    #     # Assign the same bin proportions to both male and female
    #     reference_population['male'][i] = bin_population
    #     reference_population['female'][i] = bin_population
    
    # # Output the reference population for the specified disease
    # print(f"Reference population for {disease_to_process}:")
    # print(reference_population)

    
    # # Plot the results for each simulation
    mi.plot_mean_prevalence(sim, prevalence_analyzer, 'Type2Diabetes', prevalence_data_df, inityear, endyear)
    mi.plot_mean_prevalence(sim, prevalence_analyzer, 'ChronicKidneyDisease', prevalence_data_df, inityear, endyear)
    mi.plot_mean_prevalence(sim, prevalence_analyzer, 'CervicalCancer', prevalence_data_df, inityear, endyear)
    mi.plot_mean_prevalence(sim, prevalence_analyzer, 'ProstateCancer', prevalence_data_df, inityear, endyear)

    
    # # Plot the results for each simulation
    # mi.plot_mean_prevalence_with_standardization(sim, prevalence_analyzer, 'Type2Diabetes', prevalence_data_df, inityear, endyear, reference_population)
    # mi.plot_mean_prevalence_with_standardization(sim, prevalence_analyzer, 'ChronicKidneyDisease', prevalence_data_df, inityear, endyear, reference_population)
    # mi.plot_mean_prevalence_with_standardization(sim, prevalence_analyzer, 'CervicalCancer', prevalence_data_df, inityear, endyear, reference_population)
    # mi.plot_mean_prevalence_with_standardization(sim, prevalence_analyzer, 'ProstateCancer', prevalence_data_df, inityear, endyear, reference_population)
    
    mi.plot_age_group_prevalence(sim, prevalence_analyzer, 'Type2Diabetes', prevalence_data_df, inityear, endyear, age_groups=None)
    mi.plot_age_group_prevalence(sim, prevalence_analyzer, 'ChronicKidneyDisease', prevalence_data_df, inityear, endyear, age_groups=None)
    mi.plot_age_group_prevalence(sim, prevalence_analyzer, 'CervicalCancer', prevalence_data_df, inityear, endyear, age_groups=None)
    mi.plot_age_group_prevalence(sim, prevalence_analyzer, 'ProstateCancer', prevalence_data_df, inityear, endyear, age_groups=None)


    # from scipy.interpolate import interp1d
    
    # # Example of manual interpolation validation
    # age_bins_array = np.array(age_bins['Type2Diabetes'])
    # male_prevalence = [prevalence_data['Type2Diabetes']['male'][age] for age in age_bins_array]
    # interp_func = interp1d(age_bins_array, male_prevalence, bounds_error=False, fill_value=(male_prevalence[0], male_prevalence[-1]))
    # print("Manual interpolation for ages [10, 17, 25, 45, 85]:", interp_func([10, 17, 25, 45, 85]))
    
    
    # ages = np.array([10, 17, 25, 45, 85])
    # bernoulli_probs = get_prevalence_function('Type2Diabetes')(None, sim, ages)
    # print("Bernoulli probabilities vs. interpolated values:")
    # for age, prob, interp in zip(ages, bernoulli_probs, interp_func([10, 17, 25, 45, 85])):
    #     print(f"Age: {age}, Bernoulli Probability: {prob}, Interpolated Value: {interp}")
        
    # # Check if relative susceptibility is applied to Type2Diabetes probabilities
    # adjusted_probabilities = get_prevalence_function('Type2Diabetes')(None, sim, slice(None))
    # print("Adjusted probabilities for Type2Diabetes:", adjusted_probabilities)    
    
    # def debug_adjusted_probabilities(disease, prevalence_data, age_bins, sim):
    #     """
    #     Debug the adjusted probabilities to identify age groups or population subsets being excluded.
    #     """
    #     # Step 1: Interpolated probabilities for all simulated agents
    #     interpolated_probs = mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, slice(None))
        
    #     # Step 2: Adjusted probabilities for all simulated agents
    #     adjusted_probs = get_prevalence_function(disease)(None, sim, slice(None))
        
    #     # Step 3: Debugging by age groups
    #     print(f"Debugging adjusted probabilities for {disease}:")
    #     for i, (start_age, end_age) in enumerate(zip(age_bins[disease][:-1], age_bins[disease][1:])):
    #         age_mask = (sim.people.age >= start_age) & (sim.people.age < end_age)
    #         avg_interpolated = interpolated_probs[age_mask].mean() if age_mask.any() else 0
    #         avg_adjusted = adjusted_probs[age_mask].mean() if age_mask.any() else 0
    #         print(f"Age group {start_age}-{end_age}: "
    #               f"Interpolated={avg_interpolated:.6f}, Adjusted={avg_adjusted:.6f}, "
    #               f"Count={age_mask.sum()}")
        
    #     # Step 4: Debugging by sex (if applicable)
    #     if 'male' in prevalence_data[disease]:
    #         male_mask = sim.people.male == 0  # Assuming 0 = male
    #         female_mask = sim.people.male == 1  # Assuming 1 = female
    #         avg_male = adjusted_probs[male_mask].mean() if male_mask.any() else 0
    #         avg_female = adjusted_probs[female_mask].mean() if female_mask.any() else 0
    #         print(f"Male average adjusted probability: {avg_male:.6f}")
    #         print(f"Female average adjusted probability: {avg_female:.6f}")
                
    # debug_adjusted_probabilities('Type2Diabetes', prevalence_data, age_bins, sim)

    # def print_age_bin_prevalence_2007(disease, sim, prevalence_data, age_bins, reference_population_df):
    #     """
    #     Print simulated and observed age-bin dependent prevalence for 2007.
    #     """
    #     print(f"Age-bin dependent prevalence for {disease} in 2007:")
    
    #     # Simulated prevalence
    #     simulated_prevalence = []
    
    #     # Access the 'affected' state for the disease
    #     disease_states = getattr(sim.people, disease.lower())
    #     if not isinstance(disease_states, sc.sc_odict.objdict):
    #         raise ValueError(f"The disease states for {disease} are not in the expected objdict format.")
    
    #     # Use the 'affected' state
    #     disease_mask = np.array(disease_states['affected'])
    
    #     for i, (start_age, end_age) in enumerate(zip(age_bins[disease][:-1], age_bins[disease][1:])):
    #         # Create a mask for agents in the relevant age group and alive
    #         age_mask = (sim.people.age >= start_age) & (sim.people.age < end_age) & sim.people.alive
    
    #         if age_mask.any():
    #             # Apply the mask safely
    #             prevalence = disease_mask[age_mask].mean()
    #         else:
    #             prevalence = 0
    #         simulated_prevalence.append(prevalence)
    #         print(f"Simulated prevalence for {start_age}-{end_age}: {prevalence:.6f}")
    
    #     # Observed prevalence
    #     print("\nObserved prevalence from prevalence_data:")
    #     observed_prevalence = []
    #     for i, (start_age, end_age) in enumerate(zip(age_bins[disease][:-1], age_bins[disease][1:])):
    #         age_mask = (reference_population_df['age'] >= start_age) & (reference_population_df['age'] < end_age)
    #         if age_mask.any():
    #             # Access observed prevalence data
    #             prevalence = prevalence_data[disease]['male'][start_age]  # Assuming values are stored by age bins
    #         else:
    #             prevalence = 0
    #         observed_prevalence.append(prevalence)
    #         print(f"Observed prevalence for {start_age}-{end_age}: {prevalence:.6f}")
    
    #     return simulated_prevalence, observed_prevalence
    
    
    # # Call the function for Type2Diabetes
    # simulated, observed = print_age_bin_prevalence_2007(
    #     'Type2Diabetes', sim, prevalence_data, age_bins, reference_population_df
    # )
        