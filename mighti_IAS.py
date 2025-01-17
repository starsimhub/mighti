import starsim as ss
import mighti as mi  
import pandas as pd
import pylab as pl
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


# Define diseases
ncds = ['Type2Diabetes']
diseases = ['HIV'] + ncds
beta = 0.0005
n_agents = 50000
inityear = 2021
endyear = 2050

# -------------------------
# Prevalence Data
# -------------------------
prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases, csv_file_path='mighti/data/prevalence_data_eswatini.csv', inityear=inityear
)

years = [2007, 2011, 2017, 2021]
eswatini_hiv_data = {}
for year in years:
    hiv_prevalence_data, _ = mi.initialize_prevalence_data(
        diseases= ['HIV'], 
        csv_file_path='mighti/data/prevalence_data_eswatini.csv', 
        inityear=year
    )
    eswatini_hiv_data[year] = hiv_prevalence_data['HIV']  # Store data for the specific year


# -------------------------
# Demographics
# -------------------------
fertility_rates = {'fertility_rate': pd.read_csv('tests/test_data/eswatini_asfr.csv')}
pregnancy = ss.Pregnancy(pars=fertility_rates)
death_rates = {'death_rate': pd.read_csv('tests/test_data/eswatini_deaths.csv'), 'units': 1}
death = ss.Deaths(death_rates)
ppl = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age.csv'))

# -------------------------
# Networks
# -------------------------
mf = ss.MFNet(duration=1/24, acts=80)
maternal = ss.MaternalNet()
networks = [mf, maternal]

# Define a function for disease-specific prevalence
def get_prevalence_function(disease):
    return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

# Create disease objects
disease_objects = []
for disease in ncds:
    init_prev = ss.bernoulli(get_prevalence_function(disease))
    if disease == 'Type2Diabetes':
        disease_obj = mi.Type2Diabetes(init_prev=init_prev)
    elif disease == 'Obesity':
        disease_obj = mi.Obesity(init_prev=init_prev)
    disease_objects.append(disease_obj)
    
# HIV-specific setup
hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
disease_objects.append(hiv_disease)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer_with = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)
prevalence_analyzer_without = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)

# Load existing HIV and NCD interactions
interaction_functions = {
    'Type2Diabetes': mi.hiv_type2diabetes,
    'Obesity': mi.hiv_obesity,
}


interactions = [
    mi.hiv_type2diabetes(pars={"rel_sus_hiv_type2diabetes": 10})
]

for interaction in interactions:
    print(interaction)  # Should print the interaction object

# -------------------------
# Simulation With Interactions
# -------------------------
ppl_with_interactions = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age.csv'))
networks_with_interactions = [deepcopy(net) for net in networks]
pregnancy_with_interactions = ss.Pregnancy(pars=fertility_rates)
death_with_interactions = ss.Deaths(death_rates)

disease_objects_with_interactions = deepcopy(disease_objects)


sim_with_interactions = ss.Sim(
    n_agents=n_agents,
    networks=networks_with_interactions,
    diseases=disease_objects_with_interactions,
    analyzers=[prevalence_analyzer_with],
    start=inityear,
    end=endyear,
    connectors=interactions,
    people=ppl_with_interactions,
    demographics=[pregnancy_with_interactions, death_with_interactions],
    copy_inputs=False
)


# print(f"rel_sus for Type2Diabetes before simulation: {disease_objects_with_interactions[0].rel_sus[:10]}")
sim_with_interactions.run()
# print(f"rel_sus for Type2Diabetes after simulation: {disease_objects_with_interactions[0].rel_sus[:10]}")

mortality_with_interactions = sim_with_interactions.results['new_deaths']  # New deaths per time step
cumulative_mortality_with_interactions = sim_with_interactions.results['cum_deaths']  # Cumulative deaths

# -------------------------
# Simulation Without Interactions
# -------------------------
ppl_no_interactions = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age.csv'))
networks_no_interactions = [deepcopy(net) for net in networks]
pregnancy_no_interactions = ss.Pregnancy(pars=fertility_rates)
death_no_interactions = ss.Deaths(death_rates)

disease_objects_no_interactions = deepcopy(disease_objects)

sim_no_interactions = ss.Sim(
    n_agents=n_agents,
    networks=networks_no_interactions,
    diseases=disease_objects_no_interactions,
    analyzers=[prevalence_analyzer_without],
    start=inityear,
    end=endyear,
    people=ppl_no_interactions,
    demographics=[pregnancy_no_interactions, death_no_interactions],
    copy_inputs=False
)

print("Running Simulation Without Interactions...")
sim_no_interactions.run()

mortality_no_interactions = sim_no_interactions.results['new_deaths']  # New deaths per time step
cumulative_mortality_no_interactions = sim_no_interactions.results['cum_deaths']  # Cumulative deaths







# -------------------------
# Extract Mortality Data by Disease
# -------------------------
def extract_mortality(sim, disease_name):
    """Extract mortality for a specific disease."""
    disease_results = sim.results[disease_name.lower()]  # Disease-specific results
    mortality = disease_results['new_deaths']  # Adjust the key if necessary
    return mortality



# -------------------------
# Extract Prevalence Data by Gender
# -------------------------
def extract_prevalence_by_gender(prevalence_analyzer, disease_name):
    """Extract prevalence for a specific disease by gender."""
    try:
        prevalence_male = prevalence_analyzer.results[f'{disease_name}_prevalence_male'] * 100
        prevalence_female = prevalence_analyzer.results[f'{disease_name}_prevalence_female'] * 100
        return prevalence_male, prevalence_female
    except KeyError as e:
        print(f"Prevalence data not found for {disease_name}: {e}")
        return None, None
    
def extract_prevalence_by_hiv_status(prevalence_analyzer, disease_name):
    """Extract prevalence for a specific disease stratified by HIV status and gender."""
    try:
        prevalence_hivpos_male = prevalence_analyzer.results[f'{disease_name}_prevalence_hivpos_male'] * 100
        prevalence_hivneg_male = prevalence_analyzer.results[f'{disease_name}_prevalence_hivneg_male'] * 100
        prevalence_hivpos_female = prevalence_analyzer.results[f'{disease_name}_prevalence_hivpos_female'] * 100
        prevalence_hivneg_female = prevalence_analyzer.results[f'{disease_name}_prevalence_hivneg_female'] * 100
        return prevalence_hivpos_male, prevalence_hivneg_male, prevalence_hivpos_female, prevalence_hivneg_female
    except KeyError as e:
        print(f"Prevalence data not found for {disease_name}: {e}")
        return None, None, None, None
    
    
# -------------------------
# Plotting Mortality and Prevalence
# -------------------------
# diseases_to_plot = ['HIV', 'Type2Diabetes']  # Add 'Obesity' if needed
# fig, axs = pl.subplots(len(diseases_to_plot), 2, figsize=(15, len(diseases_to_plot) * 6))

# for i, disease_name in enumerate(diseases_to_plot):
#     # Extract mortality
#     mortality_with = extract_mortality(sim_with_interactions, disease_name)
#     mortality_without = extract_mortality(sim_no_interactions, disease_name)

#     # Extract prevalence by gender
#     prevalence_with_male, prevalence_with_female = extract_prevalence_by_gender(prevalence_analyzer_with, disease_name)
#     prevalence_without_male, prevalence_without_female = extract_prevalence_by_gender(prevalence_analyzer_without, disease_name)

#     # Plot mortality
#     axs[i, 0].plot(sim_with_interactions.yearvec, mortality_with, label='With Interactions', linestyle='-')
#     axs[i, 0].plot(sim_no_interactions.yearvec, mortality_without, label='Without Interactions', linestyle='--')
#     axs[i, 0].set_title(f'Mortality Comparison: {disease_name}', fontsize=16)
#     axs[i, 0].set_xlabel('Year', fontsize=14)
#     axs[i, 0].set_ylabel('Mortality Count', fontsize=14)
#     axs[i, 0].legend()
#     axs[i, 0].grid(True)

#     # Plot prevalence
#     if prevalence_with_male is not None and prevalence_without_male is not None:
#         axs[i, 1].plot(sim_with_interactions.yearvec, prevalence_with_male.mean(axis=1), label='Male (With Interactions)', linestyle='-')
#         axs[i, 1].plot(sim_no_interactions.yearvec, prevalence_without_male.mean(axis=1), label='Male (Without Interactions)', linestyle='--')
#         axs[i, 1].plot(sim_with_interactions.yearvec, prevalence_with_female.mean(axis=1), label='Female (With Interactions)', linestyle='-')
#         axs[i, 1].plot(sim_no_interactions.yearvec, prevalence_without_female.mean(axis=1), label='Female (Without Interactions)', linestyle='--')
#         axs[i, 1].set_title(f'Prevalence Comparison: {disease_name}', fontsize=16)
#         axs[i, 1].set_xlabel('Year', fontsize=14)
#         axs[i, 1].set_ylabel('Prevalence (%)', fontsize=14)
#         axs[i, 1].legend()
#         axs[i, 1].grid(True)

# pl.tight_layout()
# pl.show()


# pl.figure(figsize=(10,5))
# pl.plot(sim_with_interactions.yearvec, mortality_with_interactions, label='With Interactions', linestyle='-', color='red')
# pl.plot(sim_no_interactions.yearvec, mortality_no_interactions, label='Without Interactions', linestyle='--', color='blue')
# pl.legend()
# pl.title("Mortality Comparison")
# pl.show()


# # Debugging: Print values
# print("Checking extracted values:")
# print("With Interactions (HIV+ Male):", t2d_with_hiv_pos_male)
# print("With Interactions (HIV+ Female):", t2d_with_hiv_pos_female)
# print("With Interactions (HIV- Male):", t2d_with_hiv_neg_male)
# print("With Interactions (HIV- Female):", t2d_with_hiv_neg_female)
# print("Without Interactions (HIV+ Male):", t2d_without_hiv_pos_male)
# print("Without Interactions (HIV+ Female):", t2d_without_hiv_pos_female)
# print("Without Interactions (HIV- Male):", t2d_without_hiv_neg_male)
# print("Without Interactions (HIV- Female):", t2d_without_hiv_neg_female)


# # Check for exact equality
# hiv_same = np.array_equal(hiv_prevalence_with, hiv_prevalence_without)
# t2d_same = np.array_equal(t2d_prevalence_with, t2d_prevalence_without)

# # Compute absolute differences
# hiv_diff = np.abs(hiv_prevalence_with - hiv_prevalence_without)
# t2d_diff = np.abs(t2d_prevalence_with - t2d_prevalence_without)

# # Print results
# print(f"HIV prevalence identical: {hiv_same}")
# print(f"Type 2 Diabetes prevalence identical: {t2d_same}")

# # Display max difference
# print(f"Max HIV prevalence difference: {np.max(hiv_diff)}")
# print(f"Max T2D prevalence difference: {np.max(t2d_diff)}")



# Function to extract prevalence by sex
def extract_prevalence_by_sex(prevalence_analyzer, disease_name):
    """Extract prevalence data by sex."""
    try:
        prevalence_male = prevalence_analyzer.results[f'{disease_name}_prevalence_male']
        prevalence_female = prevalence_analyzer.results[f'{disease_name}_prevalence_female']
        return prevalence_male, prevalence_female
    except KeyError as e:
        print(f"Prevalence data not found for {disease_name}: {e}")
        return None, None

# Extract prevalence data
hiv_prevalence_male, hiv_prevalence_female = extract_prevalence_by_sex(prevalence_analyzer_with, 'HIV')
t2d_prevalence_male, t2d_prevalence_female = extract_prevalence_by_sex(prevalence_analyzer_with, 'Type2Diabetes')

# Extract age-grouped prevalence data
hiv_prevalence_by_age = prevalence_analyzer_with.results['HIV_prevalence_male']
t2d_prevalence_by_age = prevalence_analyzer_with.results['Type2Diabetes_prevalence_male']

# Ensure year vector is properly aligned
years = sim_with_interactions.yearvec

# -------------------------
# 1. Plot HIV Prevalence Over Time (Male & Female)
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(years, hiv_prevalence_male.mean(axis=1) * 100, label='HIV Male', linestyle='-', color='blue')
plt.plot(years, hiv_prevalence_female.mean(axis=1) * 100, label='HIV Female', linestyle='--', color='red')
plt.xlabel("Year")
plt.ylabel("HIV Prevalence (%)")
plt.title("HIV Prevalence Over Time (By Sex)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# 2. Plot T2D Prevalence Over Time (Male & Female)
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(years, t2d_prevalence_male.mean(axis=1) * 100, label='T2D Male', linestyle='-', color='blue')
plt.plot(years, t2d_prevalence_female.mean(axis=1) * 100, label='T2D Female', linestyle='--', color='red')
plt.xlabel("Year")
plt.ylabel("T2D Prevalence (%)")
plt.title("Type 2 Diabetes Prevalence Over Time (By Sex)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# 3. Plot HIV Prevalence By Age Group
# -------------------------
plt.figure(figsize=(10, 5))
for i, age_bin in enumerate(prevalence_analyzer_with.age_bins['HIV']):
    plt.plot(years, hiv_prevalence_by_age[:, i] * 100, label=f'Age {age_bin}')

plt.xlabel("Year")
plt.ylabel("HIV Prevalence (%)")
plt.title("HIV Prevalence Over Time (By Age Group)")
plt.legend(title="Age Group", loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

# -------------------------
# 4. Plot T2D Prevalence By Age Group
# -------------------------
plt.figure(figsize=(10, 5))
for i, age_bin in enumerate(prevalence_analyzer_with.age_bins['Type2Diabetes']):
    plt.plot(years, t2d_prevalence_by_age[:, i] * 100, label=f'Age {age_bin}')

plt.xlabel("Year")
plt.ylabel("T2D Prevalence (%)")
plt.title("Type 2 Diabetes Prevalence Over Time (By Age Group)")
plt.legend(title="Age Group", loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()