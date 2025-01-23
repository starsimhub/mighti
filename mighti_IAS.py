import starsim as ss
import mighti as mi  
import pandas as pd
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt


# Define diseases
    # 'Type1Diabetes', 'Type2Diabetes', 'Obesity', 'Hypertension',
    # 'Depression','Accident', 'Alzheimers', 'Assault', 'CerebrovascularDisease',
    # 'ChronicLiverDisease','ChronicLowerRespiratoryDisease', 'HeartDisease',
    # 'ChronicKidneyDisease','Flu','HPV',
    # 'CervicalCancer','ColorectalCancer', 'BreastCancer', 'LungCancer', 'ProstateCancer', 'OtherCancer',
    # 'Parkinsons','Smoking', 'Alcohol', 'BRCA', 'ViralHepatitis', 'Poverty'

ncds = [
      'Type2Diabetes',
]

diseases = ['HIV'] + ncds

beta = 0.0005  # Transmission probability for HIV
n_agents = 500000  # Number of agents in the simulation
inityear = 2021  # Simulation start year
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
    
eswatini_t2d_data = {}
for year in years:
    t2d_prevalence_data, _ = mi.initialize_prevalence_data(
        diseases= ['Type2Diabetes'], 
        csv_file_path='mighti/data/prevalence_data_eswatini.csv', 
        inityear=year
    )
    eswatini_t2d_data[year] = t2d_prevalence_data['Type2Diabetes']


# -------------------------
# Demographics
# -------------------------

fertility_rates = {'fertility_rate': pd.read_csv(sc.thispath() / 'tests/test_data/eswatini_asfr.csv')}
pregnancy = ss.Pregnancy(pars=fertility_rates)
death_rates = {'death_rate': pd.read_csv(sc.thispath() / 'tests/test_data/eswatini_deaths.csv'), 'rate_units': 1}
death = ss.Deaths(death_rates)
ppl = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age_2007.csv'))

# -------------------------
# Networks
# -------------------------

mf = ss.MFNet(duration=1/24, acts=80)
maternal = ss.MaternalNet()
networks = [mf, maternal]

# -------------------------
# Disease Objects
# -------------------------

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

# disease_objects = []
# for disease in ncds:
#     init_prev = ss.bernoulli(0.1351)  # Ensure correct initialization
#     if disease == 'Type2Diabetes':
#         disease_obj = mi.Type2Diabetes()
#         disease_obj.update_pars(pars={'init_prev': init_prev})  
#     elif disease == 'Obesity':
#         disease_obj = mi.Obesity()
#         disease_obj.update_pars(pars={'init_prev': init_prev}) 
#     disease_objects.append(disease_obj)
# HIV-specific setup
hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
disease_objects.append(hiv_disease)

        
# Initialize the PrevalenceAnalyzer
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)

# Load existing HIV and NCD interactions
interaction_functions = {
    'Type2Diabetes': mi.hiv_type2diabetes,
    'Obesity': mi.hiv_obesity,
}

# Initialize interaction objects for HIV-NCD interactions
interactions = []
for disease in ncds:
    interaction_obj = interaction_functions[disease]()  # Call the corresponding function
    interactions.append(interaction_obj)



# Initialize the simulation
sim = ss.Sim(
    n_agents=n_agents,
    networks=networks,
    diseases=disease_objects,  # Pass the full list of diseases (HIV + NCDs)
    analyzers=[prevalence_analyzer],
    start=inityear,
    stop=endyear,
    connectors=interactions,  # Both HIV-NCD and NCD-NCD interactions
    people=ppl,
    demographics=[pregnancy, death],
    copy_inputs=False
)



sim.run()


# # Get total population size
# total_agents = len(sim.people)  

# # Overall T2D prevalence among PLHIV and HIV-negative individuals
# print(f"T2D Prevalence Among PLHIV in 2025: {sim.results['type2diabetes']['prevalence_in_plhiv'][0] * 100:.2f}%")
# print(f"T2D Prevalence Among HIV-negative individuals in 2025: {sim.results['type2diabetes']['prevalence_in_hivneg'][0] * 100:.2f}%")

# print(f"T2D Prevalence Among PLHIV in 2050: {sim.results['type2diabetes']['prevalence_in_plhiv'][-1] * 100:.2f}%")
# print(f"T2D Prevalence Among HIV-negative individuals in 2050: {sim.results['type2diabetes']['prevalence_in_hivneg'][-1] * 100:.2f}%")

# age_groups = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
# sexes = ['male', 'female']

# for sex in sexes:
#     for i in range(len(age_groups) - 1):
#         age_min, age_max = age_groups[i], age_groups[i + 1]
#         key_name = f'T2D_prevalence_{sex}_{age_min}_{age_max}'

#         if key_name in sim.results['type2diabetes']:
#             prevalence_2025 = sim.results['type2diabetes'][key_name][0] * 100  # Year 2025
#             prevalence_2050 = sim.results['type2diabetes'][key_name][-1] * 100  # Year 2050
#             print(f"T2D Prevalence in {sex.capitalize()} PLHIV aged {age_min}-{age_max}:")
#             print(f"  - 2025: {prevalence_2025:.2f}%")
#             print(f"  - 2050: {prevalence_2050:.2f}%\n")
#         else:
#             print(f"Key {key_name} not found in sim.results['type2diabetes']")


# # Get total population size
# total_agents = len(sim.people)

# # Overall T2D prevalence among PLHIV and HIV-negative individuals
# t2d_plhiv_2025 = sim.results['type2diabetes']['prevalence_in_plhiv'][0] * 100
# t2d_hivneg_2025 = sim.results['type2diabetes']['prevalence_in_hivneg'][0] * 100
# t2d_plhiv_2050 = sim.results['type2diabetes']['prevalence_in_plhiv'][-1] * 100
# t2d_hivneg_2050 = sim.results['type2diabetes']['prevalence_in_hivneg'][-1] * 100

# # Ratio of T2D prevalence in PLHIV vs HIV-negative
# ratio_2025 = t2d_plhiv_2025 / t2d_hivneg_2025 if t2d_hivneg_2025 > 0 else np.nan
# ratio_2050 = t2d_plhiv_2050 / t2d_hivneg_2050 if t2d_hivneg_2050 > 0 else np.nan

# # Total number of T2D cases
# total_t2d_2025 = sim.results['type2diabetes']['n_affected'][0]
# total_t2d_2050 = sim.results['type2diabetes']['n_affected'][-1]

# # Number of PLHIV
# total_plhiv_2025 = np.count_nonzero(sim.people.hiv.infected)
# total_plhiv_2050 = total_plhiv_2025  # Assuming we don't have direct HIV prevalence over time

# # Number of T2D cases in PLHIV
# t2d_in_plhiv_2025 = sim.results['type2diabetes']['prevalence_in_plhiv'][0] * total_plhiv_2025
# t2d_in_plhiv_2050 = sim.results['type2diabetes']['prevalence_in_plhiv'][-1] * total_plhiv_2050

# # Percentage of all T2D cases in PLHIV
# percent_t2d_in_plhiv_2025 = (t2d_in_plhiv_2025 / total_t2d_2025) * 100 if total_t2d_2025 > 0 else np.nan
# percent_t2d_in_plhiv_2050 = (t2d_in_plhiv_2050 / total_t2d_2050) * 100 if total_t2d_2050 > 0 else np.nan

# # Print results
# print(f"T2D Prevalence Among PLHIV in 2025: {t2d_plhiv_2025:.2f}%")
# print(f"T2D Prevalence Among HIV-negative individuals in 2025: {t2d_hivneg_2025:.2f}%")
# print(f"T2D Prevalence Among PLHIV in 2050: {t2d_plhiv_2050:.2f}%")
# print(f"T2D Prevalence Among HIV-negative individuals in 2050: {t2d_hivneg_2050:.2f}%")
# print(f"PLHIV have {ratio_2025:.2f} times higher T2D prevalence than HIV-negative individuals in 2025.")
# print(f"PLHIV have {ratio_2050:.2f} times higher T2D prevalence than HIV-negative individuals in 2050.")
# print(f"Percentage of all T2D cases occurring in PLHIV (2025): {percent_t2d_in_plhiv_2025:.2f}%")
# print(f"Percentage of all T2D cases occurring in PLHIV (2050): {percent_t2d_in_plhiv_2050:.2f}%")


# Get total population size
total_agents = len(sim.people)

# Identify PLHIV and HIV-negative individuals
plhiv = sim.people.hiv.infected
hivneg = ~plhiv

# Identify individuals aged 18+
age_18_plus = sim.people.age >= 18
plhiv_18_plus = plhiv & age_18_plus
hivneg_18_plus = hivneg & age_18_plus

# Extract correct time indices for 2025 and 2050
time_index_2025 = np.where(sim.results['timevec'] == 2025)[0][0]
time_index_2050 = np.where(sim.results['timevec'] == 2050)[0][0]

# Compute Type 2 Diabetes prevalence among PLHIV and HIV-negative individuals aged 18+ in 2025 and 2050
t2d_in_plhiv_18plus_2025 = sim.results['type2diabetes']['prevalence_in_plhiv'][time_index_2025] * np.count_nonzero(plhiv_18_plus)
t2d_in_plhiv_18plus_2050 = sim.results['type2diabetes']['prevalence_in_plhiv'][time_index_2050] * np.count_nonzero(plhiv_18_plus)

t2d_in_hivneg_18plus_2025 = sim.results['type2diabetes']['prevalence_in_hivneg'][time_index_2025] * np.count_nonzero(hivneg_18_plus)
t2d_in_hivneg_18plus_2050 = sim.results['type2diabetes']['prevalence_in_hivneg'][time_index_2050] * np.count_nonzero(hivneg_18_plus)

# Compute total T2D cases among all individuals aged 18+
total_t2d_18plus_2025 = t2d_in_plhiv_18plus_2025 + t2d_in_hivneg_18plus_2025
total_t2d_18plus_2050 = t2d_in_plhiv_18plus_2050 + t2d_in_hivneg_18plus_2050

# Compute T2D prevalence among PLHIV and HIV-negative individuals aged 18+
prevalence_in_plhiv_18plus_2025 = sim.results['type2diabetes']['prevalence_in_plhiv'][time_index_2025] * 100
prevalence_in_plhiv_18plus_2050 = sim.results['type2diabetes']['prevalence_in_plhiv'][time_index_2050] * 100

prevalence_in_hivneg_18plus_2025 = sim.results['type2diabetes']['prevalence_in_hivneg'][time_index_2025] * 100
prevalence_in_hivneg_18plus_2050 = sim.results['type2diabetes']['prevalence_in_hivneg'][time_index_2050] * 100

# Compute ratio of T2D prevalence in PLHIV vs. HIV-negative individuals (aged 18+)
ratio_18plus_2025 = prevalence_in_plhiv_18plus_2025 / prevalence_in_hivneg_18plus_2025 if prevalence_in_hivneg_18plus_2025 > 0 else np.nan
ratio_18plus_2050 = prevalence_in_plhiv_18plus_2050 / prevalence_in_hivneg_18plus_2050 if prevalence_in_hivneg_18plus_2050 > 0 else np.nan

# Identify individuals aged 50+
age_50_plus = sim.people.age >= 50
plhiv_50_plus = plhiv & age_50_plus

# Compute total T2D cases among individuals aged 50+
t2d_in_plhiv_50plus_2025 = sim.results['type2diabetes']['prevalence_in_plhiv'][time_index_2025] * np.count_nonzero(plhiv_50_plus)
t2d_in_plhiv_50plus_2050 = sim.results['type2diabetes']['prevalence_in_plhiv'][time_index_2050] * np.count_nonzero(plhiv_50_plus)

total_t2d_50plus_2025 = sim.results['type2diabetes']['prevalence'][time_index_2025] * np.count_nonzero(age_50_plus)
total_t2d_50plus_2050 = sim.results['type2diabetes']['prevalence'][time_index_2050] * np.count_nonzero(age_50_plus)

# Compute percentage of all T2D cases occurring in PLHIV aged 50+
percent_t2d_in_plhiv_50plus_2025 = (t2d_in_plhiv_50plus_2025 / total_t2d_50plus_2025) * 100 if total_t2d_50plus_2025 > 0 else np.nan
percent_t2d_in_plhiv_50plus_2050 = (t2d_in_plhiv_50plus_2050 / total_t2d_50plus_2050) * 100 if total_t2d_50plus_2050 > 0 else np.nan

# Print results
print(f"T2D Prevalence Among PLHIV aged 18+ in 2025: {prevalence_in_plhiv_18plus_2025:.2f}%")
print(f"T2D Prevalence Among HIV-negative aged 18+ in 2025: {prevalence_in_hivneg_18plus_2025:.2f}%")
print(f"T2D Prevalence Among PLHIV aged 18+ in 2050: {prevalence_in_plhiv_18plus_2050:.2f}%")
print(f"T2D Prevalence Among HIV-negative aged 18+ in 2050: {prevalence_in_hivneg_18plus_2050:.2f}%")

print(f"PLHIV have {ratio_18plus_2025:.2f} times higher T2D prevalence than HIV-negative individuals in 2025.")
print(f"PLHIV have {ratio_18plus_2050:.2f} times higher T2D prevalence than HIV-negative individuals in 2050.")

print(f"Percentage of all T2D cases occurring in PLHIV aged 50+ in 2025: {percent_t2d_in_plhiv_50plus_2025:.1f}%")
print(f"Percentage of all T2D cases occurring in PLHIV aged 50+ in 2050: {percent_t2d_in_plhiv_50plus_2050:.1f}%")
    
# Retrieve prevalence data
time = sim.results['timevec']
prevalence_plhiv = sim.results['type2diabetes']['prevalence_in_plhiv'] * 100  # Convert to %
prevalence_hivneg = sim.results['type2diabetes']['prevalence_in_hivneg'] * 100  # Convert to %

# Plot
plt.figure(figsize=(10, 5))
plt.plot(time, prevalence_plhiv, label="T2D in PLHIV", color='red', linestyle='--')
plt.plot(time, prevalence_hivneg, label="T2D in HIV-Negative", color='blue')
plt.xlabel("Year")
plt.ylabel("Prevalence (%)")
plt.title("T2D Prevalence Among PLHIV vs. HIV-Negative Individuals")
plt.legend()
plt.grid()
plt.show()