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
inityear = 2007  # Simulation start year
endyear = 2030



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
# for disease in ncds:
#     init_prev = ss.bernoulli(get_prevalence_function(disease))
#     if disease == 'Type2Diabetes':
#         disease_obj = mi.Type2Diabetes(init_prev=init_prev)
#     elif disease == 'Obesity':
#         disease_obj = mi.Obesity(init_prev=init_prev)
#     disease_objects.append(disease_obj)

disease_objects = []
for disease in ncds:
    init_prev = ss.bernoulli(0.1351)  # Ensure correct initialization
    if disease == 'Type2Diabetes':
        disease_obj = mi.Type2Diabetes()
        disease_obj.update_pars(pars={'init_prev': init_prev})  
    elif disease == 'Obesity':
        disease_obj = mi.Obesity()
        disease_obj.update_pars(pars={'init_prev': init_prev}) 
    disease_objects.append(disease_obj)
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

# Debugging Initial Prevalence Before Simulation
print("\n--- Debugging Initial Prevalence Before Simulation ---")
for disease in disease_objects:
    if hasattr(disease.pars, 'init_prev'):
        print(f"{disease.name}: init_prev exists? {hasattr(disease.pars, 'init_prev')}")
        print(f"{disease.name}: init_prev probability = {disease.pars.init_prev.pars.p}")
    else:
        print(f"{disease.name}: `init_prev` NOT FOUND")
# Run the simulation
sim.run()


    
# -------------------------
# Debugging Births & Pregnancies
# -------------------------
print("\n--- Debugging Births & Pregnancies ---")
if 'pregnancy' in sim.results:
    print("Pregnancy module is active.")
    print("Pregnancy births recorded:", sim.results['pregnancy']['births'])
else:
    print("Pregnancy module not active in results.")

# -------------------------
# Debugging Aging & Population Over Time
# -------------------------
print("\n--- Debugging Aging & Population Over Time ---")
if hasattr(ppl, 'age'):
    print("Age data found in ppl.age:", ppl.age[:10])  # First 10 ages
else:
    print("Population object 'ppl' does not have age attribute.")

print("Available sim.results keys:", sim.results.keys())

print("Population over time (n_alive):", sim.results['n_alive'])

# -------------------------
# Debugging Aging (Manual Check)
# -------------------------
# Check initial age distribution
initial_ages = ppl.age.copy()
print("Initial ages (first 10):", initial_ages[:10])



print("Susceptible before:", sim.results['type2diabetes']['n_susceptible'][sim.ti - 1])
print("Susceptible after:", sim.results['type2diabetes']['n_susceptible'][sim.ti])

print("Susceptible before:", sim.results['type2diabetes'].get('n_susceptible', 'Not Found'))
print("Susceptible after:", sim.results['type2diabetes'].get('n_susceptible', 'Not Found'))

# import starsim as ss
# import mighti as mi  
# import pandas as pd
# import sciris as sc
# import numpy as np
# import matplotlib.pyplot as plt

# # -------------------------
# # Define Simulation Parameters
# # -------------------------

# n_agents = 500000  # Total population
# inityear = 2007  # Simulation start year
# diseases = ['HIV', 'Type2Diabetes']

# # -------------------------
# # Prevalence Data
# # -------------------------

# prevalence_data, age_bins = mi.initialize_prevalence_data(
#     diseases, csv_file_path='mighti/data/prevalence_data_eswatini.csv', inityear=inityear
# )

# # -------------------------
# # Demographics
# # -------------------------

# fertility_rates = {'fertility_rate': pd.read_csv(sc.thispath() / 'tests/test_data/eswatini_asfr.csv')}
# pregnancy = ss.Pregnancy(pars=fertility_rates)
# death_rates = {'death_rate': pd.read_csv(sc.thispath() / 'tests/test_data/eswatini_deaths.csv'), 'rate_units': 1}
# death = ss.Deaths(death_rates)
# ppl = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age_2007.csv'))

# # -------------------------
# # Networks
# # -------------------------

# mf = ss.MFNet(duration=1/24, acts=80)
# maternal = ss.MaternalNet()
# networks = [mf, maternal]

# # -------------------------
# # Disease Objects
# # -------------------------

# def get_prevalence_function(disease):
#     return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

# # Debugging Initial Susceptible Population
# print("\n--- Debugging Initial Prevalence ---")

# disease_objects = []
# for disease in ['Type2Diabetes']:
#     init_prev = ss.bernoulli(0.1351, strict=False)  # Disable strict initialization
#     init_prev.initialize()  # Explicitly initialize the distribution
#     disease_obj = mi.Type2Diabetes(init_prev=init_prev)
#     disease_objects.append(disease_obj)

#     # Debug: Generate Initial Cases Using rvs()
#     initial_cases = init_prev.rvs(n_agents, strict=False)  # Allow sampling without strict initialization    
#     print(f"Expected Initial T2D Cases: {0.1351 * n_agents:.0f}")
#     print(f"Actual Initial T2D Cases: {np.sum(initial_cases)}")  # Sum to count True cases

#     # If too few are susceptible, force correct assignment
#     if np.sum(initial_cases) < (0.1 * n_agents):  # If <10% affected, reassign manually
#         disease_obj.affected[initial_cases] = True
#         disease_obj.susceptible[~initial_cases] = True
#         print("⚠️ Manually assigned susceptible population!")

# # -------------------------
# # HIV-Specific Setup
# # -------------------------

# hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=0.001)
# disease_objects.append(hiv_disease)

# # -------------------------
# # Analyzers & Interactions
# # -------------------------

# prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)

# interaction_functions = {
#     'Type2Diabetes': mi.hiv_type2diabetes,
# }

# interactions = [interaction_functions['Type2Diabetes']()]

# # -------------------------
# # Simulation Initialization
# # -------------------------

# sim = ss.Sim(
#     n_agents=n_agents,
#     networks=networks,
#     diseases=disease_objects,
#     analyzers=[prevalence_analyzer],
#     start=inityear,
#     stop=2030,
#     connectors=interactions,
#     people=ppl,
#     demographics=[pregnancy, death],
#     copy_inputs=False
# )

# # Run Simulation
# sim.run()

# # -------------------------
# # Debugging Disease Progression
# # -------------------------

# print("\n--- Debugging Disease Progression ---")
# print(f"New Cases Per Year: {sim.results['type2diabetes']['new_cases']}")
# print(f"Remissions Per Year: {sim.results['type2diabetes']['remissions']}")
# print(f"Deaths Per Year: {sim.results['new_deaths']}")

# # -------------------------
# # Plot Prevalence Over Time
# # -------------------------

# plt.plot(sim.results['timevec'], sim.results['type2diabetes']['prevalence'], label='T2D Prevalence', color='blue')
# plt.xlabel("Year")
# plt.ylabel("Prevalence (%)")
# plt.legend()
# plt.grid()
# plt.show()

# # Define age bins and labels
# age_bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
# age_group_labels = [f'{left}-{right-1}' for left, right in zip(age_bins[:-1], age_bins[1:])]
# age_group_labels.append('80+') if age_bins[-1] == 80 else None

# # Define Young & Old Age Groups (7 bins each, ignoring 80+)
# young_age_bins = age_bins[1:8]  # First 7 age groups
# old_age_bins = age_bins[8:15]  # Next 7 age groups (ignoring 80+)

# # Function to filter prevalence data by selected age bins
# def filter_prevalence_by_age_group(prevalence_data, age_bins, selected_bins):
#     indices = [i for i, age in enumerate(age_bins[1:]) if age in selected_bins]
#     if len(indices) != len(selected_bins):
#         print(f"Warning: Expected {len(selected_bins)} bins, but found {len(indices)} in selection.")
    
#     filtered_data = prevalence_data[:, indices] if len(indices) == len(selected_bins) else np.full((prevalence_data.shape[0], len(selected_bins)), np.nan)
#     return filtered_data

# # Retrieve prevalence data for plotting
# try:
#     hiv_prevalence_data_male = prevalence_analyzer.results['HIV_prevalence_male'] * 100
#     hiv_prevalence_data_female = prevalence_analyzer.results['HIV_prevalence_female'] * 100
#     t2d_prevalence_data_male = prevalence_analyzer.results['Type2Diabetes_prevalence_male'] * 100
#     t2d_prevalence_data_female = prevalence_analyzer.results['Type2Diabetes_prevalence_female'] * 100

#     # Filter prevalence data for young and old groups
#     prevalence_filtered = {
#         'HIV': {
#             'male': {'young': filter_prevalence_by_age_group(hiv_prevalence_data_male, age_bins, young_age_bins),
#                       'old': filter_prevalence_by_age_group(hiv_prevalence_data_male, age_bins, old_age_bins)},
#             'female': {'young': filter_prevalence_by_age_group(hiv_prevalence_data_female, age_bins, young_age_bins),
#                         'old': filter_prevalence_by_age_group(hiv_prevalence_data_female, age_bins, old_age_bins)}
#         },
#         'Type2Diabetes': {
#             'male': {'young': filter_prevalence_by_age_group(t2d_prevalence_data_male, age_bins, young_age_bins),
#                       'old': filter_prevalence_by_age_group(t2d_prevalence_data_male, age_bins, old_age_bins)},
#             'female': {'young': filter_prevalence_by_age_group(t2d_prevalence_data_female, age_bins, young_age_bins),
#                         'old': filter_prevalence_by_age_group(t2d_prevalence_data_female, age_bins, old_age_bins)}
#         }
#     }

#     # Function to plot prevalence trends for young/old groups
#     def plot_prevalence_side_by_side(disease, real_data, selected_age_bins):
#         fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

#         age_group_order = ['young', 'old']  # Young (Top), Old (Bottom)
#         sex_order = ['male', 'female']  # Male (Left), Female (Right)
        
#         young_handles, young_labels = [], []
#         old_handles, old_labels = [], []

#         for row, age_group in enumerate(age_group_order):  # Young = Row 0, Old = Row 1
#             for col, sex in enumerate(sex_order):  # Male = Col 0, Female = Col 1
#                 ax = axs[row, col]
#                 prevalence_data = prevalence_filtered[disease][sex][age_group]

#                 num_age_bins = len(selected_age_bins[row])  # 7 bins for both young and old
#                 age_labels = [f"{age}-{age+4}" for age in selected_age_bins[row]]
#                 cmap = plt.get_cmap('tab10', num_age_bins)

#                 # Plot simulated prevalence trends
#                 handles = []
#                 for i, label in enumerate(age_labels):
#                     line, = ax.plot(sim.results['timevec'], prevalence_data[:, i], label=f'Estimated {label}', color=cmap(i))
#                     handles.append(line)

#                 # Store handles for **one** shared legend per row (young & old)
#                 if row == 0 and col == 0:
#                     young_handles = handles
#                     young_labels = age_labels
#                 if row == 1 and col == 0:
#                     old_handles = handles
#                     old_labels = age_labels

#                 # Overlay real data points
#                 for year in years:
#                     if year in real_data:
#                         real_values = real_data[year][sex]
#                         for age_bin in real_values:
#                             if age_bin in selected_age_bins[row]:  # Ensure valid bin
#                                 bin_idx = selected_age_bins[row].index(age_bin)
#                                 ax.scatter(year, real_values[age_bin] * 100, color=cmap(bin_idx), s=100, edgecolors='black', zorder=5)

#                 ax.set_title(f"{disease} ({sex.capitalize()}, {age_group.capitalize()})")
#                 ax.grid(True)

#         # Place young legend on the right of the first row & old legend on the right of the second row
#         legend_young = fig.legend(young_handles, young_labels, title="Young Age Groups", loc="center right", bbox_to_anchor=(1.05, 0.75), fontsize=10)
#         legend_old = fig.legend(old_handles, old_labels, title="Old Age Groups", loc="center right", bbox_to_anchor=(1.05, 0.25), fontsize=10)

#         plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust to fit legends
#         plt.show()

#     # Generate plots for each disease
#     for disease in ['HIV', 'Type2Diabetes']:
#         real_data_source = eswatini_hiv_data if disease == 'HIV' else eswatini_t2d_data
#         plot_prevalence_side_by_side(disease, real_data_source, [young_age_bins, old_age_bins])

# except KeyError as e:
#     print(f"KeyError: {e} - Check if the correct result keys are being used.")