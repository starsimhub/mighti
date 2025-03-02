import starsim as ss
import mighti as mi  
import pylab as pl
import pandas as pd
import sciris as sc

### TO DO
# Check population change 
# Check if interactins are working

# ---------------------------------------------------------------------
# Define population size and simulation timeline
# ---------------------------------------------------------------------
beta = 0.001
n_agents = 5000  # Number of agents in the simulation
inityear = 2017  # Simulation start year
endyear = 2050

# ---------------------------------------------------------------------
# Specify data file paths
# ---------------------------------------------------------------------

# Parameters
csv_path_params = "mighti/data/eswatini_parameters.csv"

# Relative Risks
csv_path_interactions = "mighti/data/rel_sus.csv"

# Prevalence data
csv_prevalence = 'mighti/data/prevalence_data_eswatini.csv'

# Fertility data 
csv_path_fertility = 'mighti/data/eswatini_asfr.csv'

# Death data
csv_path_death = 'mighti/data/eswatini_deaths.csv'

# Age distribution data
csv_path_age = 'mighti/data/eswatini_age_2023.csv'

####### You do not need to modify anything below unless making custom changes #####

# ---------------------------------------------------------------------
# Load health conditions to include in the simulation
# ---------------------------------------------------------------------

# Read disease parameter file and interactions file
df_params = pd.read_csv(csv_path_params, index_col="condition")
df_interactions = pd.read_csv(csv_path_interactions)
# print(df_interactions.head())  # Print first few rows
# print(df_interactions.columns)  # Check column names

# Extract all conditions except HIV
# healthconditions = [condition for condition in df_params.index if condition != "HIV"]
healthconditions = ['Type2Diabetes']

# Combine with HIV
diseases = ["HIV"] + healthconditions

# ---------------------------------------------------------------------
# Initialize conditions, prevalence analyzer, and interactions
# ---------------------------------------------------------------------

# Initialize disease models with preloaded data
mi.initialize_conditions(df_params)

# Initialize prevalence analyzer with preloaded data
mi.initialize_prevalence_analyzer(df_params)

# Initialize interactions with preloaded data
# mi.initialize_interactions(df_interactions)

# ---------------------------------------------------------------------
# Load disease prevalence data
# ---------------------------------------------------------------------

prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases, csv_file_path=csv_prevalence, inityear=inityear
)

years = [2007, 2011, 2017, 2021]
eswatini_hiv_data = {}
for year in years:
    hiv_prevalence_data, _ = mi.initialize_prevalence_data(
        diseases=['HIV'], 
        csv_file_path=csv_prevalence, 
        inityear=year
    )
    eswatini_hiv_data[year] = hiv_prevalence_data['HIV']  # Store data for the specific year

eswatini_t2d_data = {}
for year in years:
    t2d_prevalence_data, _ = mi.initialize_prevalence_data(
        diseases=['Type2Diabetes'], 
        csv_file_path=csv_prevalence, 
        inityear=year
    )
    eswatini_t2d_data[year] = t2d_prevalence_data['Type2Diabetes']

# -------------------------
# Demographics
# -------------------------

fertility_rates = {'fertility_rate': pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rates)
death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
death = ss.Deaths(death_rates)
ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))

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

# Automatically create disease objects
disease_objects = []
for disease in healthconditions:
    init_prev = ss.bernoulli(get_prevalence_function(disease))
    
    # Dynamically get the disease class from `mi` module
    disease_class = getattr(mi, disease, None)
    
    if disease_class:
        disease_obj = disease_class(init_prev=init_prev)  # Instantiate dynamically
        disease_objects.append(disease_obj)
    else:
        print(f"[WARNING] {disease} is not found in `mighti` module. Skipping.")

# # Create disease objects
# disease_objects = []
# for disease in healthconditions:
#     init_prev = ss.bernoulli(get_prevalence_function(disease))
#     if disease == 'Type2Diabetes':
#         disease_obj = mi.Type2Diabetes(init_prev=init_prev)
#     elif disease == 'Obesity':
#         disease_obj = mi.Obesity(init_prev=init_prev)
#     disease_objects.append(disease_obj)

# HIV-specific setup
beta = 0.001  # Transmission probability for HIV
hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
disease_objects.append(hiv_disease)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)

# -------------------------
# Create HIV-NCD Connectors Dynamically
# -------------------------

# interactions = [] 

# # # **Explicitly generate HIV-NCD interactions**
# hiv_ncd_connectors = mi.create_hiv_connectors()  # Explicitly pass df_params
# interactions.extend(hiv_ncd_connectors)

# # Debugging: Show interactions created
# print(f"Final interactions added: {interactions}")  


# Load existing HIV and NCD interactions
interaction_functions = {
    'Type2Diabetes': mi.hiv_type2diabetes,
    'Obesity': mi.hiv_obesity,
}

# Initialize interaction objects for HIV-NCD interactions
interactions = []
for disease in healthconditions:
    interaction_obj = interaction_functions[disease]()  # Call the corresponding function
    interactions.append(interaction_obj)
print(f"Final interactions added ({len(interactions)}): {[i.label for i in interactions]}")


# -------------------------
# Initialize the simulation
# -------------------------

sim = ss.Sim(
    n_agents=n_agents,
    networks=networks,
    diseases=disease_objects,  # Pass the full list of diseases (HIV + NCDs)
    analyzers=[prevalence_analyzer],
    start=inityear,
    stop=endyear,
    connectors=interactions,  # **Now includes HIV-NCD interactions**
    people=ppl,
    demographics=[pregnancy, death],
    copy_inputs=False
)

    
# Run the simulation
sim.run()
print(sim.results.keys())  # See what data is stored



# # ---------------------------------------------------------------------
# # Generate Plots
# # ---------------------------------------------------------------------

# # Specify which diseases to plot
# selected_diseases = ['HIV', 'Type2Diabetes']

# # Call the plotting function from `plot_functions.py`
# mi.plot_disease_prevalence(sim, prevalence_analyzer, selected_diseases, eswatini_hiv_data, age_bins)


# # Assuming you have these lists recorded during the simulation
# time_steps = list(range(len(sim.results['n_alive'])))  # Extract time steps
# total_population = sim.results['n_alive']  # Total population at each time step
# deaths = sim.results.get('new_deaths', [0] * len(time_steps))  # Deaths per step

# # Estimate births (skip first year in calculation)
# births = [total_population[t] - total_population[t-1] + deaths[t] for t in range(1, len(time_steps))]

# # Adjust time steps and population to match births (skip first year)
# time_steps = time_steps[1:]
# total_population = total_population[1:]
# deaths = deaths[1:]

# mi.plot_demography(time_steps, total_population, deaths, births)

# import matplotlib.pyplot as plt
# import numpy as np

# # Extract male and female prevalence matrices
# male_data = prevalence_analyzer.results.get('HIV_prevalence_male', None)
# female_data = prevalence_analyzer.results.get('HIV_prevalence_female', None)

# # Ensure data exists
# if male_data is None or female_data is None:
#     print("[ERROR] No HIV prevalence data available.")
# else:
#     # Compute mean prevalence across all age groups
#     mean_prevalence_male = np.mean(male_data, axis=1) * 100  # Convert to percentage
#     mean_prevalence_female = np.mean(female_data, axis=1) * 100

#     # Create figure
#     plt.figure(figsize=(10, 5))

#     # Plot mean prevalence for males and females
#     plt.plot(sim.timevec, mean_prevalence_male, label='Male HIV Prevalence', linewidth=2, color='blue')
#     plt.plot(sim.timevec, mean_prevalence_female, label='Female HIV Prevalence', linewidth=2, color='red')

#     # Labels and title
#     plt.xlabel('Year')
#     plt.ylabel('HIV Prevalence (%)')
#     plt.title('Mean HIV Prevalence Over Time (All Ages)')
#     plt.legend()
#     plt.grid()

#     plt.show()
# Plots without dots for data
try:
    hiv_prevalence_data_male = prevalence_analyzer.results['HIV_prevalence_male'] * 100
    hiv_prevalence_data_female = prevalence_analyzer.results['HIV_prevalence_female'] * 100
    # diabetes_prevalence_data_male = prevalence_analyzer.results['Type1Diabetes_prevalence_male'] * 100
    # diabetes_prevalence_data_female = prevalence_analyzer.results['Type1Diabetes_prevalence_female'] * 100
    diabetes_prevalence_data_male = prevalence_analyzer.results['Type2Diabetes_prevalence_male'] * 100
    diabetes_prevalence_data_female = prevalence_analyzer.results['Type2Diabetes_prevalence_female'] * 100
    # obesity_prevalence_data_male = prevalence_analyzer.results['Obesity_prevalence_male'] * 100
    # obesity_prevalence_data_female = prevalence_analyzer.results['Obesity_prevalence_female'] * 100
    # hypertension_prevalence_data_male = prevalence_analyzer.results['Hypertension_prevalence_male'] * 100
    # hypertension_prevalence_data_female = prevalence_analyzer.results['Hypertension_prevalence_female'] * 100

    # Ensure age_bins is a list (fix for the previous error)
    age_bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    age_bins_list = list(age_bins)  # Convert to a list if it's not already
    
    # Create subplots for each disease, dynamically based on the number of diseases
    n_diseases = len(diseases)
    fig, axs = pl.subplots(n_diseases, 2, figsize=(18, n_diseases * 6), sharey='row')

    # Create age group labels and color map for age bins (generalized)
    # Ensure age_bins_list contains integers
    age_bins_list = [int(age_bin) for age_bin in age_bins_list]  # Convert age bins to integers
    
    # Now you can perform operations like subtraction
    age_group_labels = [f'{left}-{right-1}' for left, right in zip(age_bins_list[:-1], age_bins_list[1:])]  
    
    if age_bins_list[-1] == 80:
        age_group_labels.append('80+')
    cmap = pl.get_cmap('tab20', len(age_group_labels))  # Color map for distinct age groups
    age_bin_colors = {label: cmap(i) for i, label in enumerate(age_group_labels)}

      # Loop through each disease and plot its prevalence for males and females
    for disease_idx, disease in enumerate(diseases):
        # Access the male and female prevalence data for each disease
        male_data = prevalence_analyzer.results[f'{disease}_prevalence_male'] * 100
        female_data = prevalence_analyzer.results[f'{disease}_prevalence_female'] * 100

        # Plot male prevalence for the disease
        for i, label in enumerate(age_group_labels):
            axs[disease_idx, 0].plot(sim.timevec, male_data[:, i], label=label, color=age_bin_colors[label])
        axs[disease_idx, 0].set_title(f'{disease} (Male)', fontsize=24) 
        axs[disease_idx, 0].set_xlabel('Year', fontsize=20) 
        axs[disease_idx, 0].set_ylabel('Prevalence (%)', fontsize=20)  
        axs[disease_idx, 0].tick_params(axis='both', labelsize=18)  
        axs[disease_idx, 0].grid(True)

        # Plot female prevalence for the disease
        for i, label in enumerate(age_group_labels):
            axs[disease_idx, 1].plot(sim.timevec, female_data[:, i], color=age_bin_colors[label])
        axs[disease_idx, 1].set_title(f'{disease} (Female)', fontsize=24) 
        axs[disease_idx, 1].set_xlabel('Year', fontsize=20)  
        axs[disease_idx, 0].tick_params(axis='both', labelsize=18) 
        axs[disease_idx, 1].grid(True)

    # Add a single common legend with two rows
    handles, labels = axs[0, 0].get_legend_handles_labels()  # Get labels from one axis
    
    # Adjust ncol to ensure the legend is split into two rows
    fig.legend(handles, labels, title='Age Groups', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(age_group_labels) // 2, fontsize=12)
    
    # Adjust layout and show the plot
    pl.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the legend at the bottom
    pl.show()
    
except KeyError as e:
    print(f"KeyError: {e} - Check if the correct result keys are being used.")