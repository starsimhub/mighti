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



# ---------------------------------------------------------------------
# Generate Plots
# ---------------------------------------------------------------------

# Specify which diseases to plot
selected_diseases = ['HIV', 'Type2Diabetes']

# Call the plotting function from `plot_functions.py`
mi.plot_disease_prevalence(sim, prevalence_analyzer, selected_diseases, eswatini_hiv_data, age_bins)


import matplotlib.pyplot as plt

# Assuming you have these lists recorded during the simulation
time_steps = list(range(len(sim.results['n_alive'])))  # Extract time steps
total_population = sim.results['n_alive']  # Total population at each time step
deaths = sim.results.get('new_deaths', [0] * len(time_steps))  # Deaths per step

# Estimate births
births = [total_population[0]]  # First time step has no previous value
for t in range(2, len(time_steps)):
    births.append(total_population[t] - total_population[t-1] + deaths[t])

# Create figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# First panel: Total Population
axes[0].plot(time_steps, total_population, label='Total Population', linewidth=2)
axes[0].set_ylabel('Total Population')
axes[0].set_title('Total Population Over Time')
axes[0].legend()
axes[0].grid()

# Second panel: Estimated Births and Deaths
axes[1].plot(time_steps, births, label='Estimated Births', linestyle='dashed')
axes[1].plot(time_steps, deaths, label='Deaths', linestyle='dotted')
axes[1].set_xlabel('Time Steps')
axes[1].set_ylabel('Count')
axes[1].set_title('Estimated Births and Deaths Over Time')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()