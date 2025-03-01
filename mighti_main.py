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

n_agents = 5000  # Number of agents in the simulation
inityear = 2007  # Simulation start year
endyear = 2025

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
print(df_interactions.head())  # Print first few rows
print(df_interactions.columns)  # Check column names

# Extract all conditions except HIV
# healthconditions = [condition for condition in df_params.index if condition != "HIV"]
healthconditions = ['Type2Diabetes','Obesity']
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

# # **Explicitly generate HIV-NCD interactions**
# hiv_ncd_connectors = mi.create_hiv_connectors(df_params)  # Explicitly pass df_params
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

# ---------------------------------------------------------------------
# Generate Plots
# ---------------------------------------------------------------------

# Specify which diseases to plot
selected_diseases = ['HIV', 'Type2Diabetes']

# Call the plotting function from `plot_functions.py`
mi.plot_disease_prevalence(sim, prevalence_analyzer, selected_diseases, eswatini_hiv_data, age_bins)