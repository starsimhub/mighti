import starsim as ss
import sciris as sc
import mighti as mi
import matplotlib.pyplot as plt
import pandas as pd


### TO DO


# ---------------------------------------------------------------------
# Define population size and simulation timeline
# ---------------------------------------------------------------------
beta = 0.001
n_agents = 50000  # Number of agents in the simulation
inityear = 2017  # Simulation start year
endyear = 2050

# ---------------------------------------------------------------------
# Specify data file paths
# ---------------------------------------------------------------------

# Parameters
csv_path_params =  'mighti/data/eswatini_parameters.csv'

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

# Extract all conditions except HIV
healthconditions = [condition for condition in df_params.index if condition != "HIV"]
# healthconditions = ['Type2Diabetes']
# 
# Combine with HIV
diseases = ["HIV"] + healthconditions

# Ensure column names are trimmed to remove extra spaces
df = pd.read_csv(csv_path_params)
df.columns = df.columns.str.strip()

# Extract disease categories
ncds = df[df["disease_class"] == "ncd"]["condition"].tolist()
communicable_diseases = df[df["disease_class"] == "sis"]["condition"].tolist()

# ---------------------------------------------------------------------
# Initialize conditions, prevalence analyzer, and interactions
# ---------------------------------------------------------------------
# Initialize disease models with preloaded data
mi.initialize_conditions(df, ncds, communicable_diseases)


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

hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
disease_objects.append(hiv_disease)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)

# -------------------------
# HIV-NCD Interaction
# -------------------------

# Load existing HIV and NCD interactions
interaction_functions = {
    'Type1Diabetes': mi.hiv_type1diabetes,
    'Type2Diabetes': mi.hiv_type2diabetes,
    'Obesity': mi.hiv_obesity,
    'Hypertension': mi.hiv_hypertension,
    'CardiovascularDiseases': mi.hiv_cardiovasculardiseases,
    'ChronicKidneyDisease': mi.hiv_chronickidneydisease,
    'Hyperlipidemia': mi.hiv_hyperlipidemia,
    'CervicalCancer': mi.hiv_cervicalcancer,
    'ColorectalCancer': mi.hiv_colorectalcancer,
    'BreastCancer': mi.hiv_breastcancer,
    'LungCancer': mi.hiv_lungcancer,
    'ProstateCancer': mi.hiv_prostatecancer,
    'AlcoholUseDisorder': mi.hiv_alcoholusedisorder,
    'TobaccoUse': mi.hiv_tobaccouse,
    'HIVAssociatedDementia': mi.hiv_hivassociateddementia,
    'PTSD': mi.hiv_ptsd,
    'Depression': mi.hiv_depression,
    'HPV': mi.hiv_hpv,
    'Flu': mi.hiv_flu,
    'ViralHepatitis': mi.hiv_viralhepatitis,
    'DomesticViolence': mi.hiv_domesticviolence,
    'RoadInjuries': mi.hiv_roadinjuries,
    'ChronicLiverDisease': mi.hiv_chronicliverdisease,
    'Asthma': mi.hiv_asthma,
    'COPD': mi.hiv_copd,
    'AlzheimersDisease': mi.hiv_alzheimersdisease,
    'ParkinsonsDisease': mi.hiv_parkinsonsdisease,
}

# # Initialize interaction objects for HIV-NCD interactions
interactions = []
for disease in healthconditions:
    interaction_obj = interaction_functions[disease]()  # Call the corresponding function
    interactions.append(interaction_obj)
    
# Load NCD-NCD interactions
ncd_interactions = mi.read_interactions()  # Reads rel_sus.csv

for condition1, interactions_dict in ncd_interactions.items():
    for condition2, relative_risk in interactions_dict.items():
        if condition1 != condition2:  # Avoid self-interactions
            interaction_obj = mi.GenericNCDConnector(condition1, condition2, relative_risk)
            interactions.append(interaction_obj)

# Store disease categories in simulation parameters
sim_pars = {
    "communicable_diseases": communicable_diseases,  
    "ncds": ncds  
}


if __name__ == '__main__':
    
    # hiv = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
    # t2d = mi.Type2Diabetes(init_prev=ss.bernoulli(get_prevalence_function('Type2Diabetes')))
    # interactions = mi.HIVType2DiabetesConnector()
    # interactions = mi.HIVObesityConnector()
    
    # Initialize the first simulation without connectors
    sim1 = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        diseases=disease_objects,
        analyzers=[prevalence_analyzer],
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        copy_inputs=False,
        label='Separate'
    )

    # Initialize the second simulation with connectors
    sim2 = ss.Sim(
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

    # Run the simulations in parallel
    msim = ss.parallel(sim1, sim2)
    # msim.plot()
    
# # Plot the results for each simulation
mi.plot_mean_prevalence_two_diseases_parallel(msim, ['HIV', 'Type2Diabetes'])

# sim = ss.Sim(
#     n_agents=n_agents,
#     networks=networks,
#     diseases=disease_objects,  
#     analyzers=[prevalence_analyzer],
#     start=inityear,
#     stop=endyear,
#     connectors=interactions,  
#     people=ppl,
#     demographics=[pregnancy, death],
#     copy_inputs=False
# )

# sim.run()
# mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'Obesity')    

# ---------------------------------------------------------------------
# Generate Plots
# ---------------------------------------------------------------------

# Call the plotting function from `plot_functions.py`
# selected_diseases = ['HIV','Type2Diabetes','CervicalCancer']
# mi.plot_disease_prevalence(sim, prevalence_analyzer, selected_diseases, eswatini_hiv_data, age_bins)
# mi.plot_mean_prevalence(sim, prevalence_analyzer, 'CervicalCancer')
# mi.plot_mean_prevalence(sim, prevalence_analyzer, 'ProstateCancer')
# mi.plot_mean_prevalence(sim, prevalence_analyzer, 'Type2Diabetes')
# mi.plot_mean_prevalence(sim, prevalence_analyzer, 'HIV')
# mi.plot_mean_prevalence(sim, prevalence_analyzer, 'Hypertension')
# mi.analyze_hiv_ncd_prevalence(sim, prevalence_analyzer, 'Type2Diabetes')
