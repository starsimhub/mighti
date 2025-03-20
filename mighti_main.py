import starsim as ss
import mighti as mi
import pandas as pd
import matplotlib.pyplot as plt

### TO DO


# ---------------------------------------------------------------------
# Define population size and simulation timeline
# ---------------------------------------------------------------------
beta = 0.001
n_agents = 5000  # Number of agents in the simulation
inityear = 2017  # Simulation start year
endyear = 2030

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
df = pd.read_csv(csv_path_params)
df.columns = df.columns.str.strip()

# Extract all conditions except HIV
healthconditions = [condition for condition in df.condition if condition != "HIV"]
# healthconditions = ['Type2Diabetes']
# 
# Combine with HIV
diseases = ["HIV"] + healthconditions

# Filter the DataFrame for disease_class being 'ncd'
ncd_df = df[df["disease_class"] == "ncd"]

# Extract disease categories from the filtered DataFrame
chronic = ncd_df[ncd_df["disease_type"] == "chronic"]["condition"].tolist()
acute = ncd_df[ncd_df["disease_type"] == "acute"]["condition"].tolist()
remitting = ncd_df[ncd_df["disease_type"] == "remitting"]["condition"].tolist()

# Extract communicable diseases with disease_class as 'sis'
communicable_diseases = df[df["disease_class"] == "sis"]["condition"].tolist()

# Initialize disease models with preloaded data
mi.initialize_conditions(df, chronic, acute, remitting, communicable_diseases)

# ---------------------------------------------------------------------
# Initialize conditions, prevalence analyzer, and interactions
# ---------------------------------------------------------------------
prevalence_data_df = pd.read_csv(csv_prevalence)

prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases, prevalence_data=prevalence_data_df, inityear=inityear
)

# Define a function for disease-specific prevalence
def get_prevalence_function(disease):
    return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=healthconditions, 
                                            communicable_diseases = communicable_diseases)


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
# Disease Conditions
# -------------------------

hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)

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
        
# Combine all disease objects including HIV
disease_objects.append(hiv_disease)


# -------------------------
# Disease Interactions
# -------------------------

# Initialize interaction objects for HIV-NCD interactions
ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
interactions = [ncd_hiv_connector]

# Load NCD-NCD interactions
ncd_interactions = mi.read_interactions(csv_path_interactions)  # Reads rel_sus.csv
connectors = mi.create_connectors(ncd_interactions, communicable_diseases)

# Add NCD-NCD connectors to interactions
interactions.extend(connectors)

# if __name__ == '__main__':
    
#     # Initialize the simulation with connectors
#     sim = ss.Sim(
#         n_agents=n_agents,
#         networks=networks,
#         diseases=disease_objects,
#         analyzers=[prevalence_analyzer],
#         start=inityear,
#         stop=endyear,
#         people=ppl,
#         demographics=[pregnancy, death],
#         connectors=interactions,
#         copy_inputs=False,
#         label='Connector'
#     )

#     # Run the simulation
#     sim.run()

#     # Plot the results for each simulation
#     mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'HIV')  
#     mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'CervicalCancer')      
#     mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'ProstateCancer')  
#     mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'Type2Diabetes')  
#     mi.plot_age_dependent_prevalence(sim, prevalence_analyzer, 'Type2Diabetes', age_bins)
#     mi.plot_age_dependent_prevalence(sim, prevalence_analyzer, 'HIV', age_bins)


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

    # # Plot the results for each simulation
    # mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'HIV')  
    # mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'CervicalCancer')      
    # mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'ProstateCancer')  
    # mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'Type2Diabetes')  
    # mi.plot_age_dependent_prevalence(sim, prevalence_analyzer, 'Type2Diabetes', age_bins)
    # mi.plot_age_dependent_prevalence(sim, prevalence_analyzer, 'HIV', age_bins)

    # Perform CEA if needed
    cost_data_path = 'mighti/data/cost_data.csv'
    utility_data_path = 'mighti/data/utility_data.csv'
    cost_data = pd.read_csv(cost_data_path)
    utility_data = pd.read_csv(utility_data_path)
    
    # Perform CEA
    cea_results = mi.cea.perform_cea(prevalence_analyzer, cost_data, utility_data)
    mi.plot_cea_results(cea_results)
    
# if __name__ == '__main__':

    
#     # Initialize the first simulation without connectors
#     sim1 = ss.Sim(
#         n_agents=n_agents,
#         networks=networks,
#         diseases=disease_objects,
#         analyzers=[prevalence_analyzer],
#         start=inityear,
#         stop=endyear,
#         people=ppl,
#         demographics=[pregnancy, death],
#         copy_inputs=False,
#         label='Separate'
#     )

#     # Initialize the second simulation with connectors
#     sim2 = ss.Sim(
#         n_agents=n_agents,
#         networks=networks,
#         diseases=disease_objects,
#         analyzers=[prevalence_analyzer],
#         start=inityear,
#         stop=endyear,
#         people=ppl,
#         demographics=[pregnancy, death],
#         connectors=interactions,
#         copy_inputs=False,
#         label='Connector'
#     )

#     # Run the simulations in parallel
#     msim = ss.parallel(sim1, sim2)

    
# # # Plot the results for each simulation
# mi.plot_mean_prevalence_two_diseases_parallel(msim, ['HIV','Type2Diabetes'])    