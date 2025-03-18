import starsim as ss
import sciris as sc
import mighti as mi
import pandas as pd


# ---------------------------------------------------------------------
# Define population size and simulation timeline
# ---------------------------------------------------------------------
beta = 0.001
n_agents = 5000  # Number of agents in the simulation
inityear = 2007  # Simulation start year
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

df = pd.read_csv(csv_path_params)
df.columns = df.columns.str.strip()


# Define diseases
# conditions = ncds + communicable_diseases  # List of all diseases

ncd = ['Type2Diabetes', 'ChronicKidneyDisease'] 
diseases = ['HIV'] + ncd #+conditions # List of diseases including HIV



# Load prevalence data from the CSV file
prevalence_data_csv_path = sc.thispath() / 'mighti/data/prevalence_data_eswatini.csv'
prevalence_data_df = pd.read_csv(prevalence_data_csv_path)

# Initialize prevalence data from the DataFrame
prevalence_data, age_bins = mi.initialize_prevalence_data(diseases, prevalence_data=prevalence_data_df, inityear=inityear)

# Define a function for disease-specific prevalence
def get_prevalence_function(disease):
    return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)

# Create demographics
fertility_rates = {'fertility_rate': pd.read_csv(sc.thispath() / 'mighti/data/eswatini_asfr.csv')}
pregnancy = ss.Pregnancy(pars=fertility_rates)
death_rates = {'death_rate': pd.read_csv(sc.thispath() / 'mighti/data/eswatini_deaths.csv'), 'rate_units': 1}
death = ss.Deaths(death_rates)
ppl = ss.People(n_agents, age_data=pd.read_csv('mighti/data/eswatini_age_2023.csv'))

# Create the networks - sexual and maternal
mf = ss.MFNet(duration=1/24, acts=80)
maternal = ss.MaternalNet()
networks = [mf, maternal]

if __name__ == '__main__':
    
    hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
    
    # # Automatically create disease objects for NCDs
    # disease_objects = []
    # for disease in ncd:
    #     init_prev = ss.bernoulli(get_prevalence_function(disease))
        
    #     # Dynamically get the disease class from `mi` module
    #     disease_class = getattr(mi, disease, None)
        
    #     if disease_class:
    #         disease_obj = disease_class(init_prev=init_prev)  # Instantiate dynamically
    #         disease_objects.append(disease_obj)
    #     else:
    #         print(f"[WARNING] {disease} is not found in `mighti` module. Skipping.")
    
    # Automatically create disease objects for all diseases
    disease_objects = []
    for disease in ncd:
        init_prev = ss.bernoulli(get_prevalence_function(disease))
        
        # Dynamically get the disease class from `mi` module
        disease_class = getattr(mi, disease, None)
        
        if disease_class:
            disease_obj = disease_class(disease_name=disease, csv_path=csv_path_params, pars={"init_prev": init_prev})  # Instantiate dynamically and pass csv_path
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
    ncd_interactions = mi.read_interactions("mighti/data/rel_sus.csv")  # Reads rel_sus.csv
    connectors = mi.create_connectors(ncd_interactions)
    
    # Add NCD-NCD connectors to interactions
    interactions.extend(connectors)
     
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

    # Plot the results for each simulation
    mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'Type2Diabetes')  
    # mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'ChronicKidneyDisease')
    
    
# import starsim as ss
# import mighti as mi
# import pandas as pd


# ### TO DO


# # ---------------------------------------------------------------------
# # Define population size and simulation timeline
# # ---------------------------------------------------------------------
# beta = 0.001
# n_agents = 5000  # Number of agents in the simulation
# inityear = 2017  # Simulation start year
# endyear = 2050

# # ---------------------------------------------------------------------
# # Specify data file paths
# # ---------------------------------------------------------------------

# # Parameters
# csv_path_params =  'mighti/data/eswatini_parameters.csv'

# # Relative Risks
# csv_path_interactions = "mighti/data/rel_sus.csv"

# # Prevalence data
# csv_prevalence = 'mighti/data/prevalence_data_eswatini.csv'

# # Fertility data 
# csv_path_fertility = 'mighti/data/eswatini_asfr.csv'

# # Death data
# csv_path_death = 'mighti/data/eswatini_deaths.csv'

# # Age distribution data
# csv_path_age = 'mighti/data/eswatini_age_2023.csv'


# # Load the CSV file
# df = pd.read_csv(csv_path_params)
# df.columns = df.columns.str.strip()

# healthconditions = ['Type2Diabetes', 'ChronicKidneyDisease'] 
# diseases = ['HIV'] + healthconditions #+conditions # List of diseases including HIV


# # Load prevalence data from the CSV file
# prevalence_data_df = pd.read_csv(csv_prevalence)

# # Initialize prevalence data from the DataFrame
# prevalence_data, age_bins = mi.initialize_prevalence_data(diseases, prevalence_data=prevalence_data_df, inityear=inityear)

# # Define a function for disease-specific prevalence
# def get_prevalence_function(disease):
#     return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

# # Initialize the PrevalenceAnalyzer
# prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)

# # Create demographics
# fertility_rates = {'fertility_rate': pd.read_csv(csv_path_fertility)}
# pregnancy = ss.Pregnancy(pars=fertility_rates)
# death_rates = {'death_rate': pd.read_csv(csv_path_death), 'rate_units': 1}
# death = ss.Deaths(death_rates)
# ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))

# # Create the networks - sexual and maternal
# mf = ss.MFNet(duration=1/24, acts=80)
# maternal = ss.MaternalNet()
# networks = [mf, maternal]

# if __name__ == '__main__':
    
#     hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
    
#     # # Automatically create disease objects for NCDs
#     # disease_objects = []
#     # for disease in ncd:
#     #     init_prev = ss.bernoulli(get_prevalence_function(disease))
        
#     #     # Dynamically get the disease class from `mi` module
#     #     disease_class = getattr(mi, disease, None)
        
#     #     if disease_class:
#     #         disease_obj = disease_class(init_prev=init_prev)  # Instantiate dynamically
#     #         disease_objects.append(disease_obj)
#     #     else:
#     #         print(f"[WARNING] {disease} is not found in `mighti` module. Skipping.")
    
#     # Automatically create disease objects for all diseases
#     disease_objects = []
#     for disease in healthconditions:
#         init_prev = ss.bernoulli(get_prevalence_function(disease))
        
#         # Dynamically get the disease class from `mi` module
#         disease_class = getattr(mi, disease, None)
        
#         if disease_class:
#             disease_obj = disease_class(disease_name=disease, csv_path=csv_path_params, pars={"init_prev": init_prev})  # Instantiate dynamically and pass csv_path
#             disease_objects.append(disease_obj)
#         else:
#             print(f"[WARNING] {disease} is not found in `mighti` module. Skipping.")


#     # Combine all disease objects including HIV
#     disease_objects.append(hiv_disease)
    
#     # Initialize interaction objects for HIV-NCD interactions
#     ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
#     ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
#     interactions = [ncd_hiv_connector]
    
#     # Load NCD-NCD interactions
#     ncd_interactions = mi.read_interactions("mighti/data/rel_sus.csv")  # Reads rel_sus.csv
#     connectors = mi.create_connectors(ncd_interactions)
    
#     # Add NCD-NCD connectors to interactions
#     interactions.extend(connectors)
     
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
#     mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'Type2Diabetes')  
#     # mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'ChronicKidneyDisease')    