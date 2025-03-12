import starsim as ss
import mighti as mi  
import pylab as pl
import pandas as pd
import sciris as sc
import numpy as np
# import sys
# log_file = open("debug_output.txt", "w")
# sys.stdout = log_file  # Redirects all print outputs to this file


### TO DO
# Working on the minimal model



# Define diseases
ncd = ['Type2Diabetes'] 
diseases = ['HIV'] + ncd  # List of diseases including HIV
beta = 0.001  # Transmission probability for HIV
n_agents = 5000  # Number of agents in the simulation
inityear = 2007  # Simulation start year
endyear = 2020

# Initialize prevalence data from a CSV file
prevalence_data, age_bins = mi.initialize_prevalence_data(diseases, csv_file_path='mighti/data/prevalence_data_eswatini.csv', inityear=inityear)

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

# Define a function for disease-specific prevalence
def get_prevalence_function(disease):
    return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

# Create disease object for Type2Diabetes
disease_objects = [
    mi.Type2Diabetes(init_prev=ss.bernoulli(get_prevalence_function('Type2Diabetes')))
]

# HIV-specific setup
hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
disease_objects.append(hiv_disease)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)

# Load existing HIV and NCD interactions
interaction_functions = {
    'Type2Diabetes': mi.hiv_type2diabetes,
}

# Initialize interaction objects for HIV-NCD interactions
interactions = []
for disease in ncd:
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


# Run the simulation
sim.run()


mi.plot_mean_prevalence_two_diseases(sim, prevalence_analyzer, diseases)
