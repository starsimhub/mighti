import starsim as ss
import sciris as sc
import mighti as mi
import pandas as pd
import matplotlib.pyplot as plt

# import sys
# log_file = open("debug_output.txt", "w")
# sys.stdout = log_file  # Redirects all print outputs to this file

# Define diseases
ncd = ['Type2Diabetes'] 
diseases = ['HIV'] + ncd  # List of diseases including HIV
beta = 0.0001  # Transmission probability for HIV
n_agents = 500000  # Number of agents in the simulation
inityear = 2007  # Simulation start year
endyear = 2050
# 
# Initialize prevalence data from a CSV file
prevalence_data, age_bins = mi.initialize_prevalence_data(diseases, csv_file_path='mighti/data/prevalence_data_eswatini.csv', inityear=inityear)


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


    
# if __name__ == '__main__':
    
#     hiv = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
#     t2d = mi.Type2Diabetes(init_prev=ss.bernoulli(get_prevalence_function('Type2Diabetes')))
#     interactions = mi.HIVType2DiabetesConnector()
    
#     # Initialize the first simulation without connectors
#     sim1 = ss.Sim(
#         n_agents=n_agents,
#         networks=networks,
#         diseases=[hiv, t2d],
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
#         diseases=[hiv, t2d],
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
#     msim.plot()
    




if __name__ == '__main__':
    
    hiv = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
    t2d = mi.Type2Diabetes(init_prev=ss.bernoulli(get_prevalence_function('Type2Diabetes')))
    interactions = mi.HIVType2DiabetesConnector()
    
    # Initialize the simulation with connectors
    sim = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        diseases=[hiv, t2d],
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
mi.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'HIV')    

# mi.plot_numerator_denominator(sim, prevalence_analyzer, 'HIV')
# mi.plot_numerator_denominator(sim, prevalence_analyzer, 'Type2Diabetes')