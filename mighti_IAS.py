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



                
# Extract data from results
timevec = sim.results['timevec']  # Simulation time
age_groups = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
sexes = ['male', 'female']

# Prepare plot
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

for row, sex in enumerate(sexes):  # Male (row 0), Female (row 1)
    for col, hiv_status in enumerate(['plhiv', 'hivneg']):  # PLHIV (col 0), HIV-negative (col 1)
        ax = axs[row, col]
        
        for i in range(len(age_groups) - 1):
            age_min, age_max = age_groups[i], age_groups[i+1]
            key_name = f'T2D_prevalence_{sex}_{age_min}_{age_max}'

            if key_name in sim.results:
                prevalence = sim.results[key_name] * 100  # Convert to percentage
                ax.plot(timevec, prevalence, label=f'{age_min}-{age_max}', alpha=0.8)

        ax.set_title(f'T2D Prevalence ({sex.capitalize()}, {hiv_status.upper()})')
        ax.set_xlabel('Year')
        ax.set_ylabel('Prevalence (%)')
        ax.legend(title="Age Group", loc="upper right", fontsize=8)
        ax.grid(True)

plt.tight_layout()
plt.show()