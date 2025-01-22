import starsim as ss
import mighti as mi  
import pylab as pl
import pandas as pd
import sciris as sc

# Define diseases
    # 'Type1Diabetes', 'Type2Diabetes', 'Obesity', 'Hypertension',
    # 'Depression','Accident', 'Alzheimers', 'Assault', 'CerebrovascularDisease',
    # 'ChronicLiverDisease','ChronicLowerRespiratoryDisease', 'HeartDisease',
    # 'ChronicKidneyDisease','Flu','HPV',
    # 'CervicalCancer','ColorectalCancer', 'BreastCancer', 'LungCancer', 'ProstateCancer', 'OtherCancer',
    # 'Parkinsons','Smoking', 'Alcohol', 'BRCA', 'ViralHepatitis', 'Poverty'

ncds = [
      'Type2Diabetes',#'Obesity', 'Type1Diabetes',#'Hypertension', #
    # 'Depression','Alzheimers', 'Parkinsons','PTSD','HIVAssociatedDementia',
    # 'CerebrovascularDisease','ChronicLiverDisease','Asthma', 'IschemicHeartDisease',
    # 'TrafficAccident','DomesticViolence','TobaccoUse', 'AlcoholUseDisorder', 
    # 'ChronicKidneyDisease','Flu','HPVVaccination',
    # 'ViralHepatitis','COPD','Hyperlipidemia',
    # 'CervicalCancer','ColorectalCancer', 'BreastCancer', 'LungCancer', 'ProstateCancer', 'OtherCancer',
]

diseases = ['HIV'] + ncds

beta = 0.001  # Transmission probability for HIV
n_agents = 5000  # Number of agents in the simulation
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
        ['HIV'], csv_file_path='mighti/data/prevalence_data_eswatini.csv', inityear=year
    )
    eswatini_hiv_data[year] = hiv_prevalence_data['HIV']


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

# Run the simulation
sim.run()





# Retrieve the prevalence data for plotting
try:
    hiv_prevalence_data_male = prevalence_analyzer.results['HIV_prevalence_male'] * 100
    hiv_prevalence_data_female = prevalence_analyzer.results['HIV_prevalence_female'] * 100
    # diabetes1_prevalence_data_male = prevalence_analyzer.results['Type1Diabetes_prevalence_male'] * 100
    # diabetes1_prevalence_data_female = prevalence_analyzer.results['Type1Diabetes_prevalence_female'] * 100
    diabetes2_prevalence_data_male = prevalence_analyzer.results['Type2Diabetes_prevalence_male'] * 100
    diabetes2_prevalence_data_female = prevalence_analyzer.results['Type2Diabetes_prevalence_female'] * 100
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
            axs[disease_idx, 0].plot(sim.yearvec, male_data[:, i], label=label, color=age_bin_colors[label])
        axs[disease_idx, 0].set_title(f'{disease} (Male)', fontsize=24) 
        axs[disease_idx, 0].set_xlabel('Year', fontsize=20) 
        axs[disease_idx, 0].set_ylabel('Prevalence (%)', fontsize=20)  
        axs[disease_idx, 0].tick_params(axis='both', labelsize=18)  
        axs[disease_idx, 0].grid(True)

        # Plot female prevalence for the disease
        for i, label in enumerate(age_group_labels):
            axs[disease_idx, 1].plot(sim.yearvec, female_data[:, i], color=age_bin_colors[label])
        axs[disease_idx, 1].set_title(f'{disease} (Female)', fontsize=24) 
        axs[disease_idx, 1].set_xlabel('Year', fontsize=20)  
        axs[disease_idx, 1].tick_params(axis='both', labelsize=18) 
        axs[disease_idx, 1].grid(True)

        # Add real data points for HIV for the specific years
        if disease == 'HIV':
            for year, real_data in eswatini_hiv_data.items():
                real_male_data = real_data['male']
                real_female_data = real_data['female']

                # Plot real data points for males
                for age_bin in real_male_data:
                    age_label = f'{age_bin}-99' if age_bin == 80 else f'{age_bin}-{age_bin + 4}'
                    if age_label in age_bin_colors:  # Check if the age label exists
                        axs[disease_idx, 0].scatter(year, real_male_data[age_bin] * 100, color=age_bin_colors[age_label], s=100, zorder=5)

                # Plot real data points for females
                for age_bin in real_female_data:
                    age_label = f'{age_bin}-99' if age_bin == 80 else f'{age_bin}-{age_bin + 4}'
                    if age_label in age_bin_colors:  # Check if the age label exists
                        axs[disease_idx, 1].scatter(year, real_female_data[age_bin] * 100, color=age_bin_colors[age_label], s=100, zorder=5)

    # Add a single common legend with two rows
    handles, labels = axs[0, 0].get_legend_handles_labels()  # Get labels from one axis
    
    # Adjust ncol to ensure the legend is split into two rows
    fig.legend(handles, labels, title='Age Groups', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(age_group_labels) // 2, fontsize=12)
    
    # Adjust layout and show the plot
    pl.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the legend at the bottom
    pl.show()

except KeyError as e:
    print(f"KeyError: {e} - Check if the correct result keys are being used.")



# # Plots without dots for data
# try:
#     hiv_prevalence_data_male = prevalence_analyzer.results['HIV_prevalence_male'] * 100
#     hiv_prevalence_data_female = prevalence_analyzer.results['HIV_prevalence_female'] * 100
#     diabetes_prevalence_data_male = prevalence_analyzer.results['Type1Diabetes_prevalence_male'] * 100
#     diabetes_prevalence_data_female = prevalence_analyzer.results['Type1Diabetes_prevalence_female'] * 100
#     diabetes_prevalence_data_male = prevalence_analyzer.results['Type2Diabetes_prevalence_male'] * 100
#     diabetes_prevalence_data_female = prevalence_analyzer.results['Type2Diabetes_prevalence_female'] * 100
#     obesity_prevalence_data_male = prevalence_analyzer.results['Obesity_prevalence_male'] * 100
#     obesity_prevalence_data_female = prevalence_analyzer.results['Obesity_prevalence_female'] * 100
#     hypertension_prevalence_data_male = prevalence_analyzer.results['Hypertension_prevalence_male'] * 100
#     hypertension_prevalence_data_female = prevalence_analyzer.results['Hypertension_prevalence_female'] * 100

#     # Ensure age_bins is a list (fix for the previous error)
#     age_bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
#     age_bins_list = list(age_bins)  # Convert to a list if it's not already
    
#     # Create subplots for each disease, dynamically based on the number of diseases
#     n_diseases = len(diseases)
#     fig, axs = pl.subplots(n_diseases, 2, figsize=(18, n_diseases * 6), sharey='row')

#     # Create age group labels and color map for age bins (generalized)
#     # Ensure age_bins_list contains integers
#     age_bins_list = [int(age_bin) for age_bin in age_bins_list]  # Convert age bins to integers
    
#     # Now you can perform operations like subtraction
#     age_group_labels = [f'{left}-{right-1}' for left, right in zip(age_bins_list[:-1], age_bins_list[1:])]  
    
#     if age_bins_list[-1] == 80:
#         age_group_labels.append('80+')
#     cmap = pl.get_cmap('tab20', len(age_group_labels))  # Color map for distinct age groups
#     age_bin_colors = {label: cmap(i) for i, label in enumerate(age_group_labels)}

#       # Loop through each disease and plot its prevalence for males and females
#     for disease_idx, disease in enumerate(diseases):
#         # Access the male and female prevalence data for each disease
#         male_data = prevalence_analyzer.results[f'{disease}_prevalence_male'] * 100
#         female_data = prevalence_analyzer.results[f'{disease}_prevalence_female'] * 100

#         # Plot male prevalence for the disease
#         for i, label in enumerate(age_group_labels):
#             axs[disease_idx, 0].plot(sim.yearvec, male_data[:, i], label=label, color=age_bin_colors[label])
#         axs[disease_idx, 0].set_title(f'{disease} (Male)', fontsize=24) 
#         axs[disease_idx, 0].set_xlabel('Year', fontsize=20) 
#         axs[disease_idx, 0].set_ylabel('Prevalence (%)', fontsize=20)  
#         axs[disease_idx, 0].tick_params(axis='both', labelsize=18)  
#         axs[disease_idx, 0].grid(True)

#         # Plot female prevalence for the disease
#         for i, label in enumerate(age_group_labels):
#             axs[disease_idx, 1].plot(sim.yearvec, female_data[:, i], color=age_bin_colors[label])
#         axs[disease_idx, 1].set_title(f'{disease} (Female)', fontsize=24) 
#         axs[disease_idx, 1].set_xlabel('Year', fontsize=20)  
#         axs[disease_idx, 0].tick_params(axis='both', labelsize=18) 
#         axs[disease_idx, 1].grid(True)

#     # Add a single common legend with two rows
#     handles, labels = axs[0, 0].get_legend_handles_labels()  # Get labels from one axis
    
#     # Adjust ncol to ensure the legend is split into two rows
#     fig.legend(handles, labels, title='Age Groups', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(age_group_labels) // 2, fontsize=12)
    
#     # Adjust layout and show the plot
#     pl.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the legend at the bottom
#     pl.show()
    
# except KeyError as e:
#     print(f"KeyError: {e} - Check if the correct result keys are being used.")
