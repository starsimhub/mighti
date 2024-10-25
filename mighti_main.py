import starsim as ss
import mighti as mi  
import pylab as pl
import pandas as pd
import sciris as sc
import numpy as np


# Define diseases
ncds = ['Type1Diabetes', 'Type2Diabetes', 'Obesity', 'Hypertension',
    'Depression','Accident', 'Alzheimers', 'Assault', 'CerebrovascularDisease', 
    'ChronicLiverDisease','ChronicLowerRespiratoryDisease', 'HeartDisease', 
    'ChronicKidneyDisease','Flu','HPV',
    'CervicalCancer','ColorectalCancer', 'BreastCancer', 'LungCancer', 'ProstateCancer', 'OtherCancer', 
    'Parkinsons','Smoking', 'Alcohol', 'BRCA', 'ViralHepatitis', 'Poverty']

diseases = ['HIV'] + ncds  # List of diseases including HIV
beta = 0.001  # Transmission probability for HIV
n_agents = 50000  # Number of agents in the simulation
inityear = 2007  # Simulation start year

# Initialize prevalence data from a CSV file
prevalence_data, age_bins = mi.initialize_prevalence_data(diseases, csv_file_path='mighti/data/prevalence_data_eswatini.csv', inityear=inityear)

# Create demographics
fertility_rates = {'fertility_rate': pd.read_csv(sc.thispath() / 'tests/test_data/eswatini_asfr.csv')}
pregnancy = ss.Pregnancy(pars=fertility_rates)
death_rates = {'death_rate': pd.read_csv(sc.thispath() / 'tests/test_data/eswatini_deaths.csv'), 'units': 1}
death = ss.Deaths(death_rates)
ppl = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age_2007.csv'))

# Create the networks - sexual and maternal
mf = ss.MFNet(duration=1/24, acts=80)
maternal = ss.MaternalNet()
networks = [mf, maternal]

# Define a function for disease-specific prevalence
def get_prevalence_function(disease):
    return lambda module, sim, size: mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)

# Create disease objects
disease_objects = []
for disease in ncds:
    init_prev = ss.bernoulli(get_prevalence_function(disease))
    # Dynamically create the disease object using getattr
    disease_class = getattr(mi, disease)  # Access the disease class dynamically
    disease_obj = disease_class(init_prev=init_prev)  # Instantiate the class
    disease_objects.append(disease_obj)


# HIV-specific setup
hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
disease_objects.append(hiv_disease)

# Validate prevalence data for each disease
def validate_prevalence_data(prevalence_data, diseases):
    for disease in diseases:
        if disease not in prevalence_data:
            raise ValueError(f"Prevalence data missing for disease: {disease}")
        if 'male' not in prevalence_data[disease] or 'female' not in prevalence_data[disease]:
            raise ValueError(f"Male/Female data missing for disease: {disease}")
        if not prevalence_data[disease]['male'] or not prevalence_data[disease]['female']:
            raise ValueError(f"Age bins missing for disease: {disease}")
        print(f"Prevalence data validated for disease: {disease}")

# Validate the prevalence data
validate_prevalence_data(prevalence_data, diseases)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)

# Automatically create interaction functions for each NCD dynamically
interaction_functions = {}

for disease in ncds:
    interaction_func_name = f'hiv_{disease.lower()}'  # Assuming function names follow the 'hiv_<disease>' pattern
    if hasattr(mi, interaction_func_name):  # Check if the function exists in mighti
        interaction_func = getattr(mi, interaction_func_name)  # Dynamically get the interaction function
        interaction_functions[disease] = interaction_func

# Initialize interaction objects for HIV-NCD interactions
interactions = []
for disease in ncds:
    interaction_obj = interaction_functions[disease]()  # Call the corresponding function
    interactions.append(interaction_obj)

# NCD-NCD interactions from the interaction matrix
interaction_matrix = mi.read_interactions()
ncd_connectors = []
for condition1, interactions_dict in interaction_matrix.items():
    for condition2, rel_risk in interactions_dict.items():
        name = f'{condition1}_{condition2}'
        ncd_connectors.append(mi.GenericNCDConnector(condition1, condition2, relative_risk=rel_risk, name=name))

# Combine both HIV-NCD and NCD-NCD interaction objects
all_connectors = interactions + ncd_connectors  # Now both are lists

# Initialize the simulation
sim = ss.Sim(
    n_agents=n_agents,
    networks=networks,
    diseases=disease_objects,  # Pass the full list of diseases (HIV + NCDs)
    analyzers=[prevalence_analyzer],
    start=inityear,
    end=2024,
    connectors=all_connectors,  
    people=ppl,
    demographics=[pregnancy, death],
    copy_inputs=False
)


# Run the simulation
sim.run()

df = pd.read_csv('mighti/data/prevalence_data_eswatini.csv')

eswatini_hiv_data_2007 = {
    'male': dict(zip(df[df['Year'] == 2007]['Age'], df[df['Year'] == 2007]['HIV_male'])),
    'female': dict(zip(df[df['Year'] == 2007]['Age'], df[df['Year'] == 2007]['HIV_female']))
}

eswatini_hiv_data_2011 = {
    'male': dict(zip(df[df['Year'] == 2011]['Age'], df[df['Year'] == 2011]['HIV_male'])),
    'female': dict(zip(df[df['Year'] == 2011]['Age'], df[df['Year'] == 2011]['HIV_female']))
}

eswatini_hiv_data_2017 = {
    'male': dict(zip(df[df['Year'] == 2017]['Age'], df[df['Year'] == 2017]['HIV_male'])),
    'female': dict(zip(df[df['Year'] == 2017]['Age'], df[df['Year'] == 2017]['HIV_female']))
}

eswatini_hiv_data_2021 = {
    'male': dict(zip(df[df['Year'] == 2021]['Age'], df[df['Year'] == 2021]['HIV_male'])),
    'female': dict(zip(df[df['Year'] == 2021]['Age'], df[df['Year'] == 2021]['HIV_female']))
}


# Define real data for 2007, 2011, 2017, and 2021
eswatini_hiv_data = {
    '2007': eswatini_hiv_data_2007,
    '2011': eswatini_hiv_data_2011,
    '2017': eswatini_hiv_data_2017,
    '2021': eswatini_hiv_data_2021
}

diseases = ['HIV', 'Type2Diabetes', 'Type1Diabetes', 'Obesity', 'Hypertension']


# Retrieve the prevalence data for plotting
try:
    hiv_prevalence_data_male = prevalence_analyzer.results['HIV_prevalence_male'] * 100
    hiv_prevalence_data_female = prevalence_analyzer.results['HIV_prevalence_female'] * 100
    store_m = hiv_prevalence_data_male[0]
    store_f = hiv_prevalence_data_female[0]

    print(f"Simulated data for 2007 (male): {hiv_prevalence_data_male[0]}")
    print(f"Simulated data for 2007 (female): {hiv_prevalence_data_female[0]}")

    diabetes1_prevalence_data_male = prevalence_analyzer.results['Type1Diabetes_prevalence_male'] * 100
    diabetes1_prevalence_data_female = prevalence_analyzer.results['Type1Diabetes_prevalence_female'] * 100
    diabetes2_prevalence_data_male = prevalence_analyzer.results['Type2Diabetes_prevalence_male'] * 100
    diabetes2_prevalence_data_female = prevalence_analyzer.results['Type2Diabetes_prevalence_female'] * 100
    obesity_prevalence_data_male = prevalence_analyzer.results['Obesity_prevalence_male'] * 100
    obesity_prevalence_data_female = prevalence_analyzer.results['Obesity_prevalence_female'] * 100
    hypertension_prevalence_data_male = prevalence_analyzer.results['Hypertension_prevalence_male'] * 100
    hypertension_prevalence_data_female = prevalence_analyzer.results['Hypertension_prevalence_female'] * 100

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

    # Real data points for the years
    real_data_years = {
        2007: eswatini_hiv_data_2007,
        2011: eswatini_hiv_data_2011,
        2017: eswatini_hiv_data_2017,
        2021: eswatini_hiv_data_2021,
    }

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
            for year, real_data in real_data_years.items():
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

#config file


# Real data (converted to array from dictionary)
real_data_male_2007 = np.array(list(eswatini_hiv_data_2007['male'].values())) * 100

# Get the order (ranking) of elements in ascending order
store_m_order = np.argsort(store_m)  # Returns indices that would sort store_m
real_data_order = np.argsort(real_data_male_2007)  # Returns indices that would sort real_data_male_2007

# Print the orders
print("Order of elements in store_m (ascending):", store_m_order)
print("Order of elements in real_data_male_2007 (ascending):", real_data_order)

# Optionally, reverse for descending order
store_m_order_desc = np.argsort(-store_m)
real_data_order_desc = np.argsort(-real_data_male_2007)

print("Order of elements in store_m (descending):", store_m_order_desc)
print("Order of elements in real_data_male_2007 (descending):", real_data_order_desc)



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