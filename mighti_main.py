import starsim as ss
import mighti as mi  
import pylab as pl
import pandas as pd
import sciris as sc


# Define diseases
ncds = ['Type2Diabetes', 'Obesity']  # List of NCDs being modeled
diseases = ['HIV'] + ncds  # List of diseases including HIV
beta = 0.001  # Transmission probability for HIV
n_agents = 5000  # Number of agents in the simulation
inityear = 2007  # Simulation start year

# Initialize prevalence data from a CSV file
prevalence_data, age_bins = mi.initialize_prevalence_data(diseases, csv_file_path='mighti/data/prevalence_data_eswatini.csv', inityear=inityear)

# Create demographics
fertility_rates = {'fertility_rate': pd.read_csv(sc.thispath() / 'tests/test_data/eswatini_asfr.csv')}
pregnancy = ss.Pregnancy(pars=fertility_rates)
death_rates = {'death_rate': pd.read_csv(sc.thispath() / 'tests/test_data/eswatini_deaths.csv'), 'units': 1}
death = ss.Deaths(death_rates)
ppl = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age.csv'))

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


# Define interventions with costs, coverage, and target populations
interventions = {
    "HIV Treatment": {"cost": 500, "coverage": 0.8, "target": "hiv_infected"},
    "Diabetes Management": {"cost": 200, "coverage": 0.6, "target": "type2diabetes_affected"}
}

# Add the cost analyzer
cost_analyzer = mi.CostAnalyzer(interventions=interventions)


# Initialize the simulation
sim = ss.Sim(
    n_agents=n_agents,
    networks=networks,
    diseases=disease_objects,  # Pass the full list of diseases (HIV + NCDs)
    analyzers=[prevalence_analyzer],
    start=inityear,
    end=2020,
    connectors=interactions,  # Both HIV-NCD and NCD-NCD interactions
    people=ppl,
    demographics=[pregnancy, death],
    copy_inputs=False
)


# Run the simulation
sim.run()

eswatini_hiv_data_2007 = {
    'male': {
        0:0, 15: 0.018569463, 20: 0.123878438, 25: 0.277081792, 30: 0.437388675, 35: 0.446666475, 40: 0.408951061, 45: 0.279480401,
        50: 0.274580983, 55: 0.203923873484658, 60: 0.170053298709714, 65: 0.159627035, 70: 0.102792944, 75: 0.089528285, 80: 0.054545808
    },
    'female': {
        0:0, 15: 0.100228343, 20: 0.383694318, 25: 0.491161255, 30: 0.4503177, 35: 0.375698852, 40: 0.276657489, 45: 0.215261524,
        50: 0.186609436, 55: 0.145316219, 60: 0.089981987, 65: 0.088771152, 70: 0.071297853, 75: 0.052671538, 80: 0.020405424
    }
}

eswatini_hiv_data_2011 = {
    'male': {
        0:0, 15: 0.008153899, 20: 0.066462848, 25: 0.212564961, 30: 0.365625757, 35: 0.469877829, 40: 0.454610745, 45: 0.424535086,
        50: 0.417092794, 55: 0.309763542, 60: 0.258313612, 65: 0.242475955, 70: 0.156144084, 75 : 0.135994861, 80: 0.082855933
    },
    'female': {
        0:0, 15: 0.143296247, 20: 0.314870622, 25: 0.467469388, 30: 0.537866378, 35: 0.491198813, 40: 0.397114544, 45: 0.316176211,
        50: 0.274092013, 55: 0.21344052, 60: 0.132165578, 65: 0.130387103, 70: 0.104722314, 75: 0.077363976, 80: 0.029971495
    }
}

eswatini_hiv_data_2017 = {
    'male': {
        0:0, 15: 0.039203323, 20: 0.042302217, 25: 0.132586775, 30: 0.281448958, 35: 0.419039192, 40: 0.432948219, 45: 0.487936995, 
        50: 0.418901763, 55: 0.3178861, 60: 0.319024067, 65: 0.187721072, 70: 0.16612866, 75:0.096966056, 80:0.068853558
    },
    'female': {
        0:0, 15: 0.071699336, 20: 0.208942316, 25: 0.374644094, 30: 0.506826329, 35: 0.542306801, 40: 0.51927347, 45: 0.423031977,
        50: 0.361306241, 55: 0.295168377, 60: 0.222713574, 65: 0.103696162, 70: 0.089371202, 75: 0.068701597, 80: 0.007520089
    }
}

eswatini_hiv_data_2021 = {
    'male': {
        0:0, 15: 0.030000309, 20: 0.038736453, 25: 0.054007305, 30: 0.191607831, 35: 0.269162468, 40: 0.385042512, 45: 0.500416774,
        50: 0.491644241, 55: 0.365130888, 60: 0.304484761, 65: 0.28581627, 70: 0.184053384, 75: 0.160302675, 80: 0.097665658
    },
    'female': {
        0:0, 15: 0.055804134, 20: 0.171564278, 25: 0.302792519, 30: 0.424996426, 35: 0.524670263, 40: 0.571575379, 45: 0.501290462,
        50: 0.434566888, 55: 0.338405273, 60: 0.209545631, 65: 0.206725899, 70: 0.166034939, 75: 0.122658891, 80: 0.047519148
    }
}


# Define real data for 2007, 2011, 2017, and 2021
eswatini_hiv_data = {
    '2007': eswatini_hiv_data_2007,
    '2011': eswatini_hiv_data_2011,
    '2017': eswatini_hiv_data_2017,
    '2021': eswatini_hiv_data_2021
}

diseases = ['HIV', 'Type2Diabetes','Obesity']


# Retrieve the prevalence data for plotting
try:
    hiv_prevalence_data_male = prevalence_analyzer.results['HIV_prevalence_male'] * 100
    hiv_prevalence_data_female = prevalence_analyzer.results['HIV_prevalence_female'] * 100
    # diabetes1_prevalence_data_male = prevalence_analyzer.results['Type1Diabetes_prevalence_male'] * 100
    # diabetes1_prevalence_data_female = prevalence_analyzer.results['Type1Diabetes_prevalence_female'] * 100
    diabetes2_prevalence_data_male = prevalence_analyzer.results['Type2Diabetes_prevalence_male'] * 100
    diabetes2_prevalence_data_female = prevalence_analyzer.results['Type2Diabetes_prevalence_female'] * 100
    obesity_prevalence_data_male = prevalence_analyzer.results['Obesity_prevalence_male'] * 100
    obesity_prevalence_data_female = prevalence_analyzer.results['Obesity_prevalence_female'] * 100
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



# Define utilities (example QALY/DALY weights)
utilities = {
    "HIV": {"qaly": 0.8},
    "Type2Diabetes": {"qaly": 0.7},
}

# Run post-simulation analysis
cea = mi.CostEffectivenessAnalyzer(interventions=interventions, utilities=utilities)
cea.calculate_costs(sim)
cea.calculate_outcomes(sim)
cea.calculate_icer(baseline_cost=1000, baseline_qaly=50)  # Example baseline
cea.summarize_results()

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