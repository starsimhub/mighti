# Imports
import starsim as ss
import mighti as mi  # For handling NCDs like depression, diabetes, etc.
import pylab as pl
import pandas as pd
import sciris as sc
from prevalence_analyzer import PrevalenceAnalyzer
from disease_definitions import initialize_prevalence_data, age_sex_dependent_prevalence

# from disease_definitions import (age_sex_dependent_prevalence_hiv, 
#                                  age_sex_dependent_prevalence_depression,
#                                  hiv_age_bins, prevalence_data)  # Include prevalence_data
# Define diseases
ncds = ['Diabetes', 'Obesity', 'Hypertension']
diseases = ['HIV'] + ncds  # List of diseases including HIV
beta = 0.001  # Transmission probability for HIV
n_agents = 50000

prevalence_data, age_bins = initialize_prevalence_data(diseases, csv_file_path='mighti/data/prevalence_data_eswatini.csv')

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
    return lambda module, sim, size: age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)


# Initialize the diseases with the correct prevalence functions
disease_objects = []
for disease in ncds:
    init_prev = ss.bernoulli(get_prevalence_function(disease))
    if disease == 'Diabetes':
        disease_obj = mi.Diabetes(init_prev=init_prev)
    elif disease == 'Obesity':
        disease_obj = mi.Obesity(init_prev=init_prev)
    elif disease == 'Hypertension':
        disease_obj = mi.Hypertension(init_prev=init_prev)
    disease_objects.append(disease_obj)

# HIV-specific setup
hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
disease_objects.append(hiv_disease)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer = PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)

# Define a dictionary that maps disease names to corresponding interaction functions
interaction_functions = {
    'Diabetes': mi.hiv_diabetes,
    'Obesity': mi.hiv_obesity,
    'Hypertension': mi.hiv_hypertension
}

# Initialize an empty list to store the interaction objects
interactions = []

# Loop through NCDs and dynamically generate interactions by calling functions from the dictionary
for disease in ncds:
    interaction_obj = interaction_functions[disease]()  # Call the corresponding function
    interactions.append(interaction_obj)

sim = ss.Sim(
    n_agents=n_agents,
    networks=networks,
    diseases=disease_objects,  # Pass the full list of diseases (HIV + NCDs)
    analyzers=[prevalence_analyzer],
    start=1987,
    end=2008,
    connectors=interactions,
    people=ppl,
    demographics=[pregnancy,death],
    copy_inputs=False
)

# Run the simulation
sim.run()





# Ensure all age bins are included, even if simulation doesn't produce data for them
def fill_missing_bins(age_bins, simulated_data):
    filled_data = {}
    for age in age_bins:
        if age in simulated_data:
            filled_data[age] = simulated_data[age]
        else:
            filled_data[age] = 0  # Fill missing ages with 0, or another placeholder value
    return filled_data


# Ensure age_bins is a list (fix for the previous error)
age_bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
age_bins_list = list(age_bins)  # Convert to a list if it's not already

hiv_prevalence_data_male = prevalence_analyzer.results['HIV_prevalence_male'] * 100
hiv_prevalence_data_female = prevalence_analyzer.results['HIV_prevalence_female'] * 100

# Create a dictionary with the simulated prevalence results
simulated_hiv_prevalence_male = dict(zip(age_bins, hiv_prevalence_data_male))
simulated_hiv_prevalence_female = dict(zip(age_bins, hiv_prevalence_data_female))


# Prevalence data from KENPHIA 2018 for HIV
kenphia_hiv_data = {
    'male': {0: 0.003, 5: 0.005, 10: 0.013, 15: 0.005, 20: 0.006, 25: 0.022, 30: 0.032, 35: 0.043, 40: 0.063, 45: 0.083, 50: 0.066, 55: 0.059, 60: 0.056   
    },
    'female': {
        0: 0.005, 5: 0.011, 10: 0.008, 15: 0.012, 20: 0.034, 25: 0.060, 30: 0.095, 35: 0.087, 40: 0.119, 45: 0.106, 50: 0.117, 55: 0.090, 60: 0.062 
    }
}

kenya_depression_emod = {
    'male':{
        0:0, 18: 0.03, 25: 0.069, 30: 0.098, 40: 0.087, 45: 0.11, 50: 0.13, 60:0.2
        },
    'female':{
        0:0, 18: 0.065, 25: 0.086, 30: 0.067, 40: 0.12, 45: 0.13, 50: 0.079, 60:0.051
        }
}

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






# Extract the data of an interested year from the simulation
def get_year_index(sim, year):
    """ Get the index of a specific year in the simulation's year vector """
    return list(sim.yearvec).index(year)


# Function to automatically generate age group labels from the common_age_bins
def generate_age_group_labels(age_bins):
    age_group_labels = []
    for i in range(len(age_bins) - 1):
        age_group_labels.append(f"{age_bins[i]}-{age_bins[i+1]}")
    
    # Treat the last age bin as "80+" if it's 80
    if age_bins[-1] == 80:
        age_group_labels.append("80+")
    else:
        age_group_labels.append(f"{age_bins[-1]}-{age_bins[-1]+5}")
    
    return age_group_labels


# Modified function to plot two comparisons between input and simulated data in one figure
def plot_comparison_two_datasets(input_data_1, simulated_data_1, input_data_2, simulated_data_2, 
                                 year, ylabel):
    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(14, 6), sharey=True)  # sharey=True to share y-axis

    # Define the width of the bars
    bar_width = 0.35

    # For the first subplot
    common_age_bins_1 = sorted(set(input_data_1.keys()).intersection(simulated_data_1.keys()))
    input_values_1 = [input_data_1[age] * 100 for age in common_age_bins_1]  # Convert to percentages
    simulated_values_1 = [simulated_data_1[age] for age in common_age_bins_1]  # Already in percentages

    bar_positions_input_1 = range(len(common_age_bins_1))
    bar_positions_simulated_1 = [x + bar_width for x in bar_positions_input_1]

    # Plot first comparison for males
    ax1.bar(bar_positions_input_1, input_values_1, width=bar_width, label='Input Data', color='lightblue')
    ax1.bar(bar_positions_simulated_1, simulated_values_1, width=bar_width, label='Simulated Data', color='seagreen')

    ax1.set_title(f"Male in {year}")
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel(f"{ylabel} Prevalence (%)")
    ax1.set_xticks([r + bar_width / 2 for r in bar_positions_input_1])

    # Automatically generate age group labels based on bins
    age_group_labels_1 = generate_age_group_labels(common_age_bins_1)
    ax1.set_xticklabels(age_group_labels_1)

    ax1.grid(True)

    # For the second subplot (for females)
    common_age_bins_2 = sorted(set(input_data_2.keys()).intersection(simulated_data_2.keys()))
    input_values_2 = [input_data_2[age] * 100 for age in common_age_bins_2]  # Convert to percentages
    simulated_values_2 = [simulated_data_2[age] for age in common_age_bins_2]  # Already in percentages

    bar_positions_input_2 = range(len(common_age_bins_2))
    bar_positions_simulated_2 = [x + bar_width for x in bar_positions_input_2]

    # Plot second comparison for females
    ax2.bar(bar_positions_input_2, input_values_2, width=bar_width, label='Input Data', color='lightblue')
    ax2.bar(bar_positions_simulated_2, simulated_values_2, width=bar_width, label='Simulated Data', color='seagreen')

    ax2.set_title(f"Female in {year}")
    ax2.set_xlabel('Age Group')
    ax2.set_xticks([r + bar_width / 2 for r in bar_positions_input_2])

    # Automatically generate age group labels based on bins
    age_group_labels_2 = generate_age_group_labels(common_age_bins_2)
    ax2.set_xticklabels(age_group_labels_2)

    ax2.grid(True)

    # Add legend only once, after both plots
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    pl.tight_layout()
    pl.show()
    
    

year = 2007

hiv_data = eswatini_hiv_data_2007


# Extract HIV prevalence for male and female 
year_index = get_year_index(sim, year)
hiv_prevalence_data_male = prevalence_analyzer.results['HIV_prevalence_male'][year_index, :] * 100
hiv_prevalence_data_female = prevalence_analyzer.results['HIV_prevalence_female'][year_index, :] * 100

# Create a dictionary with the simulated prevalence results for 2004
simulated_hiv_prevalence_male = dict(zip(age_bins, hiv_prevalence_data_male))
simulated_hiv_prevalence_female = dict(zip(age_bins, hiv_prevalence_data_female))

# Use data of the initial prevalences
# input_hiv_prevalence_male = prevalence_data['HIV']['male']
# input_hiv_prevalence_female = prevalence_data['HIV']['female']


# Use data ablove as the input data for comparison
input_hiv_prevalence_male = hiv_data['male'] 
input_hiv_prevalence_female = hiv_data['female'] 


# Placeholder for age bins (as used in the simulation)
data_age_bins = list(hiv_data['male'].keys())


def fill_missing_age_bins(input_data, simulated_data, all_age_bins):
    """Ensure all age bins are present in both input and simulated data, filling missing simulated bins with 0."""
    filled_simulated_data = {}
    for age in all_age_bins:
        if age in simulated_data:
            filled_simulated_data[age] = simulated_data[age]
        else:
            # Add a placeholder value (e.g., 0) for the missing age bins
            filled_simulated_data[age] = 0  # Or use float('nan') for NaN if desired
    
    # Handle the 80+ case specifically
    if 80 in simulated_data:
        filled_simulated_data[80] = simulated_data[80]  # 80 means 80+
    elif 80 in input_data:
        filled_simulated_data[80] = input_data[80]  # Add 80+ if it's in the input data but not simulated

    return filled_simulated_data

# Define the full set of age bins (from the input data)
all_age_bins_male = list(input_hiv_prevalence_male.keys())  # Get all age bins from input data
all_age_bins_female = list(input_hiv_prevalence_female.keys())  # Get all age bins from input data

# Fill missing age bins for both male and female simulated data
simulated_hiv_prevalence_male_filled = fill_missing_age_bins(input_hiv_prevalence_male, simulated_hiv_prevalence_male, all_age_bins_male)
simulated_hiv_prevalence_female_filled = fill_missing_age_bins(input_hiv_prevalence_female, simulated_hiv_prevalence_female, all_age_bins_female)



# Now plot the comparison with the filled data
plot_comparison_two_datasets(input_hiv_prevalence_male, simulated_hiv_prevalence_male_filled, 
                              input_hiv_prevalence_female, simulated_hiv_prevalence_female_filled, 
                              year, 'HIV')
