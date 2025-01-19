import starsim as ss
import mighti as mi  
import pandas as pd
import pylab as pl
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


# Define diseases
ncds = ['Type2Diabetes']
diseases = ['HIV'] + ncds
beta = 0.0005
n_agents = 500000
inityear = 2007
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
fertility_rates = {'fertility_rate': pd.read_csv('tests/test_data/eswatini_asfr.csv')}
pregnancy = ss.Pregnancy(pars=fertility_rates)
death_rates = {'death_rate': pd.read_csv('tests/test_data/eswatini_deaths.csv'), 'units': 1}
death = ss.Deaths(death_rates)
ppl = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age.csv'))

# -------------------------
# Networks
# -------------------------
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


    # Initialize later, within the simulation context
    if disease == 'Type2Diabetes':
        disease_obj = mi.Type2Diabetes(init_prev=init_prev)
    elif disease == 'Obesity':
        disease_obj = mi.Obesity(init_prev=init_prev)

    disease_objects.append(disease_obj)
    
# HIV-specific setup
hiv_disease = ss.HIV(init_prev=ss.bernoulli(get_prevalence_function('HIV')), beta=beta)
disease_objects.append(hiv_disease)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer_with = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)
prevalence_analyzer_without = mi.PrevalenceAnalyzer(prevalence_data=prevalence_data, diseases=diseases)

# Load existing HIV and NCD interactions
interaction_functions = {
    'Type2Diabetes': mi.hiv_type2diabetes,
    'Obesity': mi.hiv_obesity,
}


interactions = [
    mi.hiv_type2diabetes(pars={"rel_sus_hiv_type2diabetes": 10})
]

for interaction in interactions:
    print(interaction)  # Should print the interaction object

# -------------------------
# Simulation With Interactions
# -------------------------
ppl_with_interactions = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age.csv'))
networks_with_interactions = [deepcopy(net) for net in networks]
pregnancy_with_interactions = ss.Pregnancy(pars=fertility_rates)
death_with_interactions = ss.Deaths(death_rates)

disease_objects_with_interactions = deepcopy(disease_objects)


sim_with_interactions = ss.Sim(
    n_agents=n_agents,
    networks=networks_with_interactions,
    diseases=disease_objects_with_interactions,
    analyzers=[prevalence_analyzer_with],
    start=inityear,
    end=endyear,
    connectors=interactions,
    people=ppl_with_interactions,
    demographics=[pregnancy_with_interactions, death_with_interactions],
    copy_inputs=False
)


# print(f"rel_sus for Type2Diabetes before simulation: {disease_objects_with_interactions[0].rel_sus[:10]}")
sim_with_interactions.run()
# print(f"rel_sus for Type2Diabetes after simulation: {disease_objects_with_interactions[0].rel_sus[:10]}")

 # Retrieve the prevalence data for plotting
try:
    hiv_prevalence_data_male = prevalence_analyzer_with.results['HIV_prevalence_male'] * 100
    hiv_prevalence_data_female = prevalence_analyzer_with.results['HIV_prevalence_female'] * 100
    t2d_prevalence_data_male = prevalence_analyzer_with.results['Type2Diabetes_prevalence_male'] * 100
    t2d_prevalence_data_female = prevalence_analyzer_with.results['Type2Diabetes_prevalence_female'] * 100

    # Define age bins
    age_bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    age_group_labels = [f'{left}-{right-1}' for left, right in zip(age_bins[:-1], age_bins[1:])]
    age_group_labels.append('80+') if age_bins[-1] == 80 else None

    
    # Define Young & Old Age Groups (7 bins each, ignoring 80+)
    young_age_bins = age_bins[1:8]  # First 7 age groups
    old_age_bins = age_bins[8:15]  # Next 7 age groups (ignoring 80+)
    
    # Function to filter prevalence data by selected age bins
    def filter_prevalence_by_age_group(prevalence_data, age_bins, selected_bins):
        indices = [i for i, age in enumerate(age_bins[1:]) if age in selected_bins]
    
        # Ensure the selected bins match expected shape
        if len(indices) != len(selected_bins):
            print(f"Warning: Expected {len(selected_bins)} bins, but found {len(indices)} in selection.")
    
        # Extract data or fill with NaN if missing
        filtered_data = prevalence_data[:, indices] if len(indices) == len(selected_bins) else np.full((prevalence_data.shape[0], len(selected_bins)), np.nan)
    
        return filtered_data
    
    # Filter prevalence data for young and old groups
    prevalence_filtered = {
        'HIV': {
            'male': {'young': filter_prevalence_by_age_group(hiv_prevalence_data_male, age_bins, young_age_bins),
                     'old': filter_prevalence_by_age_group(hiv_prevalence_data_male, age_bins, old_age_bins)},
            'female': {'young': filter_prevalence_by_age_group(hiv_prevalence_data_female, age_bins, young_age_bins),
                       'old': filter_prevalence_by_age_group(hiv_prevalence_data_female, age_bins, old_age_bins)}
        },
        'Type2Diabetes': {
            'male': {'young': filter_prevalence_by_age_group(t2d_prevalence_data_male, age_bins, young_age_bins),
                     'old': filter_prevalence_by_age_group(t2d_prevalence_data_male, age_bins, old_age_bins)},
            'female': {'young': filter_prevalence_by_age_group(t2d_prevalence_data_female, age_bins, young_age_bins),
                       'old': filter_prevalence_by_age_group(t2d_prevalence_data_female, age_bins, old_age_bins)}
        }
    }
    
    # Function to plot prevalence trends for young/old groups
    def plot_prevalence_side_by_side(disease, real_data, selected_age_bins):
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    
        # New order: Young (Top), Old (Bottom)
        age_group_order = ['young', 'old']
        sex_order = ['male', 'female']  # Male = Left, Female = Right
    
        # Separate handles for young & old legends
        young_handles, young_labels = [], []
        old_handles, old_labels = [], []
    
        for row, age_group in enumerate(age_group_order):  # Young row 0, Old row 1
            for col, sex in enumerate(sex_order):  # Male col 0, Female col 1
                ax = axs[row, col]
                prevalence_data = prevalence_filtered[disease][sex][age_group]
    
                num_age_bins = len(selected_age_bins[row])  # 7 bins for both young and old
                age_labels = [f"{age}-{age+4}" for age in selected_age_bins[row]]
                cmap = plt.get_cmap('tab10', num_age_bins)
    
                # Plot simulated prevalence trends
                handles = []
                for i, label in enumerate(age_labels):
                    line, = ax.plot(sim_with_interactions.yearvec, prevalence_data[:, i], label=f'Estimated {label}', color=cmap(i))
                    handles.append(line)
    
                # Store handles for **one** shared legend per row (young & old)
                if row == 0 and col == 0:  # Store only once for young
                    young_handles = handles
                    young_labels = age_labels
                if row == 1 and col == 0:  # Store only once for old
                    old_handles = handles
                    old_labels = age_labels
    
                # Overlay real data points
                for year in years:
                    if year in real_data:
                        real_values = real_data[year][sex]
                        for age_bin in real_values:
                            if age_bin in selected_age_bins[row]:  # Ensure valid bin
                                bin_idx = selected_age_bins[row].index(age_bin)
                                ax.scatter(year, real_values[age_bin] * 100, color=cmap(bin_idx), s=100, edgecolors='black', zorder=5)
    
                # Formatting
                ax.set_title(f"{disease} ({sex.capitalize()}, {age_group.capitalize()})")
                ax.grid(True)
    
        # **Final fix: Place young legend on the right of the first row & old legend on the right of the second row**
        legend_young = fig.legend(young_handles, young_labels, title="Young Age Groups", loc="center right", bbox_to_anchor=(1.05, 0.75), fontsize=10)
        legend_old = fig.legend(old_handles, old_labels, title="Old Age Groups", loc="center right", bbox_to_anchor=(1.05, 0.25), fontsize=10)
    
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust to fit legends
        plt.show()
    
    # Generate plots for each disease
    for disease in ['HIV', 'Type2Diabetes']:
        real_data_source = eswatini_hiv_data if disease == 'HIV' else eswatini_t2d_data
        plot_prevalence_side_by_side(disease, real_data_source, [young_age_bins, old_age_bins])
except KeyError as e:
    print(f"KeyError: {e} - Check if the correct result keys are being used.")
    