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
beta = 0.0001
n_agents = 50000
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

# eswatini_t2d_data = {}
# for year in years:
#     t2d_prevalence_data, _ = mi.initialize_prevalence_data(
#         diseases= ['Type2Diabetes'], 
#         csv_file_path='mighti/data/prevalence_data_eswatini.csv', 
#         inityear=year
#     )
#     eswatini_t2d_data[year] = t2d_prevalence_data['Type2Diabetes']
    
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
    def prevalence_lambda(module, sim, size):
        """ Extract prevalence for the given disease, ensuring it returns a numeric probability. """
        if sim is None:
            raise ValueError(f"Simulation object is None when computing prevalence for {disease}")
        
        prev = mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, size)
        
        if isinstance(prev, np.ndarray):  # If it's an array, take the mean
            return np.mean(prev)
        return float(prev)  # Ensure a float is returned

    return prevalence_lambda

# Create disease objects
disease_objects = []
for disease in ncds:
    prevalence_function = get_prevalence_function(disease)
    
    # Use `strict=False` to avoid initialization errors
    init_prev = ss.bernoulli(prevalence_function, strict=False)

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

mortality_with_interactions = sim_with_interactions.results['new_deaths']  # New deaths per time step
cumulative_mortality_with_interactions = sim_with_interactions.results['cum_deaths']  # Cumulative deaths

# -------------------------
# Simulation Without Interactions
# -------------------------
ppl_no_interactions = ss.People(n_agents, age_data=pd.read_csv('tests/test_data/eswatini_age.csv'))
networks_no_interactions = [deepcopy(net) for net in networks]
pregnancy_no_interactions = ss.Pregnancy(pars=fertility_rates)
death_no_interactions = ss.Deaths(death_rates)

disease_objects_no_interactions = deepcopy(disease_objects)

sim_no_interactions = ss.Sim(
    n_agents=n_agents,
    networks=networks_no_interactions,
    diseases=disease_objects_no_interactions,
    analyzers=[prevalence_analyzer_without],
    start=inityear,
    end=endyear,
    people=ppl_no_interactions,
    demographics=[pregnancy_no_interactions, death_no_interactions],
    copy_inputs=False
)

print("Running Simulation Without Interactions...")
sim_no_interactions.run()

mortality_no_interactions = sim_no_interactions.results['new_deaths']  # New deaths per time step
cumulative_mortality_no_interactions = sim_no_interactions.results['cum_deaths']  # Cumulative deaths







# -------------------------
# Extract Mortality Data by Disease
# -------------------------
def extract_mortality(sim, disease_name):
    """Extract mortality for a specific disease."""
    disease_results = sim.results[disease_name.lower()]  # Disease-specific results
    mortality = disease_results['new_deaths']  # Adjust the key if necessary
    return mortality



# -------------------------
# Extract Prevalence Data by Gender
# -------------------------
def extract_prevalence_by_hiv_status(prevalence_analyzer, disease_name):
    """Extract prevalence for a specific disease stratified by HIV status and gender."""
    try:
        prevalence_hivpos_male = prevalence_analyzer.results[f'{disease_name}_prevalence_hivpos_male'] * 100
        prevalence_hivneg_male = prevalence_analyzer.results[f'{disease_name}_prevalence_hivneg_male'] * 100
        prevalence_hivpos_female = prevalence_analyzer.results[f'{disease_name}_prevalence_hivpos_female'] * 100
        prevalence_hivneg_female = prevalence_analyzer.results[f'{disease_name}_prevalence_hivneg_female'] * 100
        return prevalence_hivpos_male, prevalence_hivneg_male, prevalence_hivpos_female, prevalence_hivneg_female
    except KeyError as e:
        print(f"Prevalence data not found for {disease_name}: {e}")
        return None, None, None, None
    

# Function to extract prevalence by sex
def extract_prevalence_by_sex(prevalence_analyzer, disease_name):
    """Extract prevalence data by sex."""
    try:
        prevalence_male = prevalence_analyzer.results[f'{disease_name}_prevalence_male']
        prevalence_female = prevalence_analyzer.results[f'{disease_name}_prevalence_female']
        return prevalence_male, prevalence_female
    except KeyError as e:
        print(f"Prevalence data not found for {disease_name}: {e}")
        return None, None




# Retrieve the prevalence data for plotting
try:
    hiv_prevalence_data_male = prevalence_analyzer_with.results['HIV_prevalence_male'] * 100
    hiv_prevalence_data_female = prevalence_analyzer_with.results['HIV_prevalence_female'] * 100
    t2d_prevalence_data_male = prevalence_analyzer_with.results['Type2Diabetes_prevalence_male'] * 100
    t2d_prevalence_data_female = prevalence_analyzer_with.results['Type2Diabetes_prevalence_female'] * 100

    # Ensure age_bins is properly formatted
    age_bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    age_group_labels = [f'{left}-{right-1}' for left, right in zip(age_bins[:-1], age_bins[1:])]
    age_group_labels.append('80+') if age_bins[-1] == 80 else None

    cmap = pl.get_cmap('tab20', len(age_group_labels))  # Assign colors for age bins
    age_bin_colors = {label: cmap(i) for i, label in enumerate(age_group_labels)}

    # Known real data years
    real_years = [2007, 2011, 2017, 2021]

    # Initialize the plot
    n_diseases = len(diseases)
    fig, axs = pl.subplots(n_diseases, 2, figsize=(18, n_diseases * 6), sharey='row')

    # Iterate through diseases to plot both estimated and real data
    for disease_idx, disease in enumerate(diseases):
        male_data = prevalence_analyzer_with.results[f'{disease}_prevalence_male'] * 100
        female_data = prevalence_analyzer_with.results[f'{disease}_prevalence_female'] * 100

        # Plot estimated trends for each age group
        for i, label in enumerate(age_group_labels):
            axs[disease_idx, 0].plot(sim_with_interactions.yearvec, male_data[:, i], label=label, color=age_bin_colors[label], alpha=0.8)
            axs[disease_idx, 1].plot(sim_with_interactions.yearvec, female_data[:, i], color=age_bin_colors[label], alpha=0.8)

        # Plot real observed data (HIV and T2D)
        for year in real_years:
            # HIV Real Data
            if disease == 'HIV' and year in eswatini_hiv_data:
                real_male_data = eswatini_hiv_data[year]['male']
                real_female_data = eswatini_hiv_data[year]['female']

                for age_bin in real_male_data:
                    age_label = f'{age_bin}-99' if age_bin == 80 else f'{age_bin}-{age_bin + 4}'
                    if age_label in age_bin_colors:
                        axs[disease_idx, 0].scatter(year, real_male_data[age_bin] * 100, color=age_bin_colors[age_label], s=100, edgecolors='black', zorder=5)

                for age_bin in real_female_data:
                    age_label = f'{age_bin}-99' if age_bin == 80 else f'{age_bin}-{age_bin + 4}'
                    if age_label in age_bin_colors:
                        axs[disease_idx, 1].scatter(year, real_female_data[age_bin] * 100, color=age_bin_colors[age_label], s=100, edgecolors='black', zorder=5)

            # Type 2 Diabetes (T2D) Real Data
            if disease == 'Type2Diabetes' and year in eswatini_t2d_data:
                real_male_data = eswatini_t2d_data[year]['male']
                real_female_data = eswatini_t2d_data[year]['female']

                for age_bin in real_male_data:
                    age_label = f'{age_bin}-99' if age_bin == 80 else f'{age_bin}-{age_bin + 4}'
                    if age_label in age_bin_colors:
                        axs[disease_idx, 0].scatter(year, real_male_data[age_bin] * 100, color=age_bin_colors[age_label], s=100, edgecolors='black', marker='s', zorder=5)

                for age_bin in real_female_data:
                    age_label = f'{age_bin}-99' if age_bin == 80 else f'{age_bin}-{age_bin + 4}'
                    if age_label in age_bin_colors:
                        axs[disease_idx, 1].scatter(year, real_female_data[age_bin] * 100, color=age_bin_colors[age_label], s=100, edgecolors='black', marker='s', zorder=5)

        # Formatting the plots
        axs[disease_idx, 0].set_title(f'{disease} (Male)', fontsize=22)
        axs[disease_idx, 1].set_title(f'{disease} (Female)', fontsize=22)
        axs[disease_idx, 0].set_xlabel('Year', fontsize=18)
        axs[disease_idx, 1].set_xlabel('Year', fontsize=18)
        axs[disease_idx, 0].set_ylabel('Prevalence (%)', fontsize=18)
        axs[disease_idx, 0].grid(True)
        axs[disease_idx, 1].grid(True)

    # Add a common legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Age Groups', loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=len(age_group_labels) // 2, fontsize=12)

    # Adjust layout and show
    pl.tight_layout(rect=[0, 0.05, 1, 1])
    pl.show()

except KeyError as e:
    print(f"KeyError: {e} - Check if the correct result keys are being used.")

# # -------------------------
# # 1. Plot HIV Prevalence Over Time (Male & Female)
# # -------------------------
# plt.figure(figsize=(10, 5))
# plt.plot(years, hiv_prevalence_male.mean(axis=1) * 100, label='HIV Male', linestyle='-', color='blue')
# plt.plot(years, hiv_prevalence_female.mean(axis=1) * 100, label='HIV Female', linestyle='--', color='red')
# plt.xlabel("Year")
# plt.ylabel("HIV Prevalence (%)")
# plt.title("HIV Prevalence Over Time (By Sex)")
# plt.legend()
# plt.grid(True)
# plt.show()

# # -------------------------
# # 2. Plot T2D Prevalence Over Time (Male & Female)
# # -------------------------
# plt.figure(figsize=(10, 5))
# plt.plot(years, t2d_prevalence_male.mean(axis=1) * 100, label='T2D Male', linestyle='-', color='blue')
# plt.plot(years, t2d_prevalence_female.mean(axis=1) * 100, label='T2D Female', linestyle='--', color='red')
# plt.xlabel("Year")
# plt.ylabel("T2D Prevalence (%)")
# plt.title("Type 2 Diabetes Prevalence Over Time (By Sex)")
# plt.legend()
# plt.grid(True)
# plt.show()

# # -------------------------
# # 3. Plot HIV Prevalence By Age Group
# # -------------------------
# plt.figure(figsize=(10, 5))
# for i, age_bin in enumerate(prevalence_analyzer_with.age_bins['HIV']):
#     plt.plot(years, hiv_prevalence_by_age[:, i] * 100, label=f'Age {age_bin}')

# plt.xlabel("Year")
# plt.ylabel("HIV Prevalence (%)")
# plt.title("HIV Prevalence Over Time (By Age Group)")
# plt.legend(title="Age Group", loc='upper left', bbox_to_anchor=(1, 1))
# plt.grid(True)
# plt.show()

# # -------------------------
# # 4. Plot T2D Prevalence By Age Group
# # -------------------------
# plt.figure(figsize=(10, 5))
# for i, age_bin in enumerate(prevalence_analyzer_with.age_bins['Type2Diabetes']):
#     plt.plot(years, t2d_prevalence_by_age[:, i] * 100, label=f'Age {age_bin}')

# plt.xlabel("Year")
# plt.ylabel("T2D Prevalence (%)")
# plt.title("Type 2 Diabetes Prevalence Over Time (By Age Group)")
# plt.legend(title="Age Group", loc='upper left', bbox_to_anchor=(1, 1))
# plt.grid(True)
# plt.show()