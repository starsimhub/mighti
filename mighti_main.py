"""
Demo script for running MIGHTI analyses with age-dependent HIV and depression prevalence.
"""

# Imports
import starsim as ss
import mighti as mi
import pylab as pl
import pandas as pd
import numpy as np
import sciris as sc
from prevalence_analyzer import PrevalenceAnalyzer  # Import the custom analyzer

beta = 0.001  # Set beta to a small value for transmission
if beta == 0:
    print('Warning: transmission turned off!')

# Create the networks - sexual and maternal
mf = ss.MFNet(
    duration=1/24,  # Mean duration of relationships
    acts=80,
)
maternal = ss.MaternalNet()
networks = [mf, maternal]

# Create demographics
fertility_rates = {'fertility_rate': pd.read_csv(sc.thispath() / 'tests/test_data/nigeria_asfr.csv')}
pregnancy = ss.Pregnancy(pars=fertility_rates)
death_rates = {'death_rate': pd.read_csv(sc.thispath() / 'tests/test_data/nigeria_deaths.csv'), 'units': 1}
death = ss.Deaths(death_rates)

# Define the age-dependent prevalence data for HIV
age_data = {
    0: 0,
    15: 0.056,
    20: 0.172,
    25: 0.303,
    30: 0.425,
    35: 0.525,
    40: 0.572,
    45: 0.501,
    50: 0.435,
    55: 0.338,
    60: 0.21,
    65: 0.147,
    99: 0,
}
n_age_bins = len(age_data) - 1
age_bins = list(age_data.keys())
left_bins = age_bins[:-1]
right_bins = age_bins[1:]
age_vals = list(age_data.values())

# Define age-dependent initial prevalence function for HIV
def age_dependent_prevalence(module=None, sim=None, size=None):
    ages = sim.people.age[size]  # Initial ages of agents
    prevalence = np.zeros(len(ages))
    
    for i in range(n_age_bins):
        left = age_bins[i]
        right = age_bins[i+1]
        value = age_vals[i]
        prevalence[(ages >= left) & (ages < right)] = value

    return prevalence


# Define the age-dependent prevalence data for depression (similar to HIV)
depression_age_data = {
    0: 0,
    15: 0.05,
    20: 0.12,
    25: 0.25,
    30: 0.35,
    35: 0.45,
    40: 0.50,
    45: 0.45,
    50: 0.40,
    55: 0.30,
    60: 0.20,
    65: 0.15,
    99: 0,
}
n_depression_age_bins = len(depression_age_data) - 1
depression_age_bins = list(depression_age_data.keys())
depression_left_bins = depression_age_bins[:-1]
depression_right_bins = depression_age_bins[1:]
depression_age_vals = list(depression_age_data.values())

# Define age-dependent initial prevalence function for depression
# In the age_dependent_prevalence_depression function:
def age_dependent_prevalence_depression(module=None, sim=None, size=None):
    ages = sim.people.age[size]
    prevalence = np.zeros(len(ages))
    
    for i in range(n_depression_age_bins):
        left = depression_age_bins[i]
        right = depression_age_bins[i+1]
        value = depression_age_vals[i]
        prevalence[(ages >= left) & (ages < right)] = value
    
    print(f"Age-based prevalence generated: {prevalence}")  # Debugging print
    return prevalence




# Create the disease list, including HIV with age-dependent prevalence
hiv = ss.HIV(
    init_prev=ss.bernoulli(age_dependent_prevalence),  # Use age-dependent prevalence for HIV
    beta=beta,   # MTCT probability
)

# Modify the Depression object to use age-dependent initial prevalence
depression = mi.Depression(
    init_prev=ss.bernoulli(age_dependent_prevalence_depression)  # Use age-dependent prevalence for depression
)


# Create the list of diseases
diseases = [hiv, depression]
# diseases = [depression]

# Initialize the PrevalenceAnalyzer for both HIV and Depression
prevalence_analyzer = PrevalenceAnalyzer(age_data_hiv=age_data, age_data_depression=depression_age_data)

# Attach the analyzer to the simulation
sim = ss.Sim(
    n_agents=5000,
    networks=networks,
    diseases=diseases,
    analyzers=[prevalence_analyzer],
    start=2021,
    end=2030,
    connectors=mi.hiv_depression(rel_sus_hiv_depression=1),
    copy_inputs=False
)



# Run the simulation
sim.run()

# Plot HIV prevalence
try:
    hiv_prevalence_data = prevalence_analyzer.results['hiv_prevalence'] * 100  # Convert to percentage
    fig_age, ax_hiv = pl.subplots(figsize=(12, 8))
    age_group_labels = [f'{left}-{right-1}' for left, right in zip(left_bins, right_bins)]
    for i, label in enumerate(age_group_labels):
        ax_hiv.plot(sim.yearvec, hiv_prevalence_data[:, i], label=label)

    ax_hiv.set_title('HIV Prevalence by Age Group')
    ax_hiv.set_xlabel('Year')
    ax_hiv.set_ylabel('Prevalence (%)')
    ax_hiv.legend(title='Age Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_hiv.grid(True)
    pl.tight_layout()
    pl.show()
except KeyError:
    print("No HIV prevalence data found in the analyzer results.")

# Plot Depression prevalence
try:
    depression_prevalence_data = prevalence_analyzer.results['depression_prevalence'] * 100  # Convert to percentage
    fig_age, ax_depression = pl.subplots(figsize=(12, 8))
    age_group_labels = [f'{left}-{right-1}' for left, right in zip(left_bins, right_bins)]
    for i, label in enumerate(age_group_labels):
        ax_depression.plot(sim.yearvec, depression_prevalence_data[:, i], label=label)

    ax_depression.set_title('Depression Prevalence by Age Group')
    ax_depression.set_xlabel('Year')
    ax_depression.set_ylabel('Prevalence (%)')
    ax_depression.legend(title='Age Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_depression.grid(True)
    pl.tight_layout()
    pl.show()
except KeyError:
    print("No depression prevalence data found in the analyzer results.")