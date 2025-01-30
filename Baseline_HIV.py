"""
Script to plot age-dependent HIV prevalence without interactions
"""

# Imports
import starsim as ss
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
death_rates = {'death_rate': pd.read_csv(sc.thispath() / 'tests/test_data/nigeria_deaths.csv'), 'rate_units': 1}
death = ss.Deaths(death_rates)

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
left_right = list(zip(left_bins, right_bins))
age_vals = list(age_data.values())

# Define age-dependent initial prevalence function
def age_dependent_prevalence(module=None, sim=None, size=None):
    ages = sim.people.age[size]  # Initial ages of agents
    prevalence = np.zeros(len(ages))
    
    for i in range(n_age_bins):
        left = age_bins[i]
        right = age_bins[i+1]
        value = age_vals[i]
        prevalence[(ages >= left) & (ages < right)] = value

    return prevalence

# Initialize HIV with age-dependent initial prevalence
hiv = ss.HIV(
    init_prev=ss.bernoulli(age_dependent_prevalence),
    beta=beta,  # Overall transmission rate
)

# Initialize the PrevalenceAnalyzer
prevalence_analyzer = PrevalenceAnalyzer(age_data=age_data)

# Run baseline HIV simulation without interactions
print('Running baseline HIV simulation without interactions')
baseline_sim = ss.Sim(
    n_agents=500000,
    networks=networks,
    diseases=[hiv],
    start=2021,
    end=2030,
    analyzers=[prevalence_analyzer],  # Add the analyzer here
    copy_inputs=False,
)
baseline_sim.run()

# Access and plot the prevalence data collected by the analyzer
try:
    prevalence_data = prevalence_analyzer.results['prevalence'] * 100  # Multiply by 100 to convert to percentage
    fig_age, ax_age = pl.subplots(figsize=(12, 8))
    age_group_labels = [f'{left}-{right-1}' for left, right in zip(left_bins, right_bins)]
    for i, label in enumerate(age_group_labels):
        ax_age.plot(baseline_sim.yearvec, prevalence_data[:, i], label=label)

    ax_age.set_title('HIV Prevalence by Age Group')
    ax_age.set_xlabel('Year')
    ax_age.set_ylabel('Prevalence (%)')
    ax_age.legend(title='Age Groups', bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside
    ax_age.grid(True)
    pl.tight_layout()
    pl.show()
except KeyError:
    print("No prevalence data found in the analyzer results.")
