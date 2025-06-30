"""
Provides utility functions for generating plots from simulation outputs
"""


import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import Counter


def plot_mean_prevalence_plhiv(sim, prevalence_analyzer, disease):
    """
    Plot mean prevalence over time for a given disease and both sexes.

    Parameters:
    - sim: The simulation object (provides `sim.timevec`)
    - prevalence_analyzer: The prevalence analyzer with stored results
    - disease: Name of the disease (e.g., 'HIV', 'Type2Diabetes')
    """

    def extract_results(key_pattern):
        return [prevalence_analyzer.results.get(f'{disease}_{key_pattern}_{i}', np.zeros(len(sim.timevec))) for i in range(len(prevalence_analyzer.age_bins))]

    male_num_with_HIV = np.sum(extract_results('num_with_HIV_male'), axis=0)
    female_num_with_HIV = np.sum(extract_results('num_with_HIV_female'), axis=0)
    male_den_with_HIV = np.sum(extract_results('den_with_HIV_male'), axis=0)
    female_den_with_HIV = np.sum(extract_results('den_with_HIV_female'), axis=0)
    male_num_without_HIV = np.sum(extract_results('num_without_HIV_male'), axis=0)
    female_num_without_HIV = np.sum(extract_results('num_without_HIV_female'), axis=0)
    male_den_without_HIV = np.sum(extract_results('den_without_HIV_male'), axis=0)
    female_den_without_HIV = np.sum(extract_results('den_without_HIV_female'), axis=0)

    male_den_with_HIV[male_den_with_HIV == 0] = 1
    female_den_with_HIV[female_den_with_HIV == 0] = 1
    male_den_without_HIV[male_den_without_HIV == 0] = 1
    female_den_without_HIV[female_den_without_HIV == 0] = 1

    mean_prevalence_male_with_HIV = np.nan_to_num(male_num_with_HIV / male_den_with_HIV) * 100
    mean_prevalence_female_with_HIV = np.nan_to_num(female_num_with_HIV / female_den_with_HIV) * 100
    mean_prevalence_male_without_HIV = np.nan_to_num(male_num_without_HIV / male_den_without_HIV) * 100
    mean_prevalence_female_without_HIV = np.nan_to_num(female_num_without_HIV / female_den_without_HIV) * 100
    
    # mean_prevalence_male_with_HIV = np.sum(extract_results('prev_with_HIV_male'), axis=0)
    # mean_prevalence_female_with_HIV = np.sum(extract_results('prev_with_HIV_female'), axis=0)
    # mean_prevalence_male_without_HIV = np.sum(extract_results('prev_without_HIV_male'), axis=0)
    # mean_prevalence_female_without_HIV = np.sum(extract_results('prev_without_HIV_female'), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(sim.timevec, mean_prevalence_male_with_HIV, label=f'Male {disease.capitalize()} Prevalence (HIV+)', linewidth=2, color='blue', linestyle='solid')
    ax.plot(sim.timevec, mean_prevalence_female_with_HIV, label=f'Female {disease.capitalize()} Prevalence (HIV+)', linewidth=2, color='red', linestyle='solid')
    ax.plot(sim.timevec, mean_prevalence_male_without_HIV, label=f'Male {disease.capitalize()} Prevalence (HIV-)', linewidth=2, color='blue', linestyle='dashed')
    ax.plot(sim.timevec, mean_prevalence_female_without_HIV, label=f'Female {disease.capitalize()} Prevalence (HIV-)', linewidth=2, color='red', linestyle='dashed')

    ax.set_xlabel('Year', fontsize=16)
    ax.set_ylabel(f'{disease.capitalize()} Prevalence (%)', fontsize=16)
    ax.set_title(f'Mean {disease.capitalize()} Prevalence Over Time (All Ages)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc='lower right', fontsize=14, frameon=False)   
    ax.grid()

    plt.show()


def plot_mean_prevalence(sim, prevalence_analyzer, disease, prevalence_data_df, init_year, end_year):
    """
    Plot mean prevalence over time for a given disease and both sexes, including observed data points.
    Calculate and print the mean prevalence over the specified range of years.

    Parameters:
    - sim: The simulation object (provides `sim.timevec`)
    - prevalence_analyzer: The prevalence analyzer with stored results
    - disease: Name of the disease (e.g., 'HIV', 'Type2Diabetes')
    - prevalence_data_df: The DataFrame containing observed prevalence data
    - init_year: The initial year of the simulation
    - end_year: The end year of the simulation
    """
    def extract_results(key_pattern):
        return [prevalence_analyzer.results.get(f'{disease}_{key_pattern}_{i}', np.zeros(len(sim.timevec))) for i in range(len(prevalence_analyzer.age_bins))]

    male_num = np.sum(extract_results('num_male'), axis=0)
    female_num = np.sum(extract_results('num_female'), axis=0)
    male_den = np.sum(extract_results('den_male'), axis=0)
    female_den = np.sum(extract_results('den_female'), axis=0)

    sim_length = len(sim.timevec)
    if len(male_num) != sim_length:
        male_num = np.zeros(sim_length)
    if len(female_num) != sim_length:
        female_num = np.zeros(sim_length)
    if len(male_den) != sim_length:
        male_den = np.zeros(sim_length)
    if len(female_den) != sim_length:
        female_den = np.zeros(sim_length)

    male_den[male_den == 0] = 1
    female_den[female_den == 0] = 1

    total_male_prevalence = np.nan_to_num(male_num /  male_den) * 100
    total_female_prevalence = np.nan_to_num(female_num / female_den) * 100

    mask = (sim.timevec >= init_year) & (sim.timevec <= end_year)

    plt.figure(figsize=(10, 5))
    plt.plot(sim.timevec[mask], total_male_prevalence[mask], label=f'Male {disease.capitalize()} Prevalence (Simulated)', linewidth=5, color='blue', linestyle='solid')
    plt.plot(sim.timevec[mask], total_female_prevalence[mask], label=f'Female {disease.capitalize()} Prevalence (Simulated)', linewidth=5, color='red', linestyle='solid')

    if prevalence_data_df is not None:
        plot_colors = {'male': 'blue',
                       'female': 'red'}

        for sex in ['male', 'female']:
            col = f'{disease}_{sex}'
            if col in prevalence_data_df.columns:
                observed_data = prevalence_data_df[['Year', col]].dropna()
                age_bins = [agetuple[0] for agetuple in sim.analyzers.prevalence_analyzer.age_bins]
                weights = dict(Counter(np.digitize(sim.people.age[sim.people[sex]], age_bins) - 1))

                observed_data = observed_data.groupby('Year', as_index=False).apply(
                    lambda x: np.average(
                        x[col],
                        weights=[weights.get(i, 0) for i in x.index]
                        if sum(weights.get(i, 0) for i in x.index) > 0 else None
                    ),
                    include_groups=False).rename(columns={None: col})

                observed_data[col] *= 100
                observed_data = observed_data[
                    (observed_data['Year'] >= init_year) & (observed_data['Year'] <= end_year)]

                plt.scatter(observed_data['Year'], observed_data[col],
                            color=plot_colors[sex], marker='o', edgecolor='black', s=100,
                            label=f'Observed {sex.capitalize()} Prevalence')
        
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel(f'{disease.capitalize()} Prevalence (%)')
    plt.title(f'Mean {disease.capitalize()} Prevalence Over Time (All Ages)')
    plt.grid()
    
    plt.show()


def plot_age_group_prevalence(sim, prevalence_analyzer, disease, prevalence_data_df, init_year, end_year, age_groups=None):
    """
    Plot age group prevalence over time for a given disease and both sexes, including observed data points.
    Create a figure with two panels: one for males and one for females.
    Age groups are color-coded and shared between male and female plots.

    Parameters:
    - sim: The simulation object (provides `sim.timevec`)
    - prevalence_analyzer: The prevalence analyzer with stored results
    - disease: Name of the disease (e.g., 'HIV', 'Type2Diabetes')
    - prevalence_data_df: The DataFrame containing observed prevalence data
    - init_year: The initial year of the simulation
    - end_year: The end year of the simulation
    - age_groups: List of tuples defining age group ranges and labels, e.g., [(0, 5, "0-4"), (5, 15, "5-14"), ...]
    """
    if age_groups is None:
        age_groups = [
            (0, 15, "0-15"),
            (15, 20, "15-20"),
            (20, 25, "20-25"),
            (25, 30, "25-30"),
            (30, 35, "30-35"),
            (35, 40, "35-40"),
            (40, 45, "40-45"),
            (45, 50, "45-50"),
            (50, 55, "50-55"),
            (55, 60, "55-60"),
            (60, 65, "60-65"),
            (65, 70, "65-70"),
            (70, 75, "70-75"),
            (75, 80, "75-80"),
            (80, 100, "80+")
        ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(age_groups)))

    def extract_results(key_pattern):
        return [prevalence_analyzer.results.get(f'{disease}_{key_pattern}_{i}', np.zeros(len(sim.timevec))) for i in range(len(age_groups))]

    male_num_bins = extract_results('num_male')
    female_num_bins = extract_results('num_female')
    male_den_bins = extract_results('den_male')
    female_den_bins = extract_results('den_female')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), sharey=True)
    ax_male, ax_female = axes

    for i, (start_age, end_age, label) in enumerate(age_groups):
        color = colors[i]
        
        male_prevalence = np.nan_to_num(male_num_bins[i] / male_den_bins[i]) * 100
        female_prevalence = np.nan_to_num(female_num_bins[i] / female_den_bins[i]) * 100
        mask = (sim.timevec >= init_year) & (sim.timevec <= end_year)
        ax_male.plot(sim.timevec[mask], male_prevalence[mask], label=label, color=color, linestyle='solid')
        ax_female.plot(sim.timevec[mask], female_prevalence[mask], label=label, color=color, linestyle='solid')

    # Plot observed prevalence data if available
    observed_lines = []
    if prevalence_data_df is not None:
        for i, (start_age, end_age, label) in enumerate(age_groups):
            color = colors[i]
            age_mask = (prevalence_data_df['Age'] >= start_age) & (prevalence_data_df['Age'] < end_age)

            male_col = f'{disease}_male'
            female_col = f'{disease}_female'

            if male_col in prevalence_data_df.columns:
                observed_male_data = prevalence_data_df[age_mask][['Year', male_col]].dropna()
                observed_male_data = observed_male_data.groupby('Year', as_index=False).mean()
                observed_male_data[male_col] *= 100

                observed_male_data = observed_male_data[(observed_male_data['Year'] >= init_year) & (observed_male_data['Year'] <= end_year)]

                obs_line, = ax_male.plot([], [], 'o', color=color, label=label)
                ax_male.scatter(observed_male_data['Year'], observed_male_data[male_col], 
                                color=color, marker='o', s=150)  
                observed_lines.append(obs_line)

            if female_col in prevalence_data_df.columns:
                observed_female_data = prevalence_data_df[age_mask][['Year', female_col]].dropna()
                observed_female_data = observed_female_data.groupby('Year', as_index=False).mean()
                observed_female_data[female_col] *= 100

                observed_female_data = observed_female_data[(observed_female_data['Year'] >= init_year) & (observed_female_data['Year'] <= end_year)]

                ax_female.scatter(observed_female_data['Year'], observed_female_data[female_col], 
                                  color=color, marker='o', s=150) 

    ax_male.set_xlabel('Year')
    ax_male.set_ylabel(f'{disease.capitalize()} Prevalence (%)')
    ax_male.set_title(f'Male {disease.capitalize()} Prevalence by Age Group')
    ax_male.grid()

    ax_female.set_xlabel('Year')
    ax_female.set_ylabel(f'{disease.capitalize()} Prevalence (%)')
    ax_female.set_title(f'Female {disease.capitalize()} Prevalence by Age Group')
    ax_female.grid()

    lines, labels = ax_male.get_legend_handles_labels()
    unique_labels = {label: line for label, line in zip(labels, lines)}
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', ncol=5)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()
    
    
# ---------------------------------------------------------------------
# Demographics related plot functions
# ---------------------------------------------------------------------
def plot_mx_comparison(sim_mx_df, observed_mx_csv, year, age_interval=5, figsize=(14, 10)):
    """
    Plot simulated and observed mx (mortality rate) by age group for a given year,
    using the output of calculate_mortality_rates.

    Args:
        sim_mx_df: DataFrame from calculate_mortality_rates with ['year', 'age', 'sex', 'mx']
        observed_mx_csv: Path to observed mx CSV with columns ['Time', 'Sex', 'Age', 'mx']
        year: Year to plot (should be present in both data sources)
        age_interval: Plot in this age grouping (default 5)
        figsize: Figure size
    """
    observed = observed_mx_csv
    sexes = ['Male', 'Female']
    colors = {'Male': 'blue', 'Female': 'red'}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes = [ax1, ax2]
    
    for i, sex in enumerate(sexes):
        color = colors[sex]
        sim_df = sim_mx_df[(sim_mx_df['sex'] == sex) & (sim_mx_df['year'] == year)][['age', 'mx']].copy()
        sim_df['age_group'] = (sim_df['age'] // age_interval) * age_interval
        sim_mx_grouped = sim_df.groupby('age_group')['mx'].mean().reset_index()

        obs_df = observed[(observed['Sex'] == sex) & (observed['Time'] == year)][['Age', 'mx']].copy()
        obs_df['age_group'] = (obs_df['Age'] // age_interval) * age_interval
        obs_mx_grouped = obs_df.groupby('age_group')['mx'].mean().reset_index()

        ax = axes[i]
        ax.set_yscale('log')  # ← log scale here

        ax.plot(sim_mx_grouped['age_group'], sim_mx_grouped['mx'],
                linestyle='-', linewidth=8, alpha=0.4, color=color, label='Simulated')
        ax.plot(obs_mx_grouped['age_group'], obs_mx_grouped['mx'],
                marker='s', linestyle='--', linewidth=2, markersize=8,
                color=color, label='Observed')
        ax.set_title(f"{sex} Mortality Rate (mx) Comparison, {year}", fontsize=16)
        ax.set_ylabel('mx (deaths per person-year)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel('Age Group', fontsize=14)
    plt.tight_layout()
    plt.show()    
    

def plot_life_expectancy(life_table, observed_data, year, max_age=100, figsize=(14, 10), title=None):
    """
    Plot simulated and observed life expectancy by age and sex for a given year.

    Args:
        life_table: DataFrame from simulation with ['Age', 'e(x)', 'sex', 'year']
        observed_data: Long-format DataFrame with ['Age', 'Sex', 'Time', 'ex']
        year: Year to filter
        max_age: Maximum age to plot
        figsize: Size of the figure
        title: Optional plot title
    """
    male_sim = life_table[life_table['sex'] == 'Male']
    female_sim = life_table[life_table['sex'] == 'Female']
    male_obs = observed_data[observed_data['Sex'] == 'Male']
    female_obs = observed_data[observed_data['Sex'] == 'Female']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax1.plot(male_sim['Age'], male_sim['e(x)'], '-', linewidth=8, alpha=0.4, color='blue', label='Simulated')
    ax1.plot(male_obs['Age'], male_obs['ex'], 's--', linewidth=2, markersize=8, color='blue', label='Observed')
    ax1.set_title('Male', fontsize=28)
    ax1.set_ylabel('Life Expectancy (years)', fontsize=24)
    ax1.tick_params(labelsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=20)

    ax2.plot(female_sim['Age'], female_sim['e(x)'], '-', linewidth=8, alpha=0.4, color='red', label='Simulated')
    ax2.plot(female_obs['Age'], female_obs['ex'], 's--', linewidth=2, markersize=8, color='red', label='Observed')
    ax2.set_title('Female', fontsize=28)
    ax2.set_xlabel('Age', fontsize=24)
    ax2.set_ylabel('Life Expectancy (years)', fontsize=24)
    ax2.tick_params(labelsize=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=20)

    if title:
        plt.suptitle(title, fontsize=24, y=0.98)
    else:
        plt.suptitle(f'Life Expectancy by Age in {year}', fontsize=24, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(bottom=0.1)
    plt.show()
    return fig, (ax1, ax2)


def plot_population_over_time(df, inityear, endyear, age_groups=None, nagent=50000, observed_data_path='demography/eswatini_age_distribution.csv'):
    """
    Plot population changes over time by age group and sex
    
    Args:
        df: DataFrame containing results with columns for population, birth rates, and death rates by age and sex
        inityear: Start year of the simulation
        endyear: End year of the simulation
        age_groups: List of tuples defining age groups (start_age, end_age, label)
        observed_data_path: Path to CSV file with observed age distribution
    """
    import pandas as pd
    
    if age_groups is None:
        # Default age groups
        age_groups = [
            (0, 5, "0-4"),
            (5, 15, "5-14"),
            (15, 25, "15-24"),
            (25, 35, "25-34"),
            (35, 45, "35-44"),
            (45, 55, "45-54"),
            (55, 65, "55-64"),
            (65, 75, "65-74"),
            (75, 85, "75-84"),
            (85, 101, "85+")
        ]

    observed_data = None
    if observed_data_path:
        try:
            observed_data = pd.read_csv(observed_data_path)
            print(f"Loaded observed age distribution data from {observed_data_path}")
        except Exception as e:
            print(f"Could not load observed data: {str(e)}")
    
    population_data = {
        'years': list(range(inityear, endyear + 1)),
        'male': {group[2]: [] for group in age_groups},
        'female': {group[2]: [] for group in age_groups}
    }
    
    for year in population_data['years']:
        if year not in df['year'].values:
            continue
            
        for start_age, end_age, label in age_groups:
            male_count = sum(df[df['year'] == year][f'male_population_age_{age}'] / nagent * 1000
                             for age in range(start_age, end_age + 1) if f'male_population_age_{age}' in df.columns)
            population_data['male'][label].append(male_count)
            female_count = sum(df[df['year'] == year][f'female_population_age_{age}'] / nagent * 1000
                               for age in range(start_age, end_age + 1) if f'female_population_age_{age}' in df.columns)
            population_data['female'][label].append(female_count)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    for label, counts in population_data['male'].items():
        if len(counts) == len(population_data['years']):
            ax1.plot(population_data['years'], counts, '-o', linewidth=2, label=f'Sim: {label}')
    
    for label, counts in population_data['female'].items():
        if len(counts) == len(population_data['years']):
            ax2.plot(population_data['years'], counts, '-o', linewidth=2, label=f'Sim: {label}')
    
    if observed_data is not None:
        observed_years = [str(y) for y in range(inityear, endyear + 1) 
                          if str(y) in observed_data.columns]
        
        if observed_years:
            obs_male = {}
            obs_female = {}
            
            for start_age, end_age, label in age_groups:
                male_rows = observed_data[(observed_data['sex'] == 'Male') & 
                                          (observed_data['age'] >= start_age) & 
                                          (observed_data['age'] <= end_age)]
                
                female_rows = observed_data[(observed_data['sex'] == 'Female') & 
                                            (observed_data['age'] >= start_age) & 
                                            (observed_data['age'] <= end_age)]
                
                obs_male[label] = []
                obs_female[label] = []
                
                for year_str in observed_years:
                    male_pop = male_rows[year_str].sum()
                    female_pop = female_rows[year_str].sum() 
                    
                    obs_male[label].append(male_pop)
                    obs_female[label].append(female_pop)
            
            obs_year_ints = [int(y) for y in observed_years]
            
            for label, counts in obs_male.items():
                if len(counts) > 0:
                    ax1.plot(obs_year_ints, counts, '--s', linewidth=1.5, alpha=0.7, 
                            label=f'Obs: {label}')
            
            for label, counts in obs_female.items():
                if len(counts) > 0:
                    ax2.plot(obs_year_ints, counts, '--s', linewidth=1.5, alpha=0.7, 
                            label=f'Obs: {label}')
            
            print(f"Added observed data for years: {', '.join(observed_years)}")
        else:
            print("No matching years found in observed data")
    
    ax1.set_title('Male Population by Age Group', fontsize=14)
    ax1.set_ylabel('Population Count (Thousands)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Female Population by Age Group', fontsize=14)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Population Count (Thousands)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    handles, labels = [], []
    for ax in [ax1, ax2]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=12)

    # Improve layout
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.subplots_adjust(bottom=0.15)
    plt.suptitle('Population Changes Over Time by Age Group and Sex', fontsize=16)
    plt.show()
    return fig, (ax1, ax2)
