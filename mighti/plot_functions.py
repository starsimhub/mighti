#Exact copy of minimal_mighti
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

def plot_mortality_rates(death_csv_path, sim_mortality_rates, year):
    # Load actual mortality data
    death_data = pd.read_csv(death_csv_path)
    
    # Filter for the specified year
    death_data_year = death_data[death_data['Time'] == 2020]
    
    # Extract data for males and females
    male_data = death_data_year[death_data_year['Sex'] == 'Male']
    female_data = death_data_year[death_data_year['Sex'] == 'Female']
    
    # Extract mortality rates for ages 0 to 80 from the simulation
    ages = range(91)
    sim_mortality_rates_male = {age: sim_mortality_rates['male'].get(age, 0) for age in ages}
    sim_mortality_rates_female = {age: sim_mortality_rates['female'].get(age, 0) for age in ages}
    
    # Plot mortality rates
    plt.figure(figsize=(10, 6))
    
    # # Plot actual data
    plt.scatter(male_data['AgeGrpStart'], male_data['mx'], color='blue', label='Male (Actual)', alpha=1)
    plt.scatter(female_data['AgeGrpStart'], female_data['mx'], color='red', label='Female (Actual)', alpha=1)
    
    # Plot simulated data
    plt.plot(ages, [sim_mortality_rates_male[age] for age in ages], color='blue', label='Male (Simulated)', alpha=0.6)
    plt.plot(ages, [sim_mortality_rates_female[age] for age in ages], color='red', label='Female (Simulated)', alpha=0.6)
    
    plt.xlabel('Age')
    plt.ylabel('Mortality Rate')
    plt.title(f'Mortality Rates by Age for {year}')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def plot_life_expectancy(lifeexpectancy_csv_path, simulated_male, simulated_female):
    # Load actual life expectancy data
    real_data = pd.read_csv(lifeexpectancy_csv_path)
    
    # Extract data for males and females
    male_data = real_data[real_data['sex'] == 'male']
    female_data = real_data[real_data['sex'] == 'female']
    
    # Plot life expectancy
    plt.figure(figsize=(12, 8))
    
    # Plot real data for males and females
    plt.scatter(male_data['age'], male_data['value'], label='Real Male', linestyle='--', color='blue')
    plt.scatter(female_data['age'], female_data['value'], label='Real Female', linestyle='--', color='red')
    
    # Plot simulated data for males and females
    plt.plot(simulated_male['Age'], simulated_male['e(x)'], label='Simulated Male', linestyle='-', color='blue')
    plt.plot(simulated_female['Age'], simulated_female['e(x)'], label='Simulated Female', linestyle='-', color='red')
    
    plt.xlabel('Age')
    plt.ylabel('Life Expectancy')
    plt.title('Life Expectancy: Real vs Simulated')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.show()


def plot_numerator_denominator(sim, prevalence_analyzer, disease):
    """
    Plot numerator and denominator over time for a given disease and different groups (e.g., male, female, with HIV, without HIV).

    Parameters:
    - sim: The simulation object (provides `sim.timevec`)
    - prevalence_analyzer: The prevalence analyzer with stored results
    - disease: Name of the disease (e.g., 'HIV', 'Type2Diabetes')
    """

    # Extract prevalence numerators and denominators for all groups
    def extract_results(key_pattern):
        return np.mean([prevalence_analyzer.results.get(f'{disease}_{key_pattern}_{i}', np.zeros(len(sim.timevec))) for i in range(len(prevalence_analyzer.age_bins))], axis=0)

    male_num = extract_results('num_male')
    female_num = extract_results('num_female')
    male_den = extract_results('den_male')
    female_den = extract_results('den_female')
    male_num_with_HIV = extract_results('num_with_HIV_male')
    female_num_with_HIV = extract_results('num_with_HIV_female')
    male_den_with_HIV = extract_results('den_with_HIV_male')
    female_den_with_HIV = extract_results('den_with_HIV_female')
    male_num_without_HIV = extract_results('num_without_HIV_male')
    female_num_without_HIV = extract_results('num_without_HIV_female')
    male_den_without_HIV = extract_results('den_without_HIV_male')
    female_den_without_HIV = extract_results('den_without_HIV_female')

  
    fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # Plot numerator
    axs[0].plot(sim.timevec, male_num, label='Male Numerator', linewidth=6, color='blue', linestyle='solid', alpha=0.3)
    axs[0].plot(sim.timevec, female_num, label='Female Numerator', linewidth=6, color='red', linestyle='solid', alpha=0.3)
    axs[0].plot(sim.timevec, male_num_with_HIV, label='Male Numerator (HIV+)', linewidth=2, color='blue', linestyle='dotted')
    axs[0].plot(sim.timevec, female_num_with_HIV, label='Female Numerator (HIV+)', linewidth=2, color='red', linestyle='dotted')
    axs[0].plot(sim.timevec, male_num_without_HIV, label='Male Numerator (HIV-)', linewidth=2, color='blue', linestyle='dashed')
    axs[0].plot(sim.timevec, female_num_without_HIV, label='Female Numerator (HIV-)', linewidth=2, color='red', linestyle='dashed')
    axs[0].set_ylabel('Count')
    axs[0].set_title(f'{disease.capitalize()} Numerator Over Time (All Groups)')
    axs[0].legend()
    axs[0].grid()

    # Plot denominator
    axs[1].plot(sim.timevec, male_den, label='Male Denominator', linewidth=6, color='blue', linestyle='solid', alpha=0.3)
    axs[1].plot(sim.timevec, female_den, label='Female Denominator', linewidth=6, color='red', linestyle='solid', alpha=0.3)
    axs[1].plot(sim.timevec, male_den_with_HIV, label='Male Denominator (HIV+)', linewidth=2, color='blue', linestyle='dotted')
    axs[1].plot(sim.timevec, female_den_with_HIV, label='Female Denominator (HIV+)', linewidth=2, color='red', linestyle='dotted')
    axs[1].plot(sim.timevec, male_den_without_HIV, label='Male Denominator (HIV-)', linewidth=2, color='blue', linestyle='dashed')
    axs[1].plot(sim.timevec, female_den_without_HIV, label='Female Denominator (HIV-)', linewidth=2, color='red', linestyle='dashed')
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Count')
    axs[1].set_title(f'{disease.capitalize()} Denominator Over Time (All Groups)')
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()


def plot_mean_prevalence_plhiv(sim, prevalence_analyzer, disease):
    """
    Plot mean prevalence over time for a given disease and both sexes.

    Parameters:
    - sim: The simulation object (provides `sim.timevec`)
    - prevalence_analyzer: The prevalence analyzer with stored results
    - disease: Name of the disease (e.g., 'HIV', 'Type2Diabetes')
    """

    # Extract male and female prevalence numerators and denominators
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

  
    # Check for division by zero
    male_den_with_HIV[male_den_with_HIV == 0] = 1
    female_den_with_HIV[female_den_with_HIV == 0] = 1
    male_den_without_HIV[male_den_without_HIV == 0] = 1
    female_den_without_HIV[female_den_without_HIV == 0] = 1

    # Compute mean prevalence across all age groups
    mean_prevalence_male_with_HIV = np.nan_to_num(male_num_with_HIV / male_den_with_HIV) * 100
    mean_prevalence_female_with_HIV = np.nan_to_num(female_num_with_HIV / female_den_with_HIV) * 100
    mean_prevalence_male_without_HIV = np.nan_to_num(male_num_without_HIV / male_den_without_HIV) * 100
    mean_prevalence_female_without_HIV = np.nan_to_num(female_num_without_HIV / female_den_without_HIV) * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot mean prevalence for males and females
    ax.plot(sim.timevec, mean_prevalence_male_with_HIV, label=f'Male {disease.capitalize()} Prevalence (HIV+)', linewidth=2, color='blue', linestyle='solid')
    ax.plot(sim.timevec, mean_prevalence_female_with_HIV, label=f'Female {disease.capitalize()} Prevalence (HIV+)', linewidth=2, color='red', linestyle='solid')
    ax.plot(sim.timevec, mean_prevalence_male_without_HIV, label=f'Male {disease.capitalize()} Prevalence (HIV-)', linewidth=2, color='blue', linestyle='dashed')
    ax.plot(sim.timevec, mean_prevalence_female_without_HIV, label=f'Female {disease.capitalize()} Prevalence (HIV-)', linewidth=2, color='red', linestyle='dashed')

    # Labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{disease.capitalize()} Prevalence (%)')
    ax.set_title(f'Mean {disease.capitalize()} Prevalence Over Time (All Ages)')
    ax.legend()
    ax.grid()

    # Set y-axis ticks
    # ax.set_yticks(np.arange(0, 101, 20))

    return fig


def plot_mean_prevalence(sim, prevalence_analyzer, disease):
    """
    Plot mean prevalence over time for a given disease and both sexes.

    Parameters:
    - sim: The simulation object (provides `sim.timevec`)
    - prevalence_analyzer: The prevalence analyzer with stored results
    - disease: Name of the disease (e.g., 'HIV', 'Type2Diabetes')
    """

    # Extract male and female prevalence numerators and denominators
    def extract_results(key_pattern):
        return [prevalence_analyzer.results.get(f'{disease}_{key_pattern}_{i}', np.zeros(len(sim.timevec))) for i in range(len(prevalence_analyzer.age_bins))]

    male_num = np.sum(extract_results('num_male'), axis=0)
    female_num = np.sum(extract_results('num_female'), axis=0)
    male_den = np.sum(extract_results('den_male'), axis=0)
    female_den = np.sum(extract_results('den_female'), axis=0)

    # Ensure the arrays are the correct length (44)
    sim_length = len(sim.timevec)
    if len(male_num) != sim_length:
        male_num = np.zeros(sim_length)
    if len(female_num) != sim_length:
        female_num = np.zeros(sim_length)
    if len(male_den) != sim_length:
        male_den = np.zeros(sim_length)
    if len(female_den) != sim_length:
        female_den = np.zeros(sim_length)


    # Check for division by zero
    male_den[male_den == 0] = 1
    female_den[female_den == 0] = 1

    # Compute mean prevalence across all age groups
    mean_prevalence_male = np.nan_to_num(male_num / male_den) * 100
    mean_prevalence_female = np.nan_to_num(female_num / female_den) * 100

    # Create figure
    plt.figure(figsize=(10, 5))

    # Plot mean prevalence for males and females
    plt.plot(sim.timevec, mean_prevalence_male, label=f'Male {disease.capitalize()} Prevalence (Total)', linewidth=5, color='blue', linestyle='dotted')
    plt.plot(sim.timevec, mean_prevalence_female, label=f'Female {disease.capitalize()} Prevalence (Total)', linewidth=5, color='red', linestyle='dotted')

    # Labels and title
    plt.xlabel('Year')
    plt.ylabel(f'{disease.capitalize()} Prevalence (%)')
    plt.title(f'Mean {disease.capitalize()} Prevalence Over Time (All Ages)')
    plt.legend()
    plt.grid()

    plt.show()