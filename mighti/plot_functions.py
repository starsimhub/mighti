import matplotlib.pyplot as plt
import numpy as np


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
    plt.figure(figsize=(10, 5))

    # Plot mean prevalence for males and females
    plt.plot(sim.timevec, mean_prevalence_male_with_HIV, label=f'Male {disease.capitalize()} Prevalence (HIV+)', linewidth=2, color='blue', linestyle='solid')
    plt.plot(sim.timevec, mean_prevalence_female_with_HIV, label=f'Female {disease.capitalize()} Prevalence (HIV+)', linewidth=2, color='red', linestyle='solid')
    plt.plot(sim.timevec, mean_prevalence_male_without_HIV, label=f'Male {disease.capitalize()} Prevalence (HIV-)', linewidth=2, color='blue', linestyle='dashed')
    plt.plot(sim.timevec, mean_prevalence_female_without_HIV, label=f'Female {disease.capitalize()} Prevalence (HIV-)', linewidth=2, color='red', linestyle='dashed')

    # Labels and title
    plt.xlabel('Year')
    plt.ylabel(f'{disease.capitalize()} Prevalence (%)')
    plt.title(f'Mean {disease.capitalize()} Prevalence Over Time (All Ages)')
    plt.legend()
    plt.grid()
    
    plt.yticks(np.arange(0, 101, 20))

    plt.show()


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