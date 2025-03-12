import matplotlib.pyplot as plt
import pylab as pl
import numpy as np


def plot_mean_prevalence(sim, prevalence_analyzer, disease):
    """
    Plot mean prevalence over time for a given disease and both sexes.

    Parameters:
    - sim: The simulation object (provides `sim.timevec`)
    - prevalence_analyzer: The prevalence analyzer with stored results
    - disease: Name of the disease (e.g., 'HIV', 'Type2Diabetes')
    """

    # Extract male and female prevalence matrices
    male_data = prevalence_analyzer.results.get(f'{disease}_prevalence_male', None)
    female_data = prevalence_analyzer.results.get(f'{disease}_prevalence_female', None)


    # Ensure data exists
    if male_data is None or female_data is None:
        print(f"[ERROR] No prevalence data available for {disease}.")
        return

    # Compute mean prevalence across all age groups
    mean_prevalence_male = np.mean(male_data, axis=1) * 100 # Convert to percentage
    mean_prevalence_female = np.mean(female_data, axis=1) * 100

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
    

def plot_mean_prevalence_two_diseases(sim, prevalence_analyzer, diseases):
    """
    Plot mean prevalence over time for two diseases, each with male and female trends.

    Parameters:
    - sim: The simulation object (provides `sim.timevec`)
    - prevalence_analyzer: The prevalence analyzer with stored results
    - diseases: List of two disease names (e.g., ['HIV', 'Type2Diabetes'])
    """
    if len(diseases) != 2:
        print("[ERROR] Please provide exactly two diseases for comparison.")
        return
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharey=True)  # Two side-by-side panels
    
    for i, disease in enumerate(diseases):
        # Extract male and female prevalence matrices
        male_data = prevalence_analyzer.results.get(f'{disease}_prevalence_male', None)
        female_data = prevalence_analyzer.results.get(f'{disease}_prevalence_female', None)
        
        # Ensure data exists
        if male_data is None or female_data is None:
            print(f"[ERROR] No prevalence data available for {disease}.")
            continue
        
        # Compute mean prevalence across all age groups
        mean_prevalence_male = np.mean(male_data, axis=1) * 100  # Convert to percentage
        mean_prevalence_female = np.mean(female_data, axis=1) * 100
        
        # Plot mean prevalence for males and females
        axs[i].plot(sim.timevec, mean_prevalence_male, label=f'Male {disease.capitalize()}', linewidth=5, color='blue')
        axs[i].plot(sim.timevec, mean_prevalence_female, label=f'Female {disease.capitalize()}', linewidth=5, color='red', linestyle='dotted')
        
        # Labels and title
        axs[i].set_xlabel('Year')
        axs[i].set_ylabel('Prevalence (%)') if i == 0 else None  # Only set ylabel on the first plot
        axs[i].set_title(f'Mean {disease.capitalize()} Prevalence Over Time')
        axs[i].legend()
        axs[i].grid()
    
    plt.tight_layout()
    plt.show()
