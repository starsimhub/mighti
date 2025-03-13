import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

def plot_mean_prevalence_plhiv(sim, prevalence_analyzer, disease):
    """
    Plot mean prevalence over time for a given disease and both sexes.

    Parameters:
    - sim: The simulation object (provides `sim.timevec`)
    - prevalence_analyzer: The prevalence analyzer with stored results
    - disease: Name of the disease (e.g., 'HIV', 'Type2Diabetes')
    """

    # Extract male and female prevalence matrices
    # male_data = prevalence_analyzer.results.get(f'{disease}_prevalence_male', None)
    # female_data = prevalence_analyzer.results.get(f'{disease}_prevalence_female', None)
    male_data_with_HIV = prevalence_analyzer.results.get(f'{disease}_prevalence_with_HIV_male', None)
    female_data_with_HIV = prevalence_analyzer.results.get(f'{disease}_prevalence_with_HIV_female', None)
    male_data_without_HIV = prevalence_analyzer.results.get(f'{disease}_prevalence_without_HIV_male', None)
    female_data_without_HIV = prevalence_analyzer.results.get(f'{disease}_prevalence_without_HIV_female', None)

    # # Ensure data exists
    # if male_data is None or female_data is None:
    #     print(f"[ERROR] No prevalence data available for {disease}.")
    #     return

    # Compute mean prevalence across all age groups
    # mean_prevalence_male = np.mean(male_data, axis=1)  # Convert to percentage
    # mean_prevalence_female = np.mean(female_data, axis=1)
    mean_prevalence_male_with_HIV = np.mean(male_data_with_HIV, axis=1) * 100
    mean_prevalence_female_with_HIV = np.mean(female_data_with_HIV, axis=1) * 100
    mean_prevalence_male_without_HIV = np.mean(male_data_without_HIV, axis=1) * 100
    mean_prevalence_female_without_HIV = np.mean(female_data_without_HIV, axis=1) * 100

    # Create figure
    plt.figure(figsize=(10, 5))

    # Plot mean prevalence for males and females
    # plt.plot(sim.timevec, mean_prevalence_male, label=f'Male {disease.capitalize()} Prevalence (Total)', linewidth=5, color='blue', linestyle='dotted')
    # plt.plot(sim.timevec, mean_prevalence_female, label=f'Female {disease.capitalize()} Prevalence (Total)', linewidth=5, color='red', linestyle='dotted')
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

    plt.show()
    

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
    


def plot_mean_prevalence_two_diseases_parallel(msim, diseases):
    """
    Plot mean prevalence over time for two diseases, each with male and female trends, for parallel simulations,
    combining 'Separate' and 'Connector' simulations on the same panel.

    Parameters:
    - msim: The parallel simulation object containing multiple simulations
    - diseases: List of two disease names (e.g., ['HIV', 'Type2Diabetes'])
    """
    if len(diseases) != 2:
        print("[ERROR] Please provide exactly two diseases for comparison.")
        return
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # Two side-by-side panels for each disease
    
    for i, disease in enumerate(diseases):
        for sim in msim.sims:
            prevalence_analyzer = sim.analyzers[0]  # Access the prevalence analyzer for the current simulation
            male_data = prevalence_analyzer.results.get(f'{disease}_prevalence_male', None)
            female_data = prevalence_analyzer.results.get(f'{disease}_prevalence_female', None)
            
            # Ensure data exists
            if male_data is None or female_data is None:
                print(f"[ERROR] No prevalence data available for {disease} in {sim.label}.")
                continue
            
            # Compute mean prevalence across all age groups
            mean_prevalence_male = np.mean(male_data, axis=1) * 100  # Convert to percentage
            mean_prevalence_female = np.mean(female_data, axis=1) * 100
            
            # Set line style based on simulation label
            line_style = 'dotted' if 'Connector' in sim.label else 'solid'
            
            # Plot mean prevalence for males and females
            axs[i].plot(sim.timevec, mean_prevalence_male, label=f'{sim.label} Male {disease.capitalize()}', linewidth=2, color='blue', linestyle=line_style)
            axs[i].plot(sim.timevec, mean_prevalence_female, label=f'{sim.label} Female {disease.capitalize()}', linewidth=2, color='red', linestyle=line_style)
            
            # Labels and title
            axs[i].set_xlabel('Year')
            axs[i].set_ylabel('Prevalence (%)')
            axs[i].set_title(f'Mean {disease.capitalize()} Prevalence Over Time')
            axs[i].legend()
            axs[i].grid()
    
    plt.tight_layout()
    plt.show()