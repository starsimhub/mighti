import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

def plot_disease_prevalence(sim, prevalence_analyzer, diseases, eswatini_hiv_data, age_bins):
    """
    Plot disease prevalence over time for males and females, with real-world Eswatini data for HIV.

    Parameters:
    - sim: The simulation object (provides `sim.timevec`)
    - prevalence_analyzer: The prevalence analyzer with stored results
    - diseases: List of disease names to plot
    - eswatini_hiv_data: Real-world prevalence data for HIV
    - age_bins: Age bin categories for plotting
    """

    # Ensure age_bins is a valid list of numbers
    if not age_bins or not isinstance(age_bins, (list, np.ndarray)):
        print("[ERROR] age_bins is empty or not a valid list. Using default values.")
        age_bins_list = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    else:
        try:
            age_bins_list = [int(age) for age in age_bins if str(age).isdigit()]
        except ValueError:
            print("[ERROR] age_bins contains non-numeric values. Using default values.")
            age_bins_list = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

    # Ensure age_bins_list has valid elements
    if len(age_bins_list) < 2:
        print("[ERROR] age_bins_list is too short. Using default values.")
        age_bins_list = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

    # Create age group labels
    age_group_labels = [f'{left}-{right-1}' for left, right in zip(age_bins_list[:-1], age_bins_list[1:])]
    if age_bins_list[-1] == 80:
        age_group_labels.append('80+')

    # Define color mapping for age groups
    cmap = pl.get_cmap('tab20', len(age_group_labels))
    age_bin_colors = {label: cmap(i) for i, label in enumerate(age_group_labels)}

    # Set up the figure
    n_diseases = len(diseases)
    fig, axs = pl.subplots(n_diseases, 2, figsize=(18, n_diseases * 6), sharey='row')

    for disease_idx, disease in enumerate(diseases):
        # Extract male and female prevalence data
        male_data = prevalence_analyzer.results.get(f'{disease}_prevalence_male', None)
        female_data = prevalence_analyzer.results.get(f'{disease}_prevalence_female', None)
        male_data_with_HIV = prevalence_analyzer.results.get(f'{disease}_prevalence_male_with_HIV', None)
        female_data_with_HIV = prevalence_analyzer.results.get(f'{disease}_prevalence_female_with_HIV', None)
        male_data_without_HIV = prevalence_analyzer.results.get(f'{disease}_prevalence_male_without_HIV', None)
        female_data_without_HIV = prevalence_analyzer.results.get(f'{disease}_prevalence_female_without_HIV', None)

        if male_data is None or female_data is None:
            print(f"[WARNING] No prevalence data for {disease}. Skipping plot.")
            continue  # Skip plotting this disease if no data exists

        # Convert prevalence data to percentage
        male_data *= 100
        female_data *= 100
        # male_data_with_HIV *= 100
        # female_data_with_HIV *= 100
        # male_data_without_HIV *= 100
        # female_data_without_HIV *= 100

        # Plot male prevalence
        for i, label in enumerate(age_group_labels):
            axs[disease_idx, 0].plot(sim.timevec, male_data[:, i], label=f'{label} (Total)', color=age_bin_colors[label], linestyle='solid')
            # axs[disease_idx, 0].plot(sim.timevec, male_data_with_HIV[:, i], label=f'{label} (HIV+)', color=age_bin_colors[label], linestyle='solid')
            # axs[disease_idx, 0].plot(sim.timevec, male_data_without_HIV[:, i], label=f'{label} (HIV-)', color=age_bin_colors[label], linestyle='dashed')
        axs[disease_idx, 0].set_title(f'{disease} (Male)', fontsize=24)
        axs[disease_idx, 0].set_xlabel('Year', fontsize=20)
        axs[disease_idx, 0].set_ylabel('Prevalence (%)', fontsize=20)
        axs[disease_idx, 0].tick_params(axis='both', labelsize=18)
        axs[disease_idx, 0].grid(True)

        # Plot female prevalence
        for i, label in enumerate(age_group_labels):
            axs[disease_idx, 1].plot(sim.timevec, female_data[:, i], label=f'{label} (Total)', color=age_bin_colors[label], linestyle='solid')
            # axs[disease_idx, 1].plot(sim.timevec, female_data_with_HIV[:, i], label=f'{label} (HIV+)', color=age_bin_colors[label], linestyle='solid')
            # axs[disease_idx, 1].plot(sim.timevec, female_data_without_HIV[:, i], label=f'{label} (HIV-)', color=age_bin_colors[label], linestyle='dashed')
        axs[disease_idx, 1].set_title(f'{disease} (Female)', fontsize=24)
        axs[disease_idx, 1].set_xlabel('Year', fontsize=20)
        axs[disease_idx, 1].tick_params(axis='both', labelsize=18)
        axs[disease_idx, 1].grid(True)

    # Add a common legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Age Groups', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(age_group_labels) // 2, fontsize=12)

    # Adjust layout and show
    pl.tight_layout(rect=[0, 0.05, 1, 1])
    pl.show()
    
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



    