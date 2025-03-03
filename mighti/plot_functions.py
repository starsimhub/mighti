import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt



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

        if male_data is None or female_data is None:
            print(f"[WARNING] No prevalence data for {disease}. Skipping plot.")
            continue  # Skip plotting this disease if no data exists

        male_data *= 100  # Convert to percentage
        female_data *= 100

        # Plot male prevalence
        for i, label in enumerate(age_group_labels):
            axs[disease_idx, 0].plot(sim.timevec, male_data[:, i], label=label, color=age_bin_colors[label])
        axs[disease_idx, 0].set_title(f'{disease} (Male)', fontsize=24)
        axs[disease_idx, 0].set_xlabel('Year', fontsize=20)
        axs[disease_idx, 0].set_ylabel('Prevalence (%)', fontsize=20)
        axs[disease_idx, 0].tick_params(axis='both', labelsize=18)
        axs[disease_idx, 0].grid(True)

        # Plot female prevalence
        for i, label in enumerate(age_group_labels):
            axs[disease_idx, 1].plot(sim.timevec, female_data[:, i], color=age_bin_colors[label])
        axs[disease_idx, 1].set_title(f'{disease} (Female)', fontsize=24)
        axs[disease_idx, 1].set_xlabel('Year', fontsize=20)
        axs[disease_idx, 1].tick_params(axis='both', labelsize=18)
        axs[disease_idx, 1].grid(True)

        # # Add real data points for HIV if available
        # if disease == 'HIV':
        #     for year, real_data in eswatini_hiv_data.items():
        #         real_male_data = real_data['male']
        #         real_female_data = real_data['female']

        #         for age_bin in real_male_data:
        #             age_label = f'{age_bin}-99' if age_bin == 80 else f'{age_bin}-{age_bin + 4}'
        #             if age_label in age_bin_colors:
        #                 axs[disease_idx, 0].scatter(year, real_male_data[age_bin] * 100, color=age_bin_colors[age_label], s=100, zorder=5)

        #         for age_bin in real_female_data:
        #             age_label = f'{age_bin}-99' if age_bin == 80 else f'{age_bin}-{age_bin + 4}'
        #             if age_label in age_bin_colors:
        #                 axs[disease_idx, 1].scatter(year, real_female_data[age_bin] * 100, color=age_bin_colors[age_label], s=100, zorder=5)

    # Add a common legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Age Groups', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(age_group_labels) // 2, fontsize=12)

    # Adjust layout and show
    pl.tight_layout(rect=[0, 0.05, 1, 1])
    pl.show()
    

def plot_demography(time_steps, total_population, deaths, births):
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # First panel: Total Population (from second year)
    axes[0].plot(time_steps, total_population, label='Total Population', linewidth=2)
    axes[0].set_ylabel('Total Population')
    axes[0].set_title('Total Population Over Time')
    axes[0].legend()
    axes[0].grid()

    # Second panel: Estimated Births and Deaths
    axes[1].plot(time_steps, births, label='Estimated Births', linestyle='dashed')
    axes[1].plot(time_steps, deaths, label='Deaths', linestyle='dotted')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Estimated Births and Deaths Over Time')
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
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
    mean_prevalence_male = np.mean(male_data, axis=1)   # Convert to percentage
    mean_prevalence_female = np.mean(female_data, axis=1)

    # Create figure
    plt.figure(figsize=(10, 5))

    # Plot mean prevalence for males and females
    plt.plot(sim.timevec, mean_prevalence_male, label=f'Male {disease.capitalize()} Prevalence', linewidth=2, color='blue')
    plt.plot(sim.timevec, mean_prevalence_female, label=f'Female {disease.capitalize()} Prevalence', linewidth=2, color='red')

    # Labels and title
    plt.xlabel('Year')
    plt.ylabel(f'{disease.capitalize()} Prevalence (%)')
    plt.title(f'Mean {disease.capitalize()} Prevalence Over Time (All Ages)')
    plt.legend()
    plt.grid()

    plt.show()