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


def plot_mean_prevalence_with_standardization(
    sim, prevalence_analyzer, disease, prevalence_data_df, init_year, end_year, reference_population
):
    """
    Plot age-standardized mean prevalence over time for a given disease and both sexes, including observed data points.
    Calculate and print the mean prevalence over the specified range of years.

    Parameters:
    - sim: The simulation object (provides `sim.timevec`)
    - prevalence_analyzer: The prevalence analyzer with stored results
    - disease: Name of the disease (e.g., 'HIV', 'Type2Diabetes')
    - prevalence_data_df: The DataFrame containing observed prevalence data
    - init_year: The initial year of the simulation
    - end_year: The end year of the simulation
    - reference_population: A dictionary with age-sex bin proportions for standardization
    """

    # Extract male and female prevalence numerators and denominators
    def extract_results(key_pattern):
        """
        Helper function to extract results for all age bins for a given key pattern.
        Returns a list of arrays, one for each age bin.
        """
        return [
            np.array(prevalence_analyzer.results.get(f'{disease}_{key_pattern}_{i}', np.zeros(len(sim.timevec))))
            for i in range(len(prevalence_analyzer.age_bins))
        ]

    male_num = extract_results('num_male')  # List of arrays (one per age bin)
    female_num = extract_results('num_female')
    male_den = extract_results('den_male')
    female_den = extract_results('den_female')

    # Initialize arrays for age-standardized prevalence
    standardized_prevalence_male = np.zeros(len(sim.timevec))
    standardized_prevalence_female = np.zeros(len(sim.timevec))

    # Calculate age-standardized prevalence
    for age_bin, weight in reference_population['male'].items():
        # Ensure no division by zero
        denominator = np.maximum(male_den[age_bin], 1)
        prevalence = male_num[age_bin] / denominator
        weighted_prevalence = prevalence * weight
        standardized_prevalence_male += weighted_prevalence

    for age_bin, weight in reference_population['female'].items():
        # Ensure no division by zero
        denominator = np.maximum(female_den[age_bin], 1)
        prevalence = female_num[age_bin] / denominator
        weighted_prevalence = prevalence * weight
        standardized_prevalence_female += weighted_prevalence

    # Convert to percentages
    standardized_prevalence_male *= 100
    standardized_prevalence_female *= 100

    # Debugging: Print standardized prevalence for the first few years
    print("\nChecking age-standardized prevalence values:")
    for t in range(3):  # First 3 time points for debugging
        year = init_year + t
        print(f"Year {year}: Male prevalence = {standardized_prevalence_male[t]:.2f}%, Female prevalence = {standardized_prevalence_female[t]:.2f}%")

    # Filter the data based on init_year and end_year
    mask = (sim.timevec >= init_year) & (sim.timevec <= end_year)

    # Calculate the mean prevalence over the specified range of years
    mean_prevalence_male_over_years = np.mean(standardized_prevalence_male[mask])
    mean_prevalence_female_over_years = np.mean(standardized_prevalence_female[mask])
    print(f"Mean Male Prevalence for {disease} from {init_year} to {end_year}: {mean_prevalence_male_over_years:.2f}% (age-standardized)")
    print(f"Mean Female Prevalence for {disease} from {init_year} to {end_year}: {mean_prevalence_female_over_years:.2f}% (age-standardized)")

    # Create figure
    plt.figure(figsize=(10, 5))

    # Plot mean prevalence for males and females
    plt.plot(sim.timevec[mask], standardized_prevalence_male[mask], label=f'Male {disease.capitalize()} Prevalence (Standardized)', linewidth=2, color='blue')
    plt.plot(sim.timevec[mask], standardized_prevalence_female[mask], label=f'Female {disease.capitalize()} Prevalence (Standardized)', linewidth=2, color='red')

    # Plot observed prevalence data if available
    if prevalence_data_df is not None:
        male_col = f'{disease}_male'
        female_col = f'{disease}_female'

        # Check if columns for observed male and female prevalence exist
        if male_col in prevalence_data_df.columns:
            # Drop NaN values from both Year and male prevalence data
            observed_male_data = prevalence_data_df[['Year', male_col]].dropna()
            observed_male_data = observed_male_data.groupby('Year', as_index=False).mean()
            observed_male_data[male_col] *= 100
            print(f"Observed Male Data for {disease}:\n", observed_male_data)

            # Filter observed data based on init_year and end_year
            observed_male_data = observed_male_data[(observed_male_data['Year'] >= init_year) & (observed_male_data['Year'] <= end_year)]

            # Plot observed male data
            plt.scatter(observed_male_data['Year'], observed_male_data[male_col], 
                        color='blue', marker='o', edgecolor='black', s=100, 
                        label='Observed Male Prevalence')

        if female_col in prevalence_data_df.columns:
            # Drop NaN values from both Year and female prevalence data
            observed_female_data = prevalence_data_df[['Year', female_col]].dropna()
            observed_female_data = observed_female_data.groupby('Year', as_index=False).mean()
            observed_female_data[female_col] *= 100

            # Filter observed data based on init_year and end_year
            observed_female_data = observed_female_data[(observed_female_data['Year'] >= init_year) & (observed_female_data['Year'] <= end_year)]

            # Plot observed female data
            plt.scatter(observed_female_data['Year'], observed_female_data[female_col], 
                        color='red', marker='o', edgecolor='black', s=100, 
                        label='Observed Female Prevalence')
            
    # Labels and title
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel(f'{disease.capitalize()} Prevalence (%)')
    plt.title(f'Age-Standardized {disease.capitalize()} Prevalence Over Time')
    plt.grid()

    # Show plot
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

    # Extract male and female prevalence numerators and denominators by summing over age bins
    def extract_results(key_pattern):
        return [prevalence_analyzer.results.get(f'{disease}_{key_pattern}_{i}', np.zeros(len(sim.timevec))) for i in range(len(prevalence_analyzer.age_bins))]

    male_num = np.sum(extract_results('num_male'), axis=0)
    female_num = np.sum(extract_results('num_female'), axis=0)
    male_den = np.sum(extract_results('den_male'), axis=0)
    female_den = np.sum(extract_results('den_female'), axis=0)

    # Ensure the arrays are the correct length
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

    # Compute prevalence for each year as numerator / (numerator + denominator)
    total_male_prevalence = np.nan_to_num(male_num /  male_den) * 100
    total_female_prevalence = np.nan_to_num(female_num / female_den) * 100
    print(f"total_male_prevalence is {total_male_prevalence}")
    print(f"total_female_prevalence is {total_female_prevalence}")
    # Filter the data based on init_year and end_year
    years = np.arange(init_year, end_year + 1)
    mask = (sim.timevec >= init_year) & (sim.timevec <= end_year)

    # Create figure
    plt.figure(figsize=(10, 5))

    # Plot mean prevalence for males and females
    plt.plot(sim.timevec[mask], total_male_prevalence[mask], label=f'Male {disease.capitalize()} Prevalence (Simulated)', linewidth=5, color='blue', linestyle='solid')
    plt.plot(sim.timevec[mask], total_female_prevalence[mask], label=f'Female {disease.capitalize()} Prevalence (Simulated)', linewidth=5, color='red', linestyle='solid')

    # Plot observed prevalence data if available
    if prevalence_data_df is not None:
        male_col = f'{disease}_male'
        female_col = f'{disease}_female'

        plot_colors = {'male': 'blue',
                       'female': 'red'}

        for sex in ['male', 'female']:
            col = f'{disease}_{sex}'
            if col in prevalence_data_df.columns:
                observed_data = prevalence_data_df[['Year', col]].dropna()
                age_bins = [agetuple[0] for agetuple in sim.analyzers.prevalence_analyzer.age_bins]
                weights = dict(Counter(np.digitize(sim.people.age[sim.people[sex]], age_bins) - 1))

                # arithmetic mean -- not accurate for populations of non-uniform ages
                # observed_data = observed_data.groupby('Year', as_index=False).mean()

                observed_data = observed_data.groupby('Year', as_index=False).apply(
                    lambda x: np.average(
                        x[col],
                        weights=[weights.get(i, 0) for i in x.index]
                        if sum(weights.get(i, 0) for i in x.index) > 0 else None
                    )
                ).rename(columns={None: col})

                observed_data[col] *= 100

                # Filter observed data based on init_year and end_year
                observed_data = observed_data[
                    (observed_data['Year'] >= init_year) & (observed_data['Year'] <= end_year)]

                # Plot observed male data
                plt.scatter(observed_data['Year'], observed_data[col],
                            color=plot_colors[sex], marker='o', edgecolor='black', s=100,
                            label=f'Observed {sex.capitalize()} Prevalence')
        
    # Labels and title
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel(f'{disease.capitalize()} Prevalence (%)')
    plt.title(f'Mean {disease.capitalize()} Prevalence Over Time (All Ages)')
    plt.grid()
    
    plt.show()
    
# def plot_mean_prevalence(sim, prevalence_analyzer, disease, prevalence_data_df, init_year, end_year):
#     """
#     Plot mean prevalence over time for a given disease and both sexes, including observed data points.
#     Calculate and print the mean prevalence over the specified range of years.

#     Parameters:
#     - sim: The simulation object (provides `sim.timevec`)
#     - prevalence_analyzer: The prevalence analyzer with stored results
#     - disease: Name of the disease (e.g., 'HIV', 'Type2Diabetes')
#     - prevalence_data_df: The DataFrame containing observed prevalence data
#     - init_year: The initial year of the simulation
#     - end_year: The end year of the simulation
#     """

#     # Extract male and female prevalence numerators and denominators
#     def extract_results(key_pattern):
#         return [prevalence_analyzer.results.get(f'{disease}_{key_pattern}_{i}', np.zeros(len(sim.timevec))) for i in range(len(prevalence_analyzer.age_bins))]

#     male_num = np.sum(extract_results('num_male'), axis=0)
#     female_num = np.sum(extract_results('num_female'), axis=0)
#     male_den = np.sum(extract_results('den_male'), axis=0)
#     female_den = np.sum(extract_results('den_female'), axis=0)

#     # Ensure the arrays are the correct length (44)
#     sim_length = len(sim.timevec)
#     if len(male_num) != sim_length:
#         male_num = np.zeros(sim_length)
#     if len(female_num) != sim_length:
#         female_num = np.zeros(sim_length)
#     if len(male_den) != sim_length:
#         male_den = np.zeros(sim_length)
#     if len(female_den) != sim_length:
#         female_den = np.zeros(sim_length)

#     # Check for division by zero
#     male_den[male_den == 0] = 1
#     female_den[female_den == 0] = 1

#     # Compute mean prevalence across all age groups
#     mean_prevalence_male = np.nan_to_num(male_num / male_den) * 100
#     mean_prevalence_female = np.nan_to_num(female_num / female_den) * 100


#     # Add this right after the line that prints mean prevalence
#     print("\nChecking raw prevalence values:")
#     age_bins = list(range(15))  # Assuming 15 age bins
#     timepoints = list(range(len(prevalence_analyzer.results['Type2Diabetes_num_male_0'])))
#     for t in timepoints[:3]:  # Just look at first 3 timepoints
#         year = 2007 + t
#         total_male_num = 0
#         total_male_den = 0
#         for age_bin in age_bins:
#             num_key = f'Type2Diabetes_num_male_{age_bin}'
#             den_key = f'Type2Diabetes_den_male_{age_bin}'
#             if num_key in prevalence_analyzer.results and den_key in prevalence_analyzer.results:
#                 num = prevalence_analyzer.results[num_key][t]
#                 den = prevalence_analyzer.results[den_key][t]
#                 total_male_num += num
#                 total_male_den += den
        
#         # Calculate prevalence two ways
#         raw_prev = total_male_num / (total_male_num + total_male_den) if (total_male_num + total_male_den) > 0 else 0
#         print(f"Year {year}: Male prevalence = {raw_prev:.6f} (raw) = {raw_prev*100:.4f}%")
#     # Filter the data based on init_year and end_year
#     years = np.arange(init_year, end_year + 1)
#     mask = (sim.timevec >= init_year) & (sim.timevec <= end_year)

#     # Calculate the mean prevalence over the specified range of years
#     mean_prevalence_male_over_years = np.mean(mean_prevalence_male[mask])
#     mean_prevalence_female_over_years = np.mean(mean_prevalence_female[mask])
#     print(f"Mean Male Prevalence for {disease} from {init_year} to {end_year}: {mean_prevalence_male_over_years:.2f}%")
#     print(f"Mean Female Prevalence for {disease} from {init_year} to {end_year}: {mean_prevalence_female_over_years:.2f}%")

#     # Create figure
#     plt.figure(figsize=(10, 5))

#     # Plot mean prevalence for males and females
#     plt.plot(sim.timevec[mask], mean_prevalence_male[mask], label=f'Male {disease.capitalize()} Prevalence (Total)', linewidth=5, color='blue', linestyle='solid')
#     plt.plot(sim.timevec[mask], mean_prevalence_female[mask], label=f'Female {disease.capitalize()} Prevalence (Total)', linewidth=5, color='red', linestyle='solid')

#     # Plot observed prevalence data if available
#     if prevalence_data_df is not None:
#         male_col = f'{disease}_male'
#         female_col = f'{disease}_female'

#         # Check if columns for observed male and female prevalence exist
#         if male_col in prevalence_data_df.columns:
#             # Drop NaN values from both Year and male prevalence data
#             observed_male_data = prevalence_data_df[['Year', male_col]].dropna()
#             observed_male_data = observed_male_data.groupby('Year', as_index=False).mean()
#             observed_male_data[male_col] *= 100
#             print(f"Observed Male Data for {disease}:\n", observed_male_data)

#             # Filter observed data based on init_year and end_year
#             observed_male_data = observed_male_data[(observed_male_data['Year'] >= init_year) & (observed_male_data['Year'] <= end_year)]

#             # Plot observed male data
#             plt.scatter(observed_male_data['Year'], observed_male_data[male_col], 
#                         color='blue', marker='o', edgecolor='black', s=100, 
#                         label='Observed Male Prevalence')

#         if female_col in prevalence_data_df.columns:
#             # Drop NaN values from both Year and female prevalence data
#             observed_female_data = prevalence_data_df[['Year', female_col]].dropna()
#             observed_female_data = observed_female_data.groupby('Year', as_index=False).mean()
#             observed_female_data[female_col] *= 100

#             # Filter observed data based on init_year and end_year
#             observed_female_data = observed_female_data[(observed_female_data['Year'] >= init_year) & (observed_female_data['Year'] <= end_year)]

#             # Plot observed female data
#             plt.scatter(observed_female_data['Year'], observed_female_data[female_col], 
#                         color='red', marker='o', edgecolor='black', s=100, 
#                         label='Observed Female Prevalence')
        
#     # Labels and title
#     plt.legend()
#     plt.xlabel('Year')
#     plt.ylabel(f'{disease.capitalize()} Prevalence (%)')
#     plt.title(f'Mean {disease.capitalize()} Prevalence Over Time (All Ages)')
#     plt.grid()
    
#     plt.show()


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
        # Default age groups
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

    # Define age bins and colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(age_groups)))

    # Extract results for each age bin
    def extract_results(key_pattern):
        return [prevalence_analyzer.results.get(f'{disease}_{key_pattern}_{i}', np.zeros(len(sim.timevec))) for i in range(len(age_groups))]

    male_num_bins = extract_results('num_male')
    female_num_bins = extract_results('num_female')
    male_den_bins = extract_results('den_male')
    female_den_bins = extract_results('den_female')

    # Create figure with two panels
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), sharey=True)
    ax_male, ax_female = axes

    # Plot simulated prevalence for each age group
    for i, (start_age, end_age, label) in enumerate(age_groups):
        color = colors[i]
        
        # Simulated prevalence
        male_prevalence = np.nan_to_num(male_num_bins[i] / male_den_bins[i]) * 100
        female_prevalence = np.nan_to_num(female_num_bins[i] / female_den_bins[i]) * 100

        # Filter the data based on init_year and end_year
        mask = (sim.timevec >= init_year) & (sim.timevec <= end_year)

        # Plot male prevalence
        ax_male.plot(sim.timevec[mask], male_prevalence[mask], label=label, color=color, linestyle='solid')
        
        # Plot female prevalence
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
                                color=color, marker='o', s=150)  # Increased dot size
                observed_lines.append(obs_line)

            if female_col in prevalence_data_df.columns:
                observed_female_data = prevalence_data_df[age_mask][['Year', female_col]].dropna()
                observed_female_data = observed_female_data.groupby('Year', as_index=False).mean()
                observed_female_data[female_col] *= 100

                observed_female_data = observed_female_data[(observed_female_data['Year'] >= init_year) & (observed_female_data['Year'] <= end_year)]

                ax_female.scatter(observed_female_data['Year'], observed_female_data[female_col], 
                                  color=color, marker='o', s=150)  # Increased dot size

    # Labels and title for male panel
    ax_male.set_xlabel('Year')
    ax_male.set_ylabel(f'{disease.capitalize()} Prevalence (%)')
    ax_male.set_title(f'Male {disease.capitalize()} Prevalence by Age Group')
    ax_male.grid()

    # Labels and title for female panel
    ax_female.set_xlabel('Year')
    ax_female.set_ylabel(f'{disease.capitalize()} Prevalence (%)')
    ax_female.set_title(f'Female {disease.capitalize()} Prevalence by Age Group')
    ax_female.grid()

    # Create a single set of legends
    lines, labels = ax_male.get_legend_handles_labels()
    unique_labels = {label: line for label, line in zip(labels, lines)}
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', ncol=5)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()
    
##### Plot fucntions for demography related plots #####
    
def plot_metrics(df):
    """
    Plot age-group and sex-dependent birth and death rates over time.
    
    Args:
        df (pd.DataFrame): DataFrame containing the simulation results with calculated metrics.
    """
    # Define age groups and colors
    age_groups = {
        '0': '0 year old',
        '1-4': '1-4 years old',
        '5-50': '5-50 years old',
        '51-70': '51-70 years old',
        '71-100': '71-100 years old'
    }
    colors = ['blue', 'green', 'orange', 'purple', 'brown']

    fig, ax = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    for i, (age_group, label) in enumerate(age_groups.items()):
        color = colors[i]

        # Plot male birth rates
        ax[0].plot(df['year'], df[f'male_population_{age_group}'], label=label, color=color)
        ax[0].set_ylabel('Male Population')
        ax[0].legend()

        # Plot male death rates
        ax[1].plot(df['year'], df[f'male_death_rate_{age_group}'], label=label, color=color)
        ax[1].set_ylabel('Male Death Rate')
        ax[1].legend()

        # Plot female birth rates
        ax[2].plot(df['year'], df[f'female_population_{age_group}'], label=label, color=color)
        ax[2].set_ylabel('Female Population')
        ax[2].legend()

        # Plot female death rates
        ax[3].plot(df['year'], df[f'female_death_rate_{age_group}'], label=label, color=color)
        ax[3].set_xlabel('Year')
        ax[3].set_ylabel('Female Death Rate')
        ax[3].legend()

    plt.tight_layout()
    plt.show()

def plot_aggregated_death_counts(death_tracker, age_groups=None):
    """
    Plot death counts aggregated into standard age groups
    
    Args:
        death_tracker: DeathTracker analyzer object
        age_groups: List of tuples (start_age, end_age, label)
        
    Returns:
        Figure axis object
    """
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
    
    # Get death counts
    death_counts = death_tracker.get_death_counts()
    
    # Aggregate deaths by age group
    male_group_counts = []
    female_group_counts = []
    
    for start_age, end_age, _ in age_groups:
        # Sum deaths in this age group
        male_sum = sum(death_counts['Male'].get(age, 0) for age in range(start_age, end_age))
        female_sum = sum(death_counts['Female'].get(age, 0) for age in range(start_age, end_age))
        
        male_group_counts.append(male_sum)
        female_group_counts.append(female_sum)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Set up bar positions
    labels = [label for _, _, label in age_groups]
    x = np.arange(len(labels))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, male_group_counts, width, label='Male', color='blue', alpha=0.7)
    plt.bar(x + width/2, female_group_counts, width, label='Female', color='red', alpha=0.7)
    
    # Add data labels
    for i, count in enumerate(male_group_counts):
        plt.text(i - width/2, count + 5, str(count), ha='center', va='bottom', fontsize=10)
    for i, count in enumerate(female_group_counts):
        plt.text(i + width/2, count + 5, str(count), ha='center', va='bottom', fontsize=10)
    
    # Customize the plot
    plt.grid(True, alpha=0.3, axis='y')
    plt.xlabel('Age Group', fontsize=14)
    plt.ylabel('Number of Deaths', fontsize=14)
    plt.title(f'Death Counts by Age Group and Sex (Total: {death_tracker.total_deaths_tracked})', fontsize=16)
    plt.xticks(x, labels, fontsize=12)
    plt.legend(fontsize=12)
    
    # Improve layout
    plt.tight_layout()

def plot_simulated_death_counts(death_tracker, year=None, age_interval=5, smoothing=0, figsize=(12, 8)):
    """
    Plot absolute simulated death counts by age and sex
    
    Args:
        death_tracker: The DeathTracker instance
        year: Specific year to plot (None for cumulative deaths)
        age_interval: Group ages by this interval (e.g., 5 for 5-year age groups)
        smoothing: Sigma value for Gaussian smoothing (0 for no smoothing)
        figsize: Figure size (width, height) in inches
    """
    # Get death counts for specified year or cumulative
    death_counts = death_tracker.get_death_counts(year)
    
    # Group ages if requested
    if age_interval > 1:
        grouped_male = {}
        grouped_female = {}
        for age_start in range(0, 101, age_interval):
            age_end = min(age_start + age_interval, 101)
            age_label = age_start
            
            # Sum deaths in this age group
            male_sum = sum(death_counts['Male'].get(age, 0) for age in range(age_start, age_end))
            female_sum = sum(death_counts['Female'].get(age, 0) for age in range(age_start, age_end))
            
            grouped_male[age_label] = male_sum
            grouped_female[age_label] = female_sum
        
        # Replace with grouped data
        death_counts = {
            'Male': grouped_male,
            'Female': grouped_female
        }
    
    # Extract ages and death counts
    ages = sorted(death_counts['Male'].keys())
    male_counts = [death_counts['Male'][age] for age in ages]
    female_counts = [death_counts['Female'][age] for age in ages]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot male and female counts
    plt.plot(ages, male_counts, 'b-', marker='o', markersize=6,
             linewidth=2, alpha=0.7, label='Male')
    plt.plot(ages, female_counts, 'r-', marker='o', markersize=6,
             linewidth=2, alpha=0.7, label='Female')
    
    # Add smoothed trend lines if requested
    if smoothing > 0:
        try:
            from scipy.ndimage import gaussian_filter1d
            male_smooth = gaussian_filter1d(male_counts, sigma=smoothing)
            female_smooth = gaussian_filter1d(female_counts, sigma=smoothing)
            plt.plot(ages, male_smooth, 'b-', linewidth=3, alpha=1, label='Male (Smoothed)')
            plt.plot(ages, female_smooth, 'r-', linewidth=3, alpha=1, label='Female (Smoothed)')
        except ImportError:
            print("scipy not available for smoothing")
    
    # Set appropriate title based on year
    if year is None:
        title = "Cumulative Simulated Death Counts by Age and Sex"
    else:
        title = f"Simulated Death Counts for Year {year} by Age and Sex"
    
    # Customize the plot
    plt.grid(True, alpha=0.3)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Number of Deaths', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    
    # Set axis limits
    plt.xlim(0, max(ages))
    plt.ylim(bottom=0)
    
    # Improve layout
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    total_male_deaths = sum(male_counts)
    total_female_deaths = sum(female_counts)
    print(f"\nSummary of simulated death counts:")
    print(f"Total deaths - Male: {total_male_deaths}, Female: {total_female_deaths}")
    
    # Return the death counts data for further use
    return death_counts


def plot_death_counts_comparison(death_tracker, observed_death_csv, year, age_interval=5, figsize=(14, 10)):
    """
    Plot simulated death counts and overlay observed death counts from data
    
    Args:
        death_tracker: The DeathTracker instance
        observed_death_csv: Path to CSV file with observed death data
                          (This file contains deaths in thousands)
        year: Year to plot and compare (must be a year that exists in observed data)
        age_interval: Group ages by this interval (e.g., 5 for 5-year age groups)
        figsize: Figure size (width, height) in inches
    """
    # Get simulated death counts
    sim_death_counts = death_tracker.get_death_counts(year)
    
    # Load observed death data (values are in thousands)
    try:
        observed_death = pd.read_csv(observed_death_csv)
    except Exception as e:
        print(f"Error loading observed death data: {e}")
        return
    
    # Check if the requested year exists in the data
    year_str = str(year)
    if year_str not in observed_death.columns:
        print(f"Year {year} not found in observed death data")
        available_years = [col for col in observed_death.columns if col not in ['age', 'sex']]
        print(f"Available years: {available_years}")
        return
    
    # Group simulated deaths if requested
    if age_interval > 1:
        grouped_sim_male = {}
        grouped_sim_female = {}
        for age_start in range(0, 101, age_interval):
            age_end = min(age_start + age_interval, 101)
            age_label = age_start
            
            # Sum deaths in this age group
            male_sum = sum(sim_death_counts['Male'].get(age, 0) for age in range(age_start, age_end))
            female_sum = sum(sim_death_counts['Female'].get(age, 0) for age in range(age_start, age_end))
            
            grouped_sim_male[age_label] = male_sum
            grouped_sim_female[age_label] = female_sum
        
        # Replace with grouped data
        sim_death_counts = {
            'Male': grouped_sim_male,
            'Female': grouped_sim_female
        }
    
    # Extract simulated ages and death counts
    sim_ages = sorted(sim_death_counts['Male'].keys())
    sim_male_counts = [sim_death_counts['Male'][age] for age in sim_ages]
    sim_female_counts = [sim_death_counts['Female'][age] for age in sim_ages]
    
    # Get current simulation population and its structure
    total_sim_population = len(death_tracker.sim.people)
    sim_alive = death_tracker.sim.people.alive
    sim_female = death_tracker.sim.people.female
    sim_male_pop = (~sim_female & sim_alive).sum()
    sim_female_pop = (sim_female & sim_alive).sum()
    
    # Calculate the estimated Eswatini population based on the UN data
    # Assuming the death data is representative of a ~1.2 million population
    # These are rough estimates based on Eswatini demographics
    eswatini_pop = 1_200_000
    eswatini_male_pop = 600_000
    eswatini_female_pop = 600_000
    
    # Calculate scaling factors for males and females separately
    male_scaling = (eswatini_male_pop / sim_male_pop) / 1000  # Divide by 1000 because observed data is in thousands
    female_scaling = (eswatini_female_pop / sim_female_pop) / 1000
    
    # Scale simulated counts to match observed data units (thousands) and population size
    sim_male_counts_scaled = [count * male_scaling for count in sim_male_counts]
    sim_female_counts_scaled = [count * female_scaling for count in sim_female_counts]
    
    # Group observed deaths in the same age intervals
    observed_male = observed_death[observed_death['sex'] == 'Male']
    observed_female = observed_death[observed_death['sex'] == 'Female']
    
    obs_male_counts = []
    obs_female_counts = []
    obs_ages = []
    
    for age_start in range(0, 101, age_interval):
        age_end = min(age_start + age_interval, 101)
        
        # Filter observed data for this age group
        male_age_group = observed_male[(observed_male['age'] >= age_start) & (observed_male['age'] < age_end)]
        female_age_group = observed_female[(observed_female['age'] >= age_start) & (observed_female['age'] < age_end)]
        
        # Sum deaths in this age group
        male_sum = male_age_group[year_str].sum() if not male_age_group.empty else 0
        female_sum = female_age_group[year_str].sum() if not female_age_group.empty else 0
        
        obs_ages.append(age_start)
        obs_male_counts.append(male_sum)
        obs_female_counts.append(female_sum)
    
    # Create figure with two panels for male and female
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot male data
    ax1.bar(obs_ages, obs_male_counts, width=age_interval*0.8, alpha=0.6, 
            color='blue', label=f'Observed Male ({year}) [thousands]')
    ax1.plot(sim_ages, sim_male_counts_scaled, 'bo-', markersize=6, linewidth=2,
             label=f'Simulated Male (scaled) [thousands]')
    
    ax1.set_title(f'Male Death Counts Comparison for {year}', fontsize=16)
    ax1.set_ylabel('Deaths (thousands)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot female data
    ax2.bar(obs_ages, obs_female_counts, width=age_interval*0.8, alpha=0.6,
            color='red', label=f'Observed Female ({year}) [thousands]')
    ax2.plot(sim_ages, sim_female_counts_scaled, 'ro-', markersize=6, linewidth=2, 
             label=f'Simulated Female (scaled) [thousands]')
    
    ax2.set_title(f'Female Death Counts Comparison for {year}', fontsize=16)
    ax2.set_xlabel('Age', fontsize=14)
    ax2.set_ylabel('Deaths (thousands)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Improve layout
    plt.tight_layout()
    plt.show()
    
    # Calculate death rates for comparison
    total_obs_male_deaths = sum(obs_male_counts) * 1000  # Convert back from thousands
    total_obs_female_deaths = sum(obs_female_counts) * 1000
    
    obs_male_rate = total_obs_male_deaths / eswatini_male_pop
    obs_female_rate = total_obs_female_deaths / eswatini_female_pop
    
    total_sim_male_deaths = sum(sim_male_counts)
    total_sim_female_deaths = sum(sim_female_counts)
    
    sim_male_rate = total_sim_male_deaths / sim_male_pop
    sim_female_rate = total_sim_female_deaths / sim_female_pop
    
    # Print summary statistics
    print(f"\nSummary of death counts comparison for year {year}:")
    print(f"Observed deaths - Male: {sum(obs_male_counts):.3f}k, Female: {sum(obs_female_counts):.3f}k")
    print(f"Simulated deaths - Male: {total_sim_male_deaths}, Female: {total_sim_female_deaths}")
    
    print(f"\nDeath rates comparison:")
    print(f"Observed - Male: {obs_male_rate:.4f}, Female: {obs_female_rate:.4f}")
    print(f"Simulated - Male: {sim_male_rate:.4f}, Female: {sim_female_rate:.4f}")
    print(f"Rate ratio - Male: {sim_male_rate/obs_male_rate:.2f}x, Female: {sim_female_rate/obs_female_rate:.2f}x")
    
    print(f"\nScaling factors applied:")
    print(f"Male: {male_scaling:.6f}, Female: {female_scaling:.6f}")
    
    # Return the comparison data
    comparison_data = {
        'simulated': {
            'male': sim_male_counts,
            'female': sim_female_counts,
            'male_scaled': sim_male_counts_scaled,
            'female_scaled': sim_female_counts_scaled,
            'ages': sim_ages
        },
        'observed': {
            'male': obs_male_counts,
            'female': obs_female_counts,
            'ages': obs_ages
        },
        'scaling_factors': {'male': male_scaling, 'female': female_scaling}
    }
    
    return comparison_data


def plot_life_expectancy(life_table, observed_data, year, max_age=100, figsize=(14, 10), title=None):
    """
    Plot life expectancy for each age for a given year in two panels: one for males and one for females.
    Overlay observed life expectancy data with simulated data.
    
    Args:
        life_table: DataFrame containing the complete life table
        observed_data: DataFrame containing the observed life expectancy data
        year: Year to filter the data
        max_age: Maximum age to consider (default 100)
        figsize: Figure size (width, height) in inches
        title: Custom title for the plot
        
    Returns:
        Figure and axes objects
    """
    # Filter the life table for the given year
    if 'Time' in life_table.columns:
        life_table = life_table[life_table['Time'] == year]
    
    # Separate life table by sex
    male_life_table = life_table[life_table['sex'] == 'Male']
    female_life_table = life_table[life_table['sex'] == 'Female']
    
    # Filter observed data for the given year
    observed_data_year = observed_data[['age', 'sex', str(year)]].copy()
    observed_male = observed_data_year[observed_data_year['sex'] == 'Male']
    observed_female = observed_data_year[observed_data_year['sex'] == 'Female']
    
    # Create figure with two panels for male and female
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot male life expectancy
    ax1.plot(male_life_table['Age'], male_life_table['e(x)'], linestyle='-', linewidth=8, alpha=0.4, color='blue', label='Simulated')
    ax1.plot(observed_male['age'], observed_male[str(year)], marker='s', linestyle='--', linewidth=2, markersize=8, color='blue', label='Observed')
    ax1.set_title('Male', fontsize=28)
    ax1.set_ylabel('Life Expectancy (years)', fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=24)
    
    # Plot female life expectancy
    ax2.plot(female_life_table['Age'], female_life_table['e(x)'], linestyle='-', linewidth=8, alpha=0.4, color='red', label='Simulated')
    ax2.plot(observed_female['age'], observed_female[str(year)], marker='s', linestyle='--', linewidth=2, markersize=8, color='red', label='Observed')
    ax2.set_title('Female', fontsize=28)
    ax2.set_xlabel('Age', fontsize=24)
    ax2.set_ylabel('Life Expectancy (years)', fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=24)
    
    # Set overall title if provided
    if title:
        plt.suptitle(title, fontsize=24, y=0.98)
    else:
        plt.suptitle(f'Life Expectancy by Age for {year}', fontsize=24, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(bottom=0.1)
    plt.show()
    
    return fig, (ax1, ax2)


def plot_imr(observed_data_path, simulated_data, start_year, end_year):
    # Load observed data
    observed_data = pd.read_csv(observed_data_path)

    # Filter observed data for infants (AgeGrpStart == 0) and the given year range
    observed_data_infants = observed_data[(observed_data['AgeGrpStart'] == 0) & 
                                          (observed_data['Time'] >= start_year) & 
                                          (observed_data['Time'] <= end_year)]

    # Function to get average IMR from observed data
    def get_average_observed_imr(data):
        imr_data = data.groupby('Time')['mx'].mean().reset_index()
        return imr_data

    # Get average observed IMR
    observed_imr = get_average_observed_imr(observed_data_infants)

    # Load simulated data
    simulated_data = simulated_data

    # Plot IMR for both sexes
    plt.figure(figsize=(10, 5))
    plt.plot(observed_imr['Time'], observed_imr['mx'], label='Observed IMR (Both Sexes)', marker='o')
    plt.plot(simulated_data['Year'], simulated_data['IMR'], label='Simulated IMR (Both Sexes)', marker='x')
    plt.xlabel('Year')
    plt.ylabel('IMR (per 1,000 live births)')
    plt.ylim(0, 0.1)  # Setting y-axis from 0 to 0.1 to fit the observed IMR values
    plt.title('Infant Mortality Rate (IMR) - Both Sexes')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# def plot_mortality_rates_comparison(df_metrics, observed_data_path, observed_year=None, year=None, 
#                                     log_scale=True, figsize=(14, 10), title=None):
#     """
#     Plot comparison between simulated and observed mortality rates
    
#     Args:
#         df_metrics: DataFrame with calculated mortality rates (mx)
#         observed_data_path: Path to the CSV file with observed mortality rates
#                             (columns: Time, Sex, AgeGrpStart, mx)
#         observed_year: Year to filter observed data (if None, will use simulated year)
#         year: Year to filter simulated data (if None, will use Time column from data)
#         log_scale: Whether to use log scale for mortality rates
#         figsize: Figure size (width, height) in inches
#         title: Custom title for the plot
        
#     Returns:
#         Figure and axes objects
#     """
#     # Load observed data
#     observed_rates = pd.read_csv(observed_data_path)

#     # Filter simulated data if year provided
#     if year is not None:
#         simulated_rates = df_metrics[df_metrics['year'] == year].copy()
#     else:
#         # Try to get year from the data
#         years = df_metrics['year'].unique()
#         if len(years) == 1:
#             year = years[0]
#         simulated_rates = df_metrics[df_metrics['year'] == year].copy()
#         simulated_rates[simulated_rates['age']>90]
    
#     # Filter observed data by year
#     if observed_year is None and year is not None:
#         observed_year = year
        
#     if observed_year is not None:
#         observed_rates = observed_rates[observed_rates['Time'] == observed_year].copy()

#     # Extract unique age groups from observed data
#     unique_ages = observed_rates['AgeGrpStart'].unique()
#     # Create figure with two panels for male and female
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     # Plot male data
#     male_sim = simulated_rates[simulated_rates['sex'] == 'Male'].sort_values('age')
#     male_obs = observed_rates[observed_rates['Sex'] == 'Male'].sort_values('AgeGrpStart')
    
#     # Plot male mortality rates
#     ax1.plot(male_sim['age'], male_sim['mx'], 'b-',   
#              linewidth=6, alpha=0.5, label='Simulated Male')
#     ax1.plot(male_obs['AgeGrpStart'], male_obs['mx'], 'b--', marker='s', markersize=8,
#              linewidth=2, label='Observed Male')
    
#     # Plot female data
#     female_sim = simulated_rates[simulated_rates['sex'] == 'Female'].sort_values('age')
#     female_obs = observed_rates[observed_rates['Sex'] == 'Female'].sort_values('AgeGrpStart')
    
#     # Plot female mortality rates
#     ax2.plot(female_sim['age'], female_sim['mx'], 'r-',
#              linewidth=6, alpha=0.5, label='Simulated Female')
#     ax2.plot(female_obs['AgeGrpStart'], female_obs['mx'], 'r--', marker='s', markersize=8, 
#              linewidth=2,  label='Observed Female')
    
#     # Set axis scales
#     if log_scale:
#         ax1.set_yscale('log')
#         ax2.set_yscale('log')
#     else:
#         ax1.set_ylim(0, 1)
#         ax2.set_ylim(0, 1)        
    
#     # Set titles and labels for male panel
#     ax1.set_title(f'Male', fontsize=28)
#     ax1.set_ylabel('Mortality Rate (mx)', fontsize=24)
#     ax1.tick_params(axis='both', which='major', labelsize=20)
#     ax1.grid(True, which='both', alpha=0.3)
    
    
#     # Set titles and labels for female panel
#     ax2.set_title(f'Female', fontsize=28)
#     ax2.set_xlabel('Age', fontsize=24)
#     ax2.set_ylabel('Mortality Rate (mx)', fontsize=24)
#     ax2.tick_params(axis='both', which='major', labelsize=20)
#     ax2.grid(True, which='both', alpha=0.3)

#     # Set overall title if provided
#     if title:
#         plt.suptitle(title, fontsize=28, y=0.98)
#     else:
#         plt.suptitle(f'Mortality Rates Comparison: Simulated vs Observed', fontsize=30, y=0.98)


#     # Create unified legend
#     handles, labels = [], []
#     for ax in [ax1, ax2]:
#         for handle, label in zip(*ax.get_legend_handles_labels()):
#             if label not in labels:
#                 handles.append(handle)
#                 labels.append(label)

#     fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=20)
    
#     # Improve layout
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.subplots_adjust(bottom=0.15)
#     plt.show()
    
#     return fig, (ax1, ax2)

def plot_mortality_rates_comparison(df_metrics, observed_data, observed_year=None, year=None, 
                                               log_scale=True, figsize=(14, 10), title=None):
    """
    Compare simulated and observed mortality rates using single-age data.

    Args:
        df_metrics (pd.DataFrame): Simulated data with columns ['year', 'sex', 'age', 'mx'].
        observed_data (pd.DataFrame): Observed data with columns ['Time', 'Sex', 'Age', 'mx'].
        observed_year (int, optional): Year of observed data to filter. Defaults to None.
        year (int, optional): Year of simulated data to filter. Defaults to None.
        log_scale (bool, optional): Whether to use a logarithmic scale for mortality rates. Defaults to True.
        figsize (tuple, optional): Figure size for the plots. Defaults to (14, 10).
        title (str, optional): Custom title for the plot. Defaults to None.

    Returns:
        fig, (ax1, ax2): Matplotlib figure and axes objects.
    """
    # Load observed data
    observed_rates = observed_data

    # Filter simulated data if year provided
    if year is not None:
        simulated_rates = df_metrics[df_metrics['year'] == year].copy()
    else:
        # Try to get year from the data
        years = df_metrics['year'].unique()
        if len(years) == 1:
            year = years[0]
        simulated_rates = df_metrics[df_metrics['year'] == year].copy()
    
    # Filter observed data by year
    if observed_year is None and year is not None:
        observed_year = year
        
    if observed_year is not None:
        observed_rates = observed_rates[observed_rates['Time'] == observed_year].copy()

    # Extract unique ages from observed data
    unique_ages = observed_rates['Age'].unique()
    
    # Create figure with two panels for male and female
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Plot male data
    male_sim = simulated_rates[simulated_rates['sex'] == 'Male'].sort_values('age')
    male_obs = observed_rates[observed_rates['Sex'] == 'Male'].sort_values('Age')
    
    # Plot male mortality rates
    ax1.plot(male_sim['age'], male_sim['mx'], 'b-',   
             linewidth=6, alpha=0.5, label='Simulated Male')
    ax1.plot(male_obs['Age'], male_obs['mx'], 'b--', marker='s', markersize=8,
             linewidth=2, label='Observed Male')
    
    # Plot female data
    female_sim = simulated_rates[simulated_rates['sex'] == 'Female'].sort_values('age')
    female_obs = observed_rates[observed_rates['Sex'] == 'Female'].sort_values('Age')
    
    # Plot female mortality rates
    ax2.plot(female_sim['age'], female_sim['mx'], 'r-',
             linewidth=6, alpha=0.5, label='Simulated Female')
    ax2.plot(female_obs['Age'], female_obs['mx'], 'r--', marker='s', markersize=8, 
             linewidth=2,  label='Observed Female')
    
    # Set axis scales
    if log_scale:
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    else:
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)        
    
    # Set titles and labels for male panel
    ax1.set_title(f'Male', fontsize=28)
    ax1.set_ylabel('Mortality Rate (mx)', fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.grid(True, which='both', alpha=0.3)
    
    # Set titles and labels for female panel
    ax2.set_title(f'Female', fontsize=28)
    ax2.set_xlabel('Age', fontsize=24)
    ax2.set_ylabel('Mortality Rate (mx)', fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.grid(True, which='both', alpha=0.3)

    # Set overall title if provided
    if title:
        plt.suptitle(title, fontsize=28, y=0.98)
    else:
        plt.suptitle(f'Mortality Rates Comparison: Simulated vs Observed', fontsize=30, y=0.98)

    # Create unified legend
    handles, labels = [], []
    for ax in [ax1, ax2]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=20)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(bottom=0.15)
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

    # Load observed data if path is provided
    observed_data = None
    if observed_data_path:
        try:
            observed_data = pd.read_csv(observed_data_path)
            print(f"Loaded observed age distribution data from {observed_data_path}")
        except Exception as e:
            print(f"Could not load observed data: {str(e)}")
    
    # Initialize dictionaries to store population data
    population_data = {
        'years': list(range(inityear, endyear + 1)),
        'male': {group[2]: [] for group in age_groups},
        'female': {group[2]: [] for group in age_groups}
    }
    
    # Calculate population by age group for each year
    for year in population_data['years']:
        if year not in df['year'].values:
            continue
            
        # Calculate for each age group
        for start_age, end_age, label in age_groups:
            # Calculate male population in this age group
            male_count = sum(df[df['year'] == year][f'male_population_age_{age}'] / nagent * 1000
                             for age in range(start_age, end_age + 1) if f'male_population_age_{age}' in df.columns)
            population_data['male'][label].append(male_count)
            
            # Calculate female population in this age group
            female_count = sum(df[df['year'] == year][f'female_population_age_{age}'] / nagent * 1000
                               for age in range(start_age, end_age + 1) if f'female_population_age_{age}' in df.columns)
            population_data['female'][label].append(female_count)
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot male population
    for label, counts in population_data['male'].items():
        if len(counts) == len(population_data['years']):
            ax1.plot(population_data['years'], counts, '-o', linewidth=2, label=f'Sim: {label}')
    
    # Plot female population
    for label, counts in population_data['female'].items():
        if len(counts) == len(population_data['years']):
            ax2.plot(population_data['years'], counts, '-o', linewidth=2, label=f'Sim: {label}')
    
    # Add observed data if available
    if observed_data is not None:
        # Get years as columns that exist in both data and our range
        observed_years = [str(y) for y in range(inityear, endyear + 1) 
                          if str(y) in observed_data.columns]
        
        if observed_years:
            # Process observed data for males and females by age group
            obs_male = {}
            obs_female = {}
            
            for start_age, end_age, label in age_groups:
                # Get male data for this age group by summing across ages
                male_rows = observed_data[(observed_data['sex'] == 'Male') & 
                                          (observed_data['age'] >= start_age) & 
                                          (observed_data['age'] <= end_age)]
                
                # Get female data for this age group
                female_rows = observed_data[(observed_data['sex'] == 'Female') & 
                                            (observed_data['age'] >= start_age) & 
                                            (observed_data['age'] <= end_age)]
                
                # Initialize data lists
                obs_male[label] = []
                obs_female[label] = []
                
                # Extract values for each year
                for year_str in observed_years:
                    # Sum populations for this age group and year
                    male_pop = male_rows[year_str].sum()
                    female_pop = female_rows[year_str].sum() 
                    
                    obs_male[label].append(male_pop)
                    obs_female[label].append(female_pop)
            
            # Convert year strings to integers for plotting
            obs_year_ints = [int(y) for y in observed_years]
            
            # Plot observed male data
            for label, counts in obs_male.items():
                if len(counts) > 0:
                    ax1.plot(obs_year_ints, counts, '--s', linewidth=1.5, alpha=0.7, 
                            label=f'Obs: {label}')
            
            # Plot observed female data
            for label, counts in obs_female.items():
                if len(counts) > 0:
                    ax2.plot(obs_year_ints, counts, '--s', linewidth=1.5, alpha=0.7, 
                            label=f'Obs: {label}')
            
            print(f"Added observed data for years: {', '.join(observed_years)}")
        else:
            print("No matching years found in observed data")
    
    # Configure male panel
    ax1.set_title('Male Population by Age Group', fontsize=14)
    ax1.set_ylabel('Population Count (Thousands)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    # ax1.legend(fontsize=10, loc='upper left')
    
    # Configure female panel
    ax2.set_title('Female Population by Age Group', fontsize=14)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Population Count (Thousands)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    # ax2.legend(fontsize=10, loc='upper left')
    
    # Create unified legend
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