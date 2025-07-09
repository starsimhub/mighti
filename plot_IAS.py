import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




##############################################################



def extract_t2d_prevalence_by_age_plhiv(sim, prevalence_analyzer, disease, target_years, age_groups):
    """
    Extract T2D prevalence among PLHIV for each age group and sex at specified years.
    """
    year_to_index = {int(round(year)): idx for idx, year in enumerate(sim.timevec)}
    selected_indices = {int(year): year_to_index[int(year)] for year in target_years if int(year) in year_to_index}

    def extract_results(key_pattern):
        return [
            prevalence_analyzer.results.get(f'{disease}_{key_pattern}_{i}', np.zeros(len(sim.timevec)))
            for i in range(len(age_groups))
        ]

    male_num = extract_results('num_with_HIV_male')
    female_num = extract_results('num_with_HIV_female')
    male_den = extract_results('den_with_HIV_male')
    female_den = extract_results('den_with_HIV_female')

    male_data, female_data = [], []
    for year in target_years:
        ti = selected_indices.get(int(year))
        if ti is None:
            continue

        male_row, female_row = [], []
        for i in range(len(age_groups)):
            m_num = male_num[i][ti]
            m_den = male_den[i][ti]
            f_num = female_num[i][ti]
            f_den = female_den[i][ti]

            male_row.append((m_num / m_den * 100) if m_den > 0 else 0)
            female_row.append((f_num / f_den * 100) if f_den > 0 else 0)

        male_data.append(male_row)
        female_data.append(female_row)

    age_labels = [label for _, _, label in age_groups]
    df_male = pd.DataFrame(male_data, index=target_years, columns=age_labels).T
    df_female = pd.DataFrame(female_data, index=target_years, columns=age_labels).T
    return df_male, df_female


def extract_combined_t2d_prevalence_by_age_plhiv(sim, prevalence_analyzer, disease, target_years, custom_age_groups):
    """
    Extract combined (male + female) T2D prevalence among PLHIV by custom age group and year.
    Uses fixed 15 bins from PrevalenceAnalyzer and aggregates to match custom bins.
    """
    import numpy as np
    import pandas as pd
    import sciris as sc

    # Define the fixed bins used in the PrevalenceAnalyzer
    analyzer_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45),
                     (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, float('inf'))]

    # Create mapping from custom group to list of indices in analyzer_bins
    bin_map = {}
    for i, (a_start, a_end, label) in enumerate(custom_age_groups):
        print(i)
        print(f'label is {label}, astart is {a_start}, aend is {a_end}')
        indices = [
            j for j, (b_start, b_end) in enumerate(analyzer_bins)
            if not (b_end <= a_start or b_start >= a_end)
        ]
        bin_map[label] = indices

    year_to_index = {int(round(year)): idx for idx, year in enumerate(sim.timevec)}
    selected_indices = {int(year): year_to_index[int(year)] for year in target_years if int(year) in year_to_index}

    combined_data = []

    for year in target_years:
        ti = selected_indices.get(int(year))
        if ti is None:
            continue

        row = []
        for label, indices in bin_map.items():
            m_num = sum(prevalence_analyzer.results[f'{disease}_num_with_HIV_male_{i}'][ti] for i in indices)
            m_den = sum(prevalence_analyzer.results[f'{disease}_den_with_HIV_male_{i}'][ti] for i in indices)
            f_num = sum(prevalence_analyzer.results[f'{disease}_num_with_HIV_female_{i}'][ti] for i in indices)
            f_den = sum(prevalence_analyzer.results[f'{disease}_den_with_HIV_female_{i}'][ti] for i in indices)

            total_num = m_num + f_num
            total_den = m_den + f_den
            prevalence = (total_num / total_den * 100) if total_den > 0 else 0
            row.append(prevalence)
            print(f'yearis {year}: prevalence is {prevalence}')

        combined_data.append(row)

    df_combined = pd.DataFrame(combined_data, index=target_years, columns=bin_map.keys()).T
    return df_combined


def plot_grouped_prevalence_bar(df, sex_label, colors):
    """
    Plot grouped bar chart for T2D among PLHIV by age group over years.
    Args:
        df (pd.DataFrame): age group x year prevalence (%)
        sex_label (str): 'Male' or 'Female'
        colors (list): list of colors for each age group
    """
    labels = df.columns.tolist()
    age_groups = df.index.tolist()
    x = np.arange(len(labels))  # positions for years
    width = 0.15

    fig, ax = plt.subplots(figsize=(10,5)) # larger figure

    for i, age_group in enumerate(age_groups):
        ax.bar(x + i * width, df.loc[age_group], width, label=age_group, color=colors[i])

    ax.set_xlabel('Year', fontsize=22, fontweight='bold')
    ax.set_ylabel('Type 2 Diabetes Prevalence (%)', fontsize=22, fontweight='bold')
    # ax.set_title(f'T2D Prevalence Among PLHIV by Age Group — {sex_label}', fontsize=24, fontweight='bold')
    ax.set_xticks(x + width * (len(age_groups) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_yticks(np.arange(0, 41, 5))
    ax.set_ylim(0, 40)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)

    ax.legend(title='Age Group', fontsize=20, title_fontsize=18, loc='upper left', frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.6)


    plt.tight_layout()
    plt.show()
    
    
def plot_grouped_prevalence_bar_combined(df, colors=None):
    """
    Plot grouped bar chart for combined (male + female) T2D prevalence among PLHIV by age group over selected years.

    Args:
        df (pd.DataFrame): DataFrame with age groups as rows and years as columns (values: prevalence %).
        colors (list): Optional list of colors for age groups.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = df.columns.tolist()        # years (x-axis)
    age_groups = df.index.tolist()      # age bins (bar series)
    x = np.arange(len(labels))          # x-axis positions for years
    n_groups = len(age_groups)
    width = 0.8 / n_groups              # bar width, adjusted to number of groups

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_groups))  # fallback color palette

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, age_group in enumerate(age_groups):
        ax.bar(x + i * width, df.loc[age_group], width, label=age_group, color=colors[i])

    ax.set_xlabel('Year', fontsize=22, fontweight='bold')
    ax.set_ylabel('T2D Prevalence among PLHIV (%)', fontsize=20, fontweight='bold')
    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(labels, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)

    ax.legend(title='Age Group', title_fontsize=16, fontsize=14, frameon=False, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # ax.set_title('T2D Prevalence Among PLHIV by Age Group (Combined Sex)', fontsize=22, fontweight='bold')

    plt.tight_layout()
    plt.show() 


def plot_t2d_prevalence_by_age_and_year(data, colors=None):
    """
    Plot grouped bar chart for T2D prevalence among PLHIV by age group over selected years.

    Args:
        data (dict or pd.DataFrame): Keys (or index) are age groups; columns are years; values are prevalence (%).
        colors (list): Optional list of colors for the bars.
    """
    # Convert dict to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame(data).T  # transpose so rows = age groups, columns = years
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Input must be a DataFrame or a dict")

    labels = df.columns.astype(str).tolist()  # years on x-axis
    age_groups = df.index.tolist()            # age bins (bar series)
    x = np.arange(len(labels))                # x-axis positions for years
    n_groups = len(age_groups)
    width = 0.8 / n_groups                    # bar width

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_groups))  # default palette

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, age_group in enumerate(age_groups):
        ax.bar(x + i * width, df.loc[age_group], width, label=age_group, color=colors[i])

    ax.set_xlabel('Year', fontsize=22, fontweight='bold')
    ax.set_ylabel('T2D Prevalence among PLHIV (%)', fontsize=20, fontweight='bold')
    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(labels, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)

    ax.legend(title='Age Group', title_fontsize=16, fontsize=14, frameon=False, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

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

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(sim.timevec, mean_prevalence_male_with_HIV, label=f'Male {disease.capitalize()} Prevalence (HIV+)', linewidth=3, color='blue', linestyle='solid')
    ax.plot(sim.timevec, mean_prevalence_female_with_HIV, label=f'Female {disease.capitalize()} Prevalence (HIV+)', linewidth=3, color='red', linestyle='solid')
    ax.plot(sim.timevec, mean_prevalence_male_without_HIV, label=f'Male {disease.capitalize()} Prevalence (HIV-)', linewidth=3, color='blue', linestyle='dashed')
    ax.plot(sim.timevec, mean_prevalence_female_without_HIV, label=f'Female {disease.capitalize()} Prevalence (HIV-)', linewidth=3, color='red', linestyle='dashed')

    ax.set_xlabel('Year', fontsize=22, fontweight='bold')
    ax.set_ylabel('T2D Prevalence (%)', fontsize=20, fontweight='bold')
    # ax.set_title(f'Mean {disease.capitalize()} Prevalence Over Time (All Ages)', fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.legend(loc='lower right', fontsize=14, frameon=False)   
    ax.grid()

    plt.show()

# # ##### Run the following 4 times after changing the end_year #####
# # # Get the T2D disease module
# t2d = sim.diseases['type2diabetes']

# # Define custom age groups
# age_groups = [
#     (0, 18, '0–17'),
#     (18, 35, '18–34'),
#     (35, 50, '35–49'),
#     (50, 65, '50–64'),
#     (65, 150, '65+'),
# ]

# print("Age Group | Total People | T2D Cases | % with T2D")
# print("---------------------------------------------------")

# for a0, a1, label in age_groups:
#     age_mask = (sim.people.age >= a0) & (sim.people.age < a1)
#     total = age_mask.sum()
#     affected = (age_mask & t2d.affected).sum()
#     prevalence = 100 * affected / total if total > 0 else 0
#     print(f"{label:>8} |     {total:5}     |    {affected:5}   |   {prevalence:6.2f}%")
    
    
# # # Get the T2D disease module
# # t2d = sim.diseases['type2diabetes']

# # print("Age | Total People | T2D Cases |  % with T2D")
# # print("---------------------------------------------")

# # for age in range(6):  # Ages 0 through 5
# #     is_age = (sim.people.age >= age) & (sim.people.age < age + 1)
# #     total = is_age.sum()
# #     affected = (is_age & t2d.affected).sum()
# #     prevalence = 100 * affected / total if total > 0 else 0
# #     print(f" {age}  |     {total:5}     |    {affected:5}   |   {prevalence:6.2f}%")  


# data = {
#     "0-17":  [0.06, 0.10, 0.09, 0.08],
#     "18-34": [1.5, 1.99, 2.16, 2.01],
#     "35-49": [3.77, 5.37, 5.97, 6.12],
#     "50-64": [12.06, 14.21, 14.2, 15.13],
#     "65+":   [24.22, 29.35, 32.66, 35.58],
# }
# columns = [2008, 2021, 2035, 2050]
# df = pd.DataFrame(data, index=columns).T  # transpose to have age groups as index

# colors = ['#08306b', '#2171b3', '#6baed6', '#9ecae1', '#c6dbef']  # One color per age group (5 total)

# plot_t2d_prevalence_by_age_and_year(df,colors)  
# plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'Type2Diabetes')
