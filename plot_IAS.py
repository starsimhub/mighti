##### Place these in mighti_main to run this chunk #####

    # df = death_cause_analyzer.to_df()   
    # df['HIV only'] = df['had_hiv'] & ~df['had_type2diabetes']
    # df['T2D only'] = df['had_type2diabetes'] & ~df['had_hiv']
    # df['Both'] = df['had_hiv'] & df['had_type2diabetes']
    # df['Neither'] = ~df['had_hiv'] & ~df['had_type2diabetes']
    # counts = df[['HIV only', 'T2D only', 'Both', 'Neither']].sum()
    # print(counts)
    # df.groupby('sex')[['HIV only', 'T2D only', 'Both', 'Neither']].sum()
   
   # import plot_IAS
   # target_years = [2007, 2024, 2030, 2050]
   # age_groups = [
   #     (0, 5, 'Under 5'),
   #     (5, 15, '5 to 14'),
   #     (15, 50, '15 to 49'),
   #     (50, 70, '50 to 70'),
   #     (70, 100, 'Over 70'),
   # ]
   # df_male, df_female = plot_IAS.extract_t2d_prevalence_by_age_plhiv(sim, prevalence_analyzer, 'Type2Diabetes', target_years, age_groups)


   # colors = ['#08306b', '#2171b3', '#6baed6', '#9ecae1', '#c6dbef']  # One color per age group (5 total)

   # plot_IAS.plot_grouped_prevalence_bar_combined(df_female, colors)
   # plot_IAS.plot_mean_prevalence_plhiv(sim, prevalence_analyzer, 'Type2Diabetes')


##############################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def extract_combined_t2d_prevalence_by_age_plhiv(sim, prevalence_analyzer, disease, target_years, age_groups):
    """
    Extract combined (male + female) T2D prevalence among PLHIV by age group and year.
    """
    year_to_index = {int(round(year)): idx for idx, year in enumerate(sim.timevec)}
    selected_indices = {int(year): year_to_index[int(year)] for year in target_years if int(year) in year_to_index}

    def extract_results(key_pattern):
        return [
            prevalence_analyzer.results.get(f'{disease}_{key_pattern}_{i}', np.zeros(len(sim.timevec)))
            for i in range(len(age_groups))
        ]

    # Extract male and female numerators and denominators
    male_num = extract_results('num_with_HIV_male')
    female_num = extract_results('num_with_HIV_female')
    male_den = extract_results('den_with_HIV_male')
    female_den = extract_results('den_with_HIV_female')

    combined_data = []

    for year in target_years:
        ti = selected_indices.get(int(year))
        if ti is None:
            continue

        row = []
        for i in range(len(age_groups)):
            m_num = male_num[i][ti]
            m_den = male_den[i][ti]
            f_num = female_num[i][ti]
            f_den = female_den[i][ti]

            total_num = m_num + f_num
            total_den = m_den + f_den
            prevalence = (total_num / total_den * 100) if total_den > 0 else 0
            row.append(prevalence)

        combined_data.append(row)

    age_labels = [label for _, _, label in age_groups]
    df_combined = pd.DataFrame(combined_data, index=target_years, columns=age_labels).T
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


