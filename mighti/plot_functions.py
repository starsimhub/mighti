"""
Provides utility functions for generating plots from simulation outputs
"""


import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import Counter


def _safe_get_result(analyzer, key, sim):
    """
    Return analyzer result as a numpy array aligned to sim.timevec length.
    Works for both Starsim ≤2.x and ≥3.x result structures.
    """
    val = analyzer.results.get(key)
    if val is None:
        return np.zeros(len(sim.timevec))
    if hasattr(val, "values"):  # e.g. pandas Series
        val = val.values
    val = np.array(val)
    if len(val) != len(sim.timevec):
        val = np.pad(val, (0, max(0, len(sim.timevec) - len(val))), mode="edge")[: len(sim.timevec)]
    return val

logger = logging.getLogger(__name__)

def plot_mean_prevalence_plhiv(sim, prevalence_analyzer, disease):
    """
    Plot mean prevalence over time for a given disease and both sexes.
    """
    disease = disease.lower() 

    def extract_results(key_pattern):
        matching_keys = [k for k in prevalence_analyzer.results.keys()
                         if k.startswith(f'{disease}_{key_pattern}_')]
        matching_keys = sorted(matching_keys, key=lambda x: int(x.split('_')[-1]))
        if not matching_keys:
            print(f'No keys found for pattern {disease}_{key_pattern}_')
        return [_safe_get_result(prevalence_analyzer, k, sim) for k in matching_keys]

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

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sim.timevec, mean_prevalence_male_with_HIV, label=f'Male {disease} (HIV+)', color='blue', lw=2)
    ax.plot(sim.timevec, mean_prevalence_female_with_HIV, label=f'Female {disease} (HIV+)', color='red', lw=2)
    ax.plot(sim.timevec, mean_prevalence_male_without_HIV, '--', color='blue', lw=2, label=f'Male {disease} (HIV–)')
    ax.plot(sim.timevec, mean_prevalence_female_without_HIV, '--', color='red', lw=2, label=f'Female {disease} (HIV–)')
    ax.set_xlabel('Year', fontsize=16)
    ax.set_ylabel(f'{disease} Prevalence (%)', fontsize=16)
    ax.legend()
    ax.grid(True)
    plt.show()
    
    
def plot_mean_prevalence(sim, prevalence_analyzer, disease, prevalence_data_df, init_year, end_year):
    """
    Plot mean prevalence over time for a given disease and both sexes, including observed data points.
    Ensures x-axis uses numeric years, not dates.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import units, dates

    disease = disease.lower()

    # --- Safe getter ---
    def _safe_get_result(analyzer, key, sim):
        val = analyzer.results.get(key)
        if val is None:
            return np.zeros(len(sim.timevec))
        if hasattr(val, "values"):
            val = val.values
        val = np.array(val, dtype=float)
        if len(val) != len(sim.timevec):
            val = np.pad(val, (0, max(0, len(sim.timevec) - len(val))), mode="edge")[:len(sim.timevec)]
        return val

    # --- Extract results ---
    def extract_results(key_pattern):
        matching_keys = [k for k in prevalence_analyzer.results.keys()
                         if k.startswith(f"{disease}_{key_pattern}_")]
        matching_keys = sorted(matching_keys, key=lambda x: int(x.split('_')[-1]))
        if not matching_keys:
            print(f"⚠️ No keys found for pattern {disease}_{key_pattern}_")
        return [_safe_get_result(prevalence_analyzer, k, sim) for k in matching_keys]

    male_num = np.sum(extract_results("num_male"), axis=0)
    female_num = np.sum(extract_results("num_female"), axis=0)
    male_den = np.sum(extract_results("den_male"), axis=0)
    female_den = np.sum(extract_results("den_female"), axis=0)

    male_den[male_den == 0] = 1
    female_den[female_den == 0] = 1

    total_male_prev = np.nan_to_num(male_num / male_den) * 100
    total_female_prev = np.nan_to_num(female_num / female_den) * 100

    mask = (sim.timevec >= init_year) & (sim.timevec <= end_year)
    time_years = np.array(sim.timevec[mask], dtype=float)

    # --- Plot simulated curves ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_years, total_male_prev[mask], label="Male Simulated", color="blue", lw=3)
    ax.plot(time_years, total_female_prev[mask], label="Female Simulated", color="red", lw=3)

    # --- Clean observed data ---
    df = prevalence_data_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "year" not in df.columns:
        raise ValueError("prevalence_data_df must include a 'year' column")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    most_recent = df["year"].max()
    df = df[(df["year"] >= init_year) & (df["year"] <= min(end_year, most_recent))]

    disease_col = disease.lower()

    if disease_col in df.columns and "sex" in df.columns:
        for sex, color in [("male", "blue"), ("female", "red")]:
            obs = df[df["sex"].str.lower() == sex]
            if not obs.empty:
                ax.scatter(
                    obs["year"].astype(float), obs[disease_col] * 100,
                    label=f"{sex.capitalize()} Observed",
                    color=color, edgecolor="black", s=80, zorder=5
                )

    # --- Ensure numeric (not datetime) x-axis ---
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    units.registry[np.ndarray] = None   # 🚫 disable automatic datetime conversion
    dates.set_epoch('0000-12-31T00:00:00')  # safety for some Matplotlib versions

    # --- Style ---
    ax.set_title(f"{disease.capitalize()} Prevalence Over Time (All Ages)", fontsize=16)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Prevalence (%)", fontsize=14)
    ax.set_xlim(init_year - 1, end_year + 1)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=12)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    print(f" Observed data restricted to {init_year}–{min(end_year, most_recent)} ({len(df)} rows used).")
    return total_male_prev, total_female_prev   
    
    
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
                marker='s', linestyle='--', linewidth=2, markersize=5,
                color='black', label='Observed')
        ax.set_title(f"{sex} Mortality Rate (mx) Comparison, {year}", fontsize=24)
        ax.set_ylabel('m(x)', fontsize=24)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel('Age Group', fontsize=24)
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

    ax1.plot(male_sim['Age'], male_sim['e(x)'], '-', linewidth=12, alpha=0.4, color='blue', label='Simulated')
    ax1.plot(male_obs['Age'], male_obs['ex'], 's--', linewidth=2, markersize=5, color='black', label='Observed')
    ax1.set_title('Male', fontsize=28)
    ax1.set_ylabel('Life Expectancy (years)', fontsize=24)
    ax1.tick_params(labelsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=20)

    ax2.plot(female_sim['Age'], female_sim['e(x)'], '-', linewidth=12, alpha=0.4, color='red', label='Simulated')
    ax2.plot(female_obs['Age'], female_obs['ex'], 's--', linewidth=2, markersize=5, color='black', label='Observed')
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
