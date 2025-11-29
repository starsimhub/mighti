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
            print(f" No keys found for pattern {disease}_{key_pattern}_")
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
    ax.plot(time_years, total_male_prev[mask], label="Male Simulated", color="blue", lw=3, marker='o', markersize=3, alpha=0.7)
    ax.plot(time_years, total_female_prev[mask], label="Female Simulated", color="red", lw=3, marker='o', markersize=3, alpha=0.7)

    # --- Clean observed data ---
    df = prevalence_data_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "year" not in df.columns:
        raise ValueError(f"prevalence_data_df must include a 'year' column. Available columns: {df.columns.tolist()}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    most_recent = df["year"].max()
    df = df[(df["year"] >= init_year) & (df["year"] <= min(end_year, most_recent))]

    # Try to find disease column with multiple possible names
    disease_col = None
    disease_col_male = None
    disease_col_female = None
    possible_names = [
        disease.lower(),
        disease.lower().replace("type", "type_").replace("diabetes", "diabetes"),
        disease.lower().replace("2", "_2").replace("1", "_1"),
        disease.lower().replace("type2", "type_2").replace("type1", "type_1"),
    ]
    
    # First, try to find the base disease column
    for name in possible_names:
        if name in df.columns:
            disease_col = name
            break
    
    # If base column not found, look for sex-specific columns
    if disease_col is None:
        for name in possible_names:
            male_name = f"{name}_male"
            female_name = f"{name}_female"
            if male_name in df.columns:
                disease_col_male = male_name
            if female_name in df.columns:
                disease_col_female = female_name
            if disease_col_male or disease_col_female:
                break
        
        # If still not found, try to find any column that contains the disease name
        if disease_col_male is None and disease_col_female is None:
            for col in df.columns:
                if disease.lower() in col.lower() or col.lower() in disease.lower():
                    # Check if it's sex-specific
                    if "_male" in col.lower():
                        disease_col_male = col
                    elif "_female" in col.lower():
                        disease_col_female = col
                    else:
                        disease_col = col
                        print(f"  Warning: Using column '{col}' for disease '{disease}'")
                    break
    
    # Find sex column (try multiple possible names)
    sex_col = None
    for possible_sex_col in ["sex", "gender", "male", "female"]:
        if possible_sex_col in df.columns:
            sex_col = possible_sex_col
            break
    
    # Plot observed data
    # Handle sex-specific columns first (if found)
    if disease_col_male or disease_col_female:
        # We have sex-specific columns, plot them separately
        for col, sex, color in [(disease_col_male, "male", "blue"), (disease_col_female, "female", "red")]:
            if col is not None and col in df.columns:
                # Check if data is already in percentage format
                sample_values = df[col].dropna()
                if len(sample_values) > 0:
                    max_val = sample_values.max()
                    is_percentage = max_val > 1.0
                    
                    # Extract values and years - ensure we're using the filtered dataframe
                    obs_values = df[col].astype(float)
                    
                    # Make sure year column exists and is numeric
                    if "year" not in df.columns:
                        print(f"  Error: 'year' column not found in observed data. Available columns: {df.columns.tolist()}")
                        continue
                    
                    obs_years = pd.to_numeric(df["year"], errors="coerce")
                    
                    # Remove rows where disease value is NaN OR year is NaN, keeping years aligned
                    valid_mask = ~obs_values.isna() & ~obs_years.isna()
                    obs_df_valid = df[valid_mask].copy()
                    
                    if len(obs_df_valid) > 0:
                        # Aggregate by year (mean) if there are multiple rows per year
                        # This handles cases where data has multiple age groups or other dimensions per year
                        obs_df_valid["year"] = obs_years[valid_mask]
                        obs_df_valid["value"] = obs_values[valid_mask]
                        
                        # Group by year and take mean (or could use median, sum, etc.)
                        aggregated = obs_df_valid.groupby("year")["value"].mean().reset_index()
                        obs_years_agg = aggregated["year"].values
                        obs_values_agg = aggregated["value"].values
                        
                        # Debug: print year range and sample data
                        if len(obs_years_agg) > 0:
                            print(f"  {sex.capitalize()} observed: {len(obs_df_valid)} raw points -> {len(obs_years_agg)} aggregated points")
                            print(f"    Years range: {obs_years_agg.min():.0f} to {obs_years_agg.max():.0f}, unique years: {len(np.unique(obs_years_agg))}")
                            print(f"    Sample years: {obs_years_agg[:10] if len(obs_years_agg) > 10 else obs_years_agg}")
                            print(f"    Sample values: {obs_values_agg[:5] if len(obs_values_agg) > 5 else obs_values_agg}")
                        
                        if len(obs_values_agg) > 0:
                            # Convert to percentage if needed
                            if not is_percentage:
                                obs_values_agg = obs_values_agg * 100
                            
                            ax.scatter(
                                obs_years_agg, obs_values_agg,
                                label=f"{sex.capitalize()} Observed",
                                color=color, edgecolor="black", s=100, zorder=5, alpha=0.8, marker='s'
                            )
    elif disease_col is not None and disease_col in df.columns:
        # Standard disease column - check if data is already in percentage format
        sample_values = df[disease_col].dropna()
        if len(sample_values) > 0:
            max_val = sample_values.max()
            is_percentage = max_val > 1.0
            
            # Handle standard format with sex column
            if sex_col and sex_col in df.columns:
                # Standard sex column - filter by sex
                for sex, color in [("male", "blue"), ("female", "red")]:
                    if sex_col == "sex" or sex_col == "gender":
                        obs = df[df[sex_col].str.lower().str.strip() == sex].copy()
                    else:
                        # If sex_col is "male" or "female", filter differently
                        obs = df[df[sex_col] == 1].copy() if sex_col == sex else df[df[sex_col] == 0].copy()
                    
                    if not obs.empty and disease_col in obs.columns:
                        # Extract values and years, keeping them aligned
                        obs_values = obs[disease_col].astype(float)
                        obs_years = pd.to_numeric(obs["year"], errors="coerce")
                        
                        # Remove rows where disease value is NaN or year is NaN
                        valid_mask = ~obs_values.isna() & ~obs_years.isna()
                        obs_valid = obs[valid_mask].copy()
                        
                        if len(obs_valid) > 0:
                            # Aggregate by year (mean) if there are multiple rows per year
                            obs_valid["year"] = obs_years[valid_mask].values
                            obs_valid["value"] = obs_values[valid_mask].values
                            
                            aggregated = obs_valid.groupby("year")["value"].mean().reset_index()
                            obs_years_agg = aggregated["year"].values
                            obs_values_agg = aggregated["value"].values
                            
                            if len(obs_values_agg) > 0:
                                # Convert to percentage if needed
                                if not is_percentage:
                                    obs_values_agg = obs_values_agg * 100
                                
                                ax.scatter(
                                    obs_years_agg, obs_values_agg,
                                    label=f"{sex.capitalize()} Observed",
                                    color=color, edgecolor="black", s=100, zorder=5, alpha=0.8, marker='s'
                                )
            else:
                # No sex column - plot all data
                obs_values = df[disease_col].astype(float)
                obs_years = pd.to_numeric(df["year"], errors="coerce")
                
                # Remove rows where disease value is NaN or year is NaN
                valid_mask = ~obs_values.isna() & ~obs_years.isna()
                obs_valid = df[valid_mask].copy()
                
                if len(obs_valid) > 0:
                    # Aggregate by year (mean) if there are multiple rows per year
                    obs_valid["year"] = obs_years[valid_mask].values
                    obs_valid["value"] = obs_values[valid_mask].values
                    
                    aggregated = obs_valid.groupby("year")["value"].mean().reset_index()
                    obs_years_agg = aggregated["year"].values
                    obs_values_agg = aggregated["value"].values
                    
                    if len(obs_values_agg) > 0:
                        if not is_percentage:
                            obs_values_agg = obs_values_agg * 100
                        
                        ax.scatter(
                            obs_years_agg, obs_values_agg,
                            label="Observed",
                            color="black", edgecolor="white", s=100, zorder=5, alpha=0.8, marker='s'
                        )
    else:
        print(f"  Warning: Could not find disease column for '{disease}' in observed data.")
        print(f"  Available columns: {df.columns.tolist()}")
        print(f"  Tried names: {possible_names}")

    # --- Ensure numeric (not datetime) x-axis ---
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    units.registry[np.ndarray] = None   # disable automatic datetime conversion

    # --- Style ---
    ax.set_title(f"{disease.capitalize()} Prevalence Over Time (All Ages)", fontsize=16)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Prevalence (%)", fontsize=14)
    ax.set_xlim(init_year - 1, end_year + 1)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, fontsize=11, loc='best')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    print(f"  Observed data restricted to {init_year}–{min(end_year, most_recent)} ({len(df)} rows used).")
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


def plot_cost_effectiveness_plane(results, wtp_thresholds=[100, 500, 1000],
                                  figsize=(8, 6),savepath=None,show=True):
    
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each result as a scatter point
    for res in results:
        ax.scatter(res['delta_daly'], res['delta_cost'], label=res['label'], s=80)
        ax.annotate(res['label'], (res['delta_daly'], res['delta_cost']), fontsize=10,
                    xytext=(5, 5), textcoords='offset points')

    # Plot WTP threshold lines
    x_vals = np.linspace(0, max([r['delta_daly'] for r in results]) * 1.1, 100)
    for wtp in wtp_thresholds:
        ax.plot(x_vals, wtp * x_vals, linestyle='--', label=f'${wtp}/DALY')

    # Axis labels and formatting
    ax.set_xlabel('Incremental DALYs averted', fontsize=12)
    ax.set_ylabel('Incremental cost ($)', fontsize=12)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend()
    ax.set_title('Cost-Effectiveness Plane', fontsize=14)
    ax.grid(True)

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_adherence_by_condition(sim, label):
    results = sim.results[label]
    t = np.array(results["time"])
    with_cond = np.array(results["on_with_condition"])
    without_cond = np.array(results["on_without_condition"])

    plt.plot(t, with_cond, label="With condition")
    plt.plot(t, without_cond, label="Without condition")
    plt.xlabel("Year")
    plt.ylabel("Proportion on ART")
    plt.title(f"Adherence by {label}")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_severity_distribution(sim, disease_name, show_weights=True, figsize=(12, 5)):
    """
    Plot severity distribution for a disease showing:
    1. Current proportion of affected individuals in each severity level
    2. Severity weights (disability weights) for each level
    
    Parameters:
        sim: Simulation object
        disease_name (str): Name of the disease (e.g., "Type2Diabetes")
        show_weights (bool): Whether to show disability weights plot
        figsize: Figure size tuple
    """
    disease_key = disease_name.lower()
    disease = sim.diseases.get(disease_key, None)
    
    if disease is None:
        logger.warning(f"Disease '{disease_name}' not found in simulation. Available diseases: {list(sim.diseases.keys())}")
        return
    
    # Check if disease has severity system
    if not hasattr(disease, 'severity_level'):
        logger.warning(f"Disease '{disease_name}' does not have severity_level state. Severity system may not be initialized.")
        return
    
    # Get affected individuals
    if hasattr(disease, 'affected'):
        affected_uids = disease.affected.uids
        affected_mask = disease.affected
    elif hasattr(disease, 'infected'):
        affected_uids = disease.infected.uids
        affected_mask = disease.infected
    else:
        logger.warning(f"Disease '{disease_name}' does not have 'affected' or 'infected' state.")
        return
    
    if len(affected_uids) == 0:
        logger.warning(f"No affected individuals found for '{disease_name}' at end of simulation.")
        return
    
    # Get severity levels for affected individuals
    severity_levels = disease.severity_level[affected_uids]
    
    # Count by severity level
    n_levels = disease.n_severity_levels if hasattr(disease, 'n_severity_levels') else 3
    severity_counts = np.bincount(severity_levels, minlength=n_levels)
    severity_proportions = severity_counts / len(affected_uids) if len(affected_uids) > 0 else np.zeros(n_levels)
    
    # Get severity weights if available
    severity_weights = None
    if hasattr(disease, 'severity_weights'):
        severity_weights = disease.severity_weights
    
    # Create figure with subplots
    if show_weights and severity_weights is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        ax2 = None
    
    # Plot 1: Severity distribution (bar chart)
    severity_labels = [f"Level {i}" for i in range(n_levels)]
    if n_levels == 3:
        severity_labels = ["Mild", "Moderate", "Severe"]
    elif n_levels == 2:
        severity_labels = ["Mild", "Severe"]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, n_levels))  # Green (mild) to Red (severe)
    
    bars = ax1.bar(severity_labels, severity_proportions * 100, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Proportion of Affected Individuals (%)', fontsize=12)
    ax1.set_xlabel('Severity Level', fontsize=12)
    ax1.set_title(f'{disease_name} Severity Distribution\n(n={len(affected_uids):,} affected)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(severity_proportions) * 100 * 1.2 if max(severity_proportions) > 0 else 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, prop) in enumerate(zip(bars, severity_proportions)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{prop*100:.1f}%\n(n={severity_counts[i]:,})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Severity weights (disability weights)
    if ax2 is not None and severity_weights is not None:
        bars2 = ax2.bar(severity_labels, severity_weights, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Disability Weight', fontsize=12)
        ax2.set_xlabel('Severity Level', fontsize=12)
        ax2.set_title(f'{disease_name} Disability Weights', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, max(severity_weights) * 1.2 if max(severity_weights) > 0 else 1.0)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, weight in zip(bars2, severity_weights):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{weight:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{disease_name} Severity Summary:")
    print(f"  Total affected: {len(affected_uids):,}")
    for i, (label, count, prop) in enumerate(zip(severity_labels, severity_counts, severity_proportions)):
        weight_str = f" (DW={severity_weights[i]:.3f})" if severity_weights is not None else ""
        print(f"  {label}: {count:,} ({prop*100:.1f}%){weight_str}")
    
    if severity_weights is not None:
        # Calculate weighted average disability weight
        weighted_avg = np.sum(severity_proportions * severity_weights)
        print(f"  Weighted average disability weight: {weighted_avg:.3f}")
    
    return fig
    