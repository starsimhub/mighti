import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def plot_mortality_rates_comparison(simulated_rates, observed_rates, observed_year=None, year=None, 
                              log_scale=True, figsize=(14, 10), title=None):
    """
    Plot comparison between simulated and observed mortality rates
    
    Args:
        simulated_rates: DataFrame with simulated mortality rates 
                        (columns: Time, Sex, AgeGrpStart, mx)
        observed_rates: DataFrame with observed mortality rates
                       (columns: Time, Sex, AgeGrpStart, mx)
        observed_year: Year to filter observed data (if None, will use simulated year)
        year: Year to filter simulated data (if None, will use Time column from data)
        log_scale: Whether to use log scale for mortality rates
        figsize: Figure size (width, height) in inches
        title: Custom title for the plot
        
    Returns:
        Figure and axes objects
    """
    # Filter simulated data if year provided
    if year is not None:
        simulated_rates = simulated_rates[simulated_rates['Time'] == year].copy()
    else:
        # Try to get year from the data
        years = simulated_rates['Time'].unique()
        if len(years) == 1:
            year = years[0]
    
    # Filter observed data by year
    if observed_year is None and year is not None:
        observed_year = year
        
    if observed_year is not None:
        observed_rates = observed_rates[observed_rates['Time'] == observed_year].copy()
    
    # Create figure with two panels for male and female
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Get unique age groups from both datasets
    sim_ages = sorted(simulated_rates['AgeGrpStart'].unique())
    obs_ages = sorted(observed_rates['AgeGrpStart'].unique())
    
    # Plot male data - FIXED THE BUG - was using Female data for Male
    male_sim = simulated_rates[simulated_rates['Sex'] == 'Male']
    male_obs = observed_rates[observed_rates['Sex'] == 'Male']  # FIXED: Was incorrectly using Female
    
    # Extract x and y values
    sim_male_x = male_sim['AgeGrpStart'].values
    sim_male_y = male_sim['mx'].values
    obs_male_x = male_obs['AgeGrpStart'].values
    obs_male_y = male_obs['mx'].values
    
    # Plot male mortality rates
    ax1.plot(sim_male_x, sim_male_y, 'b-', marker='o', markersize=8, 
             linewidth=2, label=f'Simulated Male ({year})')
    ax1.plot(obs_male_x, obs_male_y, 'b--', marker='s', markersize=8,
             linewidth=2, alpha=0.7, label=f'Observed Male ({observed_year})')
    
    # Plot female data
    female_sim = simulated_rates[simulated_rates['Sex'] == 'Female']
    female_obs = observed_rates[observed_rates['Sex'] == 'Female']
    
    # Extract x and y values
    sim_female_x = female_sim['AgeGrpStart'].values
    sim_female_y = female_sim['mx'].values
    obs_female_x = female_obs['AgeGrpStart'].values
    obs_female_y = female_obs['mx'].values
    
    # Plot female mortality rates
    ax2.plot(sim_female_x, sim_female_y, 'r-', marker='o', markersize=8,
             linewidth=2, label=f'Simulated Female ({year})')
    ax2.plot(obs_female_x, obs_female_y, 'r--', marker='s', markersize=8, 
             linewidth=2, alpha=0.7, label=f'Observed Female ({observed_year})')
    
    # Set axis scales
    if log_scale:
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    
    # Set titles and labels for male panel
    ax1.set_title(f'Male Mortality Rates Comparison', fontsize=16)
    ax1.set_ylabel('Mortality Rate (mx)', fontsize=14)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Set titles and labels for female panel
    ax2.set_title(f'Female Mortality Rates Comparison', fontsize=16)
    ax2.set_xlabel('Age Group Start', fontsize=14)
    ax2.set_ylabel('Mortality Rate (mx)', fontsize=14)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=12)
    
    # Set overall title if provided
    if title:
        plt.suptitle(title, fontsize=18, y=0.98)
    else:
        plt.suptitle(f'Mortality Rates Comparison: Simulated vs Observed', fontsize=18, y=0.98)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    # Print summary comparison
    print("\nMortality rates comparison summary:")
    
    # Compare average rates for broad age categories 
    categories = [(0, 5, "Infant/Child"), (5, 15, "Youth"), (15, 45, "Young/Middle Adult"), 
                 (45, 65, "Older Adult"), (65, 101, "Elderly")]
    
    for sex in ['Male', 'Female']:
        print(f"\n{sex} mortality rates:")
        
        sim_sex_data = simulated_rates[simulated_rates['Sex'] == sex]
        obs_sex_data = observed_rates[observed_rates['Sex'] == sex]
        
        for start, end, label in categories:
            # Calculate average simulated rate
            sim_mask = (sim_sex_data['AgeGrpStart'] >= start) & (sim_sex_data['AgeGrpStart'] < end)
            sim_avg = sim_sex_data.loc[sim_mask, 'mx'].mean() if any(sim_mask) else 0
            
            # Calculate average observed rate
            obs_mask = (obs_sex_data['AgeGrpStart'] >= start) & (obs_sex_data['AgeGrpStart'] < end)
            obs_avg = obs_sex_data.loc[obs_mask, 'mx'].mean() if any(obs_mask) else 0
            
            # Calculate ratio between simulated and observed
            ratio = sim_avg / obs_avg if obs_avg > 0 else float('nan')
            
            print(f"  {label} ({start}-{end-1}): Sim mx={sim_avg:.6f}, Obs mx={obs_avg:.6f}, Ratio={ratio:.2f}")
    
    return fig, (ax1, ax2)