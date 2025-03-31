import starsim as ss
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import mighti as mi



class DeathTracker(ss.Analyzer):
    def __init__(self):
        super().__init__()
        self.name = 'death_tracker'
        self.death_data = {'Male': {}, 'Female': {}}
        self.yearly_death_data = defaultdict(lambda: {'Male': {}, 'Female': {}})
        self.total_deaths_tracked = 0
        self.previous_alive = None
        self.previous_uids = None

    def step(self):
        """Called automatically on each timestep to track deaths"""
        sim = self.sim
        current_year = sim.t.now()  # Get current year at the beginning
        
        # Get current state
        current_alive = sim.people.alive
        current_uids = sim.people.uid
        
        # Check if this is the first timestep
        if self.ti == 0 or self.previous_alive is None:
            # Store the initial state of alive
            self.previous_alive = current_alive.copy()
            self.previous_uids = current_uids.copy()
            return
        
        # Find people who were alive in the previous timestep but are now dead
        # We need to match UIDs to handle population growth
        newly_dead_indices = []
        
        # Create mapping from UID to index for current timestep 
        current_uid_to_index = {uid: idx for idx, uid in enumerate(current_uids)}
        
        # Loop through previous timestep's UIDs
        for prev_idx, prev_uid in enumerate(self.previous_uids):
            # Check if this UID still exists in the current timestep
            if prev_uid in current_uid_to_index:
                current_idx = current_uid_to_index[prev_uid]
                # Check if this person was alive before but is dead now
                if self.previous_alive[prev_idx] and not current_alive[current_idx]:
                    newly_dead_indices.append(current_idx)
        
        # Count for debugging
        self.total_deaths_tracked += len(newly_dead_indices)
        
        # Log each newly dead person
        for idx in newly_dead_indices:
            age = int(sim.people.age[idx])
            sex = 'Male' if not sim.people.female[idx] else 'Female'
            
            # Update cumulative death data
            if age not in self.death_data[sex]:
                self.death_data[sex][age] = 0
            self.death_data[sex][age] += 1
            
            # Update yearly death data - MOVED INSIDE THE LOOP
            if age not in self.yearly_death_data[current_year][sex]:
                self.yearly_death_data[current_year][sex][age] = 0
            self.yearly_death_data[current_year][sex][age] += 1
        
        # Print debug information
        if len(newly_dead_indices) > 0:
            print(f"Time {current_year}: Tracked {len(newly_dead_indices)} new deaths, total tracked: {self.total_deaths_tracked}")
        
        # Update previous state for next timestep comparison
        self.previous_alive = current_alive.copy()
        self.previous_uids = current_uids.copy()

    def get_death_counts(self, year=None):
        """
        Return death counts by age and sex
        
        Args:
            year: If provided, return deaths for that specific year
                  If None, return cumulative deaths across all years
        """
        if year is None:
            # Return cumulative deaths
            death_counts = {'Male': {}, 'Female': {}}
            for age in range(101):  # 0 to 100
                death_counts['Male'][age] = self.death_data['Male'].get(age, 0)
                death_counts['Female'][age] = self.death_data['Female'].get(age, 0)
            
            print(f"Getting cumulative death counts - Total deaths tracked: {self.total_deaths_tracked}")
        
        else:
            # Return deaths for specific year
            if year not in self.yearly_death_data:
                print(f"No death data available for year {year}")
                return {'Male': {age: 0 for age in range(101)}, 'Female': {age: 0 for age in range(101)}}
            
            death_counts = {'Male': {}, 'Female': {}}
            for age in range(101):  # 0 to 100
                death_counts['Male'][age] = self.yearly_death_data[year]['Male'].get(age, 0)
                death_counts['Female'][age] = self.yearly_death_data[year]['Female'].get(age, 0)
            
            year_total = sum(self.yearly_death_data[year]['Male'].values()) + sum(self.yearly_death_data[year]['Female'].values())
            print(f"Getting death counts for year {year} - Deaths in this year: {year_total}")
        
        return death_counts    
    
    
    def plot_cum_death_counts(self, smoothing=3, age_interval=1, figsize=(12, 8)):
        """
        Plot the absolute number of deaths by age and sex
        
        Args:
            smoothing: Sigma value for Gaussian smoothing (0 for no smoothing)
            age_interval: Group ages by this interval (e.g., 5 for 5-year age groups)
            figsize: Figure size (width, height) in inches
        """
        death_counts = self.get_death_counts()
        
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
        
        # Plot male and female lines
        plt.plot(ages, male_counts, 'b-', linewidth=2, alpha=0.7, label='Male')
        plt.plot(ages, female_counts, 'r-', linewidth=2, alpha=0.7, label='Female')
        
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
        
        # Customize the plot
        plt.grid(True, alpha=0.3)
        plt.xlabel('Age', fontsize=14)
        plt.ylabel('Number of Deaths', fontsize=14)
        plt.title(f'Death Counts by Age and Sex (Total: {self.total_deaths_tracked})', fontsize=16)
        plt.legend(fontsize=12)
        
        # Set axis limits
        plt.xlim(0, max(ages))
        plt.ylim(bottom=0)
        
        # Improve layout
        plt.tight_layout()
        
        plt.show()
        return  # Return the axis for further modifications if needed
    
    def plot_cum_aggregated_death_counts(self, age_groups=None):
        """
        Plot death counts aggregated into standard age groups
        
        Args:
            age_groups: List of tuples (start_age, end_age, label)
        """
        if age_groups is None:
            # Default age groups
 
            # Age groups aligned with the provided age bins
            age_groups = [
                (0, 15, "0-14"),
                (15, 21, "15-20"),
                (21, 26, "21-25"),
                (26, 31, "26-30"),
                (31, 36, "31-35"),
                (36, 41, "36-40"),
                (41, 46, "41-45"),
                (46, 51, "46-50"),
                (51, 56, "51-55"),
                (56, 61, "56-60"),
                (61, 66, "61-65"),
                (66, 71, "66-70"),
                (71, 76, "71-75"),
                (76, 81, "76-80"),
                (80, 101, "80+")
            ]
        
        # Get death counts
        death_counts = self.get_death_counts()
        
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
        plt.title(f'Death Counts by Age Group and Sex (Total: {self.total_deaths_tracked})', fontsize=16)
        plt.xticks(x, labels, fontsize=12)
        plt.legend(fontsize=12)
        
        # Improve layout
        plt.tight_layout()
        
        plt.show()
        return 

    
    def plot_death_counts(self, year=None, smoothing=3, age_interval=1, figsize=(12, 8)):
        """
        Plot the absolute number of deaths by age and sex
        
        Args:
            year: If provided, plot deaths for that specific year
                  If None, plot cumulative deaths across all years
            smoothing: Sigma value for Gaussian smoothing (0 for no smoothing)
            age_interval: Group ages by this interval (e.g., 5 for 5-year age groups)
            figsize: Figure size (width, height) in inches
        """
        # Get death counts for specified year or cumulative
        death_counts = self.get_death_counts(year)
        
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
        
        # Plot male and female lines
        plt.plot(ages, male_counts, 'b-', linewidth=2, alpha=0.7, label='Male')
        plt.plot(ages, female_counts, 'r-', linewidth=2, alpha=0.7, label='Female')
        
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
        
        # Customize the plot
        plt.grid(True, alpha=0.3)
        plt.xlabel('Age', fontsize=14)
        plt.ylabel('Number of Deaths', fontsize=14)
        
        # Set title based on whether we're showing yearly or cumulative data
        if year is None:
            title = f'Cumulative Death Counts by Age and Sex (Total: {self.total_deaths_tracked})'
        else:
            # Calculate total deaths for this year
            year_male_deaths = sum(death_counts['Male'].values())
            year_female_deaths = sum(death_counts['Female'].values())
            year_total = year_male_deaths + year_female_deaths
            title = f'Death Counts for Year {year} by Age and Sex (Total: {year_total})'
        
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12)
        
        # Set axis limits
        plt.xlim(0, max(ages))
        plt.ylim(bottom=0)
        
        # Improve layout
        plt.tight_layout()
        
        plt.show()
        return 
    
    
    def plot_aggregated_death_counts(self, year=None, age_groups=None):
        """
        Plot death counts aggregated into standard age groups
        
        Args:
            year: If provided, plot deaths for that specific year
                  If None, plot cumulative deaths across all years
            age_groups: List of tuples (start_age, end_age, label)
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
        
        # Get death counts for specified year or cumulative
        death_counts = self.get_death_counts(year)
        
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
        
        # Set title based on whether we're showing yearly or cumulative data
        if year is None:
            title = f'Cumulative Death Counts by Age Group and Sex (Total: {self.total_deaths_tracked})'
        else:
            # Calculate total deaths for this year
            year_male_deaths = sum(death_counts['Male'].values())
            year_female_deaths = sum(death_counts['Female'].values())
            year_total = year_male_deaths + year_female_deaths
            title = f'Death Counts for Year {year} by Age Group and Sex (Total: {year_total})'
        
        plt.title(title, fontsize=16)
        plt.xticks(x, labels, fontsize=12)
        plt.legend(fontsize=12)
        
        # Improve layout
        plt.tight_layout()
        
        plt.show()
        return
    
    

    
    
    def calculate_and_plot_mortality_rates(self, observed_data_path, year=None, 
                                       age_groups=None, log_scale=True):
        """
        Calculate mortality rates from simulation and compare with observed data
        
        Args:
            observed_data_path: Path to CSV file with observed mortality rates
            year: Year to calculate simulated mortality rates for (None for cumulative)
                  and year to filter observed data (optional)
            age_groups: List of age group start values for grouping
            log_scale: Whether to use log scale for plotting
            
        Returns:
            Figure and axes from the plot
        """
        # Load observed data
        observed_rates = pd.read_csv(observed_data_path)
        
        # Calculate simulated mortality rates
        simulated_rates = mi.calculate_mortality_rates(self, year, age_groups)
        
        # Plot comparison
        return mi.plot_mortality_rates_comparison(
            simulated_rates, 
            observed_rates, 
            year=year, 
            log_scale=log_scale,
            title=f'Simulated vs Observed Mortality Rates' 
                  f'{f" for Year {year}" if year else ""}'
        )
    
    def record_death(self, person_id, age, is_female):
        """Record a death in the tracker with added verification"""
        # Add duplicate detection
        if person_id in self.recorded_deaths:
            print(f"WARNING: Death already recorded for person {person_id}!")
            return
            
        # Store the death info
        self.recorded_deaths.add(person_id)
        
        year = int(self.sim.t.year)
        sex = 'Female' if is_female else 'Male'
        
        if year not in self.deaths_by_year:
            self.deaths_by_year[year] = {'Male': defaultdict(int), 'Female': defaultdict(int)}
            
        # Increment the count for this age and sex
        self.deaths_by_year[year][sex][int(age)] += 1
        
        # For debugging purposes, log deaths of young children
        if age <= 5:
            print(f"Death recorded: {sex}, age {age}, year {year}, person ID {person_id}")
            
    def track_populations_over_time(self):
        """
        Track population counts at the start of each time step 
        for accurate mortality rate calculations
        """
        if not hasattr(self, 'population_by_year'):
            self.population_by_year = {}
        
        year = int(self.sim.t.year)
        ages = self.sim.people.age
        females = self.sim.people.female
        alive = self.sim.people.alive
        
        # Count population by age and sex
        population = {'Male': defaultdict(int), 'Female': defaultdict(int)}
        
        # Only count alive people
        alive_indices = np.where(alive)[0]
        
        for idx in alive_indices:
            age = int(ages[idx])
            sex = 'Female' if females[idx] else 'Male'
            population[sex][age] += 1
        
        # Store population counts for this year
        self.population_by_year[year] = population
        
        # Print summary
        young_male = sum(population['Male'].get(age, 0) for age in range(5))
        young_female = sum(population['Female'].get(age, 0) for age in range(5))
        total_male = sum(population['Male'].values())
        total_female = sum(population['Female'].values())
        
        print(f"Time {year}: Population tracked - Male: {total_male} (under 5: {young_male}), "
              f"Female: {total_female} (under 5: {young_female})")
        

# In mighti/death_tracker.py

# After your DeathTracker class...

def check_child_mortality_parameters(sim):
    """Check if child mortality parameters are set correctly in the simulation"""
    # This will depend on how mortality is implemented in your model
    
    print("\nChecking child mortality parameters:")
    
    # Check if there are specific mortality parameters for children
    if hasattr(sim.pars, 'mortality_by_age') or hasattr(sim, 'mortality_by_age'):
        mortality_pars = getattr(sim.pars, 'mortality_by_age', getattr(sim, 'mortality_by_age', None))
        
        if mortality_pars is not None:
            # Print the mortality parameters for young ages
            print("Mortality parameters for young ages:")
            for age in range(6):
                if age in mortality_pars:
                    print(f"  Age {age}: {mortality_pars[age]}")
                else:
                    print(f"  Age {age}: Not specified")
    
    # Look for specific mortality rates
    if hasattr(sim, 'people') and hasattr(sim.people, 'mort_probs'):
        print("\nMortality probabilities for young ages:")
        mort_probs = sim.people.mort_probs
        ages = sim.people.age
        females = sim.people.female
        
        # Sample some young people to check their mortality probabilities
        for age in range(6):
            # Find agents of this age
            if hasattr(sim.people, 'alive'):
                young_males = np.where((ages == age) & (~females) & sim.people.alive)[0]
                young_females = np.where((ages == age) & females & sim.people.alive)[0]
            else:
                young_males = np.where((ages == age) & (~females))[0]
                young_females = np.where((ages == age) & females)[0]
                
            # Get mortality probabilities for each
            if len(young_males):
                male_probs = mort_probs[young_males]
                print(f"  Age {age} Male: {len(young_males)} agents, avg mort_prob: {male_probs.mean():.4f}")
                
            if len(young_females):
                female_probs = mort_probs[young_females]
                print(f"  Age {age} Female: {len(young_females)} agents, avg mort_prob: {female_probs.mean():.4f}")
    
    # Check if there are any specific disease parameters affecting children
    if hasattr(sim, 'diseases'):
        print("\nDisease parameters affecting children:")
        for disease_name, disease in sim.diseases.items():
            if hasattr(disease, 'pars'):
                child_related = {}
                for param_name, value in disease.pars.items():
                    if 'child' in param_name.lower() or 'infant' in param_name.lower() or 'age' in param_name.lower():
                        child_related[param_name] = value
                
                if child_related:
                    print(f"  {disease_name}: {child_related}")
                    
    # Look for demographic parameters that might affect children
    print("\nDemographic parameters:")
    if hasattr(sim, 'pars'):
        for key, value in sim.pars.items():
            if isinstance(key, str) and ('child' in key.lower() or 'infant' in key.lower() or 'mortality' in key.lower()):
                print(f"  {key}: {value}")


# Add this to mighti/death_tracker.py or wherever is appropriate

def analyze_mortality_implementation(sim, death_tracker):
    """
    Analyze how mortality is implemented in the simulation
    and compare to actual death rates
    """
    print("\n--- Mortality Implementation Analysis ---")
    
    # Check for mortality calculation functions
    mortality_functions = []
    for attr_name in dir(sim):
        attr = getattr(sim, attr_name)
        if callable(attr) and ('mortality' in attr_name.lower() or 'death' in attr_name.lower()):
            mortality_functions.append(attr_name)
    
    print(f"Potential mortality functions found: {mortality_functions}")
    
    # Check the people object for mortality-related attributes
    people_mortality_attrs = []
    for attr_name in dir(sim.people):
        if 'mort' in attr_name.lower() or 'death' in attr_name.lower() or 'die' in attr_name.lower():
            people_mortality_attrs.append(attr_name)
    
    print(f"People mortality attributes found: {people_mortality_attrs}")
    
    # Get current people data
    ages = sim.people.age
    females = sim.people.female
    alive = sim.people.alive
    
    # Check if there's a mortality attribute we can analyze
    if hasattr(sim.people, 'mort_probs'):
        mort_probs = sim.people.mort_probs
        print("\nAnalyzing mortality probabilities by age and sex:")
        
        # Calculate average mortality probability by age and sex
        age_max = int(np.max(ages)) + 1
        male_mort_by_age = np.zeros(age_max)
        female_mort_by_age = np.zeros(age_max)
        male_count_by_age = np.zeros(age_max)
        female_count_by_age = np.zeros(age_max)
        
        for i in range(len(ages)):
            if alive[i]:
                age = int(ages[i])
                if females[i]:
                    female_mort_by_age[age] += mort_probs[i]
                    female_count_by_age[age] += 1
                else:
                    male_mort_by_age[age] += mort_probs[i]
                    male_count_by_age[age] += 1
        
        # Print average mortality rates for each age
        print("\nAverage mortality probabilities:")
        print("Age | Male Count | Male Mort Prob | Female Count | Female Mort Prob")
        print("----|------------|----------------|--------------|----------------")
        
        for age in range(min(20, age_max)):  # Focus on first 20 years
            male_avg = male_mort_by_age[age] / male_count_by_age[age] if male_count_by_age[age] > 0 else 0
            female_avg = female_mort_by_age[age] / female_count_by_age[age] if female_count_by_age[age] > 0 else 0
            
            print(f"{age:3d} | {int(male_count_by_age[age]):10d} | {male_avg:14.6f} | {int(female_count_by_age[age]):12d} | {female_avg:16.6f}")
        
        # Check if mortality probabilities seem reasonable
        young_male_avg = np.sum(male_mort_by_age[:5]) / np.sum(male_count_by_age[:5]) if np.sum(male_count_by_age[:5]) > 0 else 0
        young_female_avg = np.sum(female_mort_by_age[:5]) / np.sum(female_count_by_age[:5]) if np.sum(female_count_by_age[:5]) > 0 else 0
        
        print(f"\nAverage mortality probability for ages 0-4:")
        print(f"  Male: {young_male_avg:.6f}")
        print(f"  Female: {young_female_avg:.6f}")
        
        if young_male_avg > 0.5 or young_female_avg > 0.5:
            print("\nWARNING: Child mortality probabilities are extremely high!")
            print("This explains why almost all children are dying in the simulation.")
            print("Normal child mortality rates should be around 0.01-0.1 at most.")
    
    # Also look at how deaths are distributed over the year
    if hasattr(death_tracker, 'deaths_by_year'):
        year = int(sim.t.year)
        if year in death_tracker.deaths_by_year:
            total_deaths = 0
            young_deaths = 0
            
            for sex in ['Male', 'Female']:
                for age, count in death_tracker.deaths_by_year[year][sex].items():
                    total_deaths += count
                    if age < 5:
                        young_deaths += count
            
            print(f"\nIn year {year}, {young_deaths} out of {total_deaths} deaths ({young_deaths/total_deaths*100:.1f}%) were children under 5")
    
    # Try to identify the specific mortality calculation
    print("\nAttempting to trace mortality calculation process...")
    
    # Look at a specific function that might be relevant
    mortality_fn_names = ["update_mortality", "calculate_mortality", "apply_mortality", 
                     "compute_deaths", "apply_deaths", "update_deaths"]
    
    for fn_name in mortality_fn_names:
        if hasattr(sim, fn_name) and callable(getattr(sim, fn_name)):
            fn = getattr(sim, fn_name)
            print(f"\nFound mortality function: {fn_name}")
            print(f"Function details: {fn.__doc__ if fn.__doc__ else 'No documentation available'}")
            
# Modified version of calculate_adjusted_mortality_rates function for death_tracker.py
# This applies age-specific caps to mortality rates

def calculate_adjusted_mortality_rates(death_tracker, year=None, age_groups=None, max_rate=2.0):
    """
    Calculate mortality rates from death tracker data with age-appropriate safeguards
    
    Args:
        death_tracker: DeathTracker instance
        year: Specific year to calculate for (None for cumulative)
        age_groups: List of age group start values for grouping
        max_rate: Maximum allowed mortality rate (cap for outliers), default 2.0 matching observed data
        
    Returns:
        DataFrame with mortality rates by age group and sex
    """
    sim = death_tracker.sim
    
    # Default age groups if not provided
    if age_groups is None:
        age_groups = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    
    # Get death counts from tracker
    death_counts = death_tracker.get_death_counts(year)
    
    # Get current population state
    ages = sim.people.age
    females = sim.people.female
    alive = sim.people.alive
    
    # Count population by age and sex
    population = {'Male': defaultdict(int), 'Female': defaultdict(int)}
    
    # Only count alive people
    alive_indices = np.where(alive)[0]
    
    for idx in alive_indices:
        age = int(ages[idx])
        sex = 'Female' if females[idx] else 'Male'
        
        # Find the appropriate age group
        for i in range(len(age_groups)):
            if i == len(age_groups) - 1 or (age >= age_groups[i] and age < age_groups[i+1]):
                age_group = age_groups[i]
                break
                
        population[sex][age_group] += 1
    
    # Calculate deaths by age group
    deaths_by_group = {'Male': defaultdict(int), 'Female': defaultdict(int)}
    
    for sex in ['Male', 'Female']:
        for age, count in death_counts[sex].items():
            # Find the appropriate age group
            for i in range(len(age_groups)):
                if i == len(age_groups) - 1 or (age >= age_groups[i] and age < age_groups[i+1]):
                    age_group = age_groups[i]
                    break
                    
            deaths_by_group[sex][age_group] += count
    
    # Calculate mid-year adjusted population
    adjusted_population = {'Male': {}, 'Female': {}}
    for sex in ['Male', 'Female']:
        for age_group in age_groups:
            deaths = deaths_by_group[sex][age_group] 
            pop = population[sex].get(age_group, 0)
            
            # For robust estimation, assume deaths were distributed throughout the year
            # Add back half of deaths to approximate mid-year population
            # Ensure we don't have zero population
            adjusted_population[sex][age_group] = max(1, pop + deaths * 0.5)
    
    # Calculate mortality rates
    data = []
    
    current_year = sim.t.year if year is None else year
    
    for sex in ['Male', 'Female']:
        for age_group in age_groups:
            deaths = deaths_by_group[sex][age_group]
            pop = adjusted_population[sex].get(age_group, 0)
            
            # Calculate mortality rate (avoid division by zero)
            raw_rate = deaths / pop if pop > 0 else 0
            
            # Apply age-specific caps
            if age_group < 5:
                # Cap young children's mortality at 0.05 (5%)
                mx = min(raw_rate, 0.05)
            elif age_group < 80:
                # Cap middle ages at reasonble rates (0.5 or 50%)
                mx = min(raw_rate, 0.5)
            else:
                # Allow higher mortality for oldest ages (up to max_rate)
                mx = min(raw_rate, max_rate)
            
            # Add to data
            data.append({
                'Time': current_year,
                'Sex': sex,
                'AgeGrpStart': age_group,
                'mx': mx
            })
    
    # Create DataFrame
    mortality_rates = pd.DataFrame(data)
    
    # Print summary of calculated rates
    print("\nAdjusted mortality rates with age-appropriate caps:")
    for sex in ['Male', 'Female']:
        sex_rates = mortality_rates[mortality_rates['Sex'] == sex]
        for age_group in [0, 5, 15, 45, 75, 95]:  # Print a few representative age groups
            if age_group in age_groups:  # Make sure this age group exists
                row = sex_rates[sex_rates['AgeGrpStart'] == age_group]
                if len(row) > 0:
                    rate = row['mx'].values[0]
                    deaths = deaths_by_group[sex][age_group]
                    pop_mid = adjusted_population[sex].get(age_group, 0)
                    
                    # Check if rate was capped
                    max_cap = 0.05 if age_group < 5 else (0.5 if age_group < 80 else max_rate)
                    was_capped = raw_rate > max_cap
                    capped_note = " (capped)" if was_capped else ""
                    
                    print(f"{sex} age {age_group}+: deaths={deaths}, pop={pop_mid:.1f}, rate={rate:.4f}{capped_note}")
    
    return mortality_rates