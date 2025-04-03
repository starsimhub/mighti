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
        
        # Add to DeathTracker.step() method
        if getattr(self.sim.t, 'dt_year', 1) == 1:  # Only track once per year, with safe attribute access
            self.track_population()
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
    
    def plot_age_distribution_over_time(self, inityear, endyear, age_groups=None):
        """
        Plot the age distribution of the population over time by sex and age group
        
        Args:
            inityear: Start year of the simulation
            endyear: End year of the simulation
            age_groups: List of tuples (start_age, end_age, label)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
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
        
        # Initialize dictionaries to store population data
        population_data = {
            'years': list(range(inityear, endyear + 1)),
            'male': {group[2]: [] for group in age_groups},
            'female': {group[2]: [] for group in age_groups},
        }
        
        # Retrieve population data for each year
        for year in population_data['years']:
            total_population = self.population_by_year.get(year, {'Male': {}, 'Female': {}})
            
            for start_age, end_age, label in age_groups:
                # Calculate male population in this age group
                male_population = sum(total_population.get('Male', {}).get(age, 0) for age in range(start_age, end_age))
                population_data['male'][label].append(male_population)
                
                # Calculate female population in this age group
                female_population = sum(total_population.get('Female', {}).get(age, 0) for age in range(start_age, end_age))
                population_data['female'][label].append(female_population)
        
        print(f"population_data male: {population_data['male']}")
        print(f"population_data female: { population_data['female']}")
        # Create figure with two panels
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Define a consistent color palette for age groups
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Plot male population data
        for i, (label, counts) in enumerate(population_data['male'].items()):
            if len(counts) == len(population_data['years']):
                # Apply color consistently based on age group
                color_idx = i % len(colors)
                
                # Plot population counts with solid line
                ax1.plot(population_data['years'], counts, 
                        linestyle='-', marker='o', markersize=5, 
                        color=colors[color_idx], linewidth=2, label=label)
        
        # Plot female population data
        for i, (label, counts) in enumerate(population_data['female'].items()):
            if len(counts) == len(population_data['years']):
                # Apply color consistently based on age group
                color_idx = i % len(colors)
                
                # Plot population counts with solid line
                ax2.plot(population_data['years'], counts,
                        linestyle='-', marker='o', markersize=5,
                        color=colors[color_idx], linewidth=2, label=label)
        
        # Configure male panel
        ax1.set_title('Male Population by Age Group', fontsize=14)
        ax1.set_ylabel('Population Count', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Configure female panel
        ax2.set_title('Female Population by Age Group', fontsize=14)
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Population Count', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Improve layout
        plt.tight_layout()
        plt.suptitle('Population Age Distribution Over Time by Sex and Age Group', fontsize=16)
        plt.subplots_adjust(top=0.93)
        
        plt.show()
        return
    
    def plot_mortality_rates_for_young_children(self, inityear, endyear):
        """
        Plot mortality rates for young children (0-4 age group) over time
        
        Args:
            inityear: Start year of the simulation
            endyear: End year of the simulation
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Initialize data to store mortality rates
        mortality_rates = {
            'years': list(range(inityear, endyear + 1)),
            'male': [],
            'female': []
        }
        population = {
            'years': list(range(inityear, endyear + 1)),
            'male': [],
            'female': []
        }
        deaths = {
            'years': list(range(inityear, endyear + 1)),
            'male': [],
            'female': []
        }
        births = {
            'years': list(range(inityear, endyear + 1)),
            'male': [],
            'female': []
        }
        
        # Track previous population for birth calculation
        previous_population = {
            'male': 0,
            'female': 0
        }
        
        # Retrieve mortality rates for each year
        for year in mortality_rates['years']:
            yearly_deaths = self.get_death_counts(year)
            total_population = self.population_by_year.get(year, {'Male': {}, 'Female': {}})
            
            # Calculate mortality rate for male children (0-4 age group)
            male_deaths = sum(yearly_deaths['Male'].get(age, 0) for age in range(0, 5))
            male_population = sum(total_population.get('Male', {}).get(age, 0) for age in range(0, 5))
            male_mortality_rate = male_deaths / male_population if male_population > 0 else 0
            mortality_rates['male'].append(male_mortality_rate)
            population['male'].append(male_population)
            deaths['male'].append(male_deaths)
            
            # Calculate male births
            current_population_male = sum(total_population.get('Male', {}).values())
            male_births = max(0, current_population_male - previous_population['male'])
            births['male'].append(male_births)
            previous_population['male'] = current_population_male
            
            # Calculate mortality rate for female children (0-4 age group)
            female_deaths = sum(yearly_deaths['Female'].get(age, 0) for age in range(0, 5))
            female_population = sum(total_population.get('Female', {}).get(age, 0) for age in range(0, 5))
            female_mortality_rate = female_deaths / female_population if female_population > 0 else 0
            mortality_rates['female'].append(female_mortality_rate)
            population['female'].append(female_population)
            deaths['female'].append(female_deaths)
            
            # Calculate female births
            current_population_female = sum(total_population.get('Female', {}).values())
            female_births = max(0, current_population_female - previous_population['female'])
            births['female'].append(female_births)
            previous_population['female'] = current_population_female
        
        print(f"mortality_rates male: {mortality_rates['male']}")
        print(f"mortality_rates female: {mortality_rates['female']}")
        print(f"population male: {population['male']}")
        print(f"population female: {population['female']}")
        print(f"deaths male: {deaths['male']}")
        print(f"deaths female: {deaths['female']}")
        print(f"births male: {births['male']}")
        print(f"births female: {births['female']}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot male and female mortality rates
        ax.plot(mortality_rates['years'], mortality_rates['male'], linestyle='-', marker='o', color='blue', linewidth=2, label='Male')
        ax.plot(mortality_rates['years'], mortality_rates['female'], linestyle='-', marker='o', color='red', linewidth=2, label='Female')
        
        # Configure plot
        ax.set_title('Mortality Rates for Young Children (0-4 Age Group) Over Time', fontsize=14)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Mortality Rate (Deaths per Population)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Improve layout
        plt.tight_layout()
        plt.show()
        return 

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

    def plot_sex_age_group_death_counts_over_time(self, inityear, endyear, age_groups=None):
        """
        Plot death counts over time by sex and age group
        
        Args:
            inityear: Start year of the simulation
            endyear: End year of the simulation
            age_groups: List of tuples (start_age, end_age, label)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
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
        
        # Initialize dictionaries to store death counts over time
        death_data = {
            'years': list(range(inityear, endyear + 1)),
            'male': {group[2]: [] for group in age_groups},
            'female': {group[2]: [] for group in age_groups},
        }
        
        # Retrieve death counts for each year
        for year in death_data['years']:
            yearly_deaths = self.get_death_counts(year)
            
            for start_age, end_age, label in age_groups:
                # Calculate male deaths in this age group
                male_deaths = sum(yearly_deaths['Male'].get(age, 0) for age in range(start_age, end_age))
                death_data['male'][label].append(male_deaths)
                
                # Calculate female deaths in this age group
                female_deaths = sum(yearly_deaths['Female'].get(age, 0) for age in range(start_age, end_age))
                death_data['female'][label].append(female_deaths)
        
        # Create figure with two panels
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Define a consistent color palette for age groups
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Plot male death counts data
        for i, (label, counts) in enumerate(death_data['male'].items()):
            if len(counts) == len(death_data['years']):
                # Apply color consistently based on age group
                color_idx = i % len(colors)
                
                # Plot death counts with solid line
                ax1.plot(death_data['years'], counts, 
                        linestyle='-', marker='o', markersize=5, 
                        color=colors[color_idx], linewidth=2)
        
        # Plot female death counts data
        for i, (label, counts) in enumerate(death_data['female'].items()):
            if len(counts) == len(death_data['years']):
                # Apply color consistently based on age group
                color_idx = i % len(colors)
                
                # Plot death counts with solid line
                ax2.plot(death_data['years'], counts,
                        linestyle='-', marker='o', markersize=5,
                        color=colors[color_idx], linewidth=2)
        
        # Configure male panel
        ax1.set_title('Male Death Counts by Age Group', fontsize=14)
        ax1.set_ylabel('Number of Deaths', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Configure female panel
        ax2.set_title('Female Death Counts by Age Group', fontsize=14)
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Number of Deaths', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Improve layout
        plt.tight_layout()
        plt.suptitle('Death Counts Over Time by Age Group and Sex', fontsize=16)
        plt.subplots_adjust(top=0.93)
        
        plt.show()
        return fig, (ax1, ax2)
    
    def plot_normalized_sex_age_group_annual_death_counts(self, inityear, endyear, age_groups=None):
        """
        Plot annual death counts normalized by total population by sex and age group for each year
        
        Args:
            inityear: Start year of the simulation
            endyear: End year of the simulation
            age_groups: List of tuples (start_age, end_age, label)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
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
        
        # Initialize dictionaries to store annual death counts
        annual_death_data = {
            'years': list(range(inityear, endyear + 1)),
            'male': {group[2]: [] for group in age_groups},
            'female': {group[2]: [] for group in age_groups},
        }
        
        # Retrieve death counts and population data for each year
        for year in annual_death_data['years']:
            yearly_deaths = self.get_death_counts(year)
            total_population = sum(self.population_by_year.get(year, {'Male': {}, 'Female': {}}).get('Male', {}).values()) + \
                               sum(self.population_by_year.get(year, {'Male': {}, 'Female': {}}).get('Female', {}).values())
            
            for start_age, end_age, label in age_groups:
                # Calculate male deaths in this age group normalized by total population
                male_deaths = sum(yearly_deaths['Male'].get(age, 0) for age in range(start_age, end_age))
                male_death_rate = male_deaths / total_population if total_population > 0 else 0
                annual_death_data['male'][label].append(male_death_rate)
                # Calculate female deaths in this age group normalized by total population
                female_deaths = sum(yearly_deaths['Female'].get(age, 0) for age in range(start_age, end_age))
                female_death_rate = female_deaths / total_population if total_population > 0 else 0
                annual_death_data['female'][label].append(female_death_rate)

        print(f"annual_death_data male: {annual_death_data['male']}")
        print(f"annual_death_data female: {annual_death_data['female']}")

 
        # Create figure with two panels
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Define a consistent color palette for age groups
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Plot male death rates data
        for i, (label, counts) in enumerate(annual_death_data['male'].items()):
            if len(counts) == len(annual_death_data['years']):
                # Apply color consistently based on age group
                color_idx = i % len(colors)
                
                # Plot death rates with solid line
                ax1.plot(annual_death_data['years'], counts, 
                        linestyle='-', marker='o', markersize=5, 
                        color=colors[color_idx], linewidth=2, label=label)
        
        # Plot female death rates data
        for i, (label, counts) in enumerate(annual_death_data['female'].items()):
            if len(counts) == len(annual_death_data['years']):
                # Apply color consistently based on age group
                color_idx = i % len(colors)
                
                # Plot death rates with solid line
                ax2.plot(annual_death_data['years'], counts,
                        linestyle='-', marker='o', markersize=5,
                        color=colors[color_idx], linewidth=2, label=label)
        
        # Configure male panel
        ax1.set_title('Male Annual Death Rates by Age Group', fontsize=14)
        ax1.set_ylabel('Death Rate (per Total Population)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Configure female panel
        ax2.set_title('Female Annual Death Rates by Age Group', fontsize=14)
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Death Rate (per Total Population)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Improve layout
        plt.tight_layout()
        plt.suptitle('Annual Death Rates by Age Group and Sex (Normalized by Total Population)', fontsize=16)
        plt.subplots_adjust(top=0.93)
        
        plt.show()
        return fig, (ax1, ax2)
  

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
    
    def plot_population_over_time(self, inityear, endyear, age_groups=None, observed_data_path='demography/eswatini_age_distribution.csv', normalize=True):
        """
        Plot population changes over time by age group and sex
        
        Args:
            inityear: Start year of the simulation
            endyear: End year of the simulation
            age_groups: List of tuples defining age groups (start_age, end_age, label)
            observed_data_path: Path to CSV file with observed age distribution
            normalize: If True, normalize data as percentages of total population
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Define age groups if not provided
        if age_groups is None:
            age_groups = [
                (0, 5, "0-4"),
                (5, 15, "5-14"),
                (15, 25, "15-24"),
                (25, 45, "25-44"),
                (45, 65, "45-64"),
                (65, 101, "65+")
            ]
        
        # Define a consistent color palette for age groups
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
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
            'female': {group[2]: [] for group in age_groups},
            'total_male': [],
            'total_female': []
        }
        
        # Calculate population by age group for each year
        for year in population_data['years']:
            if year not in self.population_by_year:
                continue
                
            # Calculate total population for this year
            total_male = sum(self.population_by_year[year]['Male'].values())
            total_female = sum(self.population_by_year[year]['Female'].values())
            
            population_data['total_male'].append(total_male)
            population_data['total_female'].append(total_female)
                
            # Calculate for each age group
            for start_age, end_age, label in age_groups:
                # Calculate male population in this age group
                male_count = sum(self.population_by_year[year]['Male'].get(age, 0) 
                               for age in range(start_age, end_age))
                population_data['male'][label].append(male_count)
                
                # Calculate female population in this age group
                female_count = sum(self.population_by_year[year]['Female'].get(age, 0) 
                                for age in range(start_age, end_age))
                population_data['female'][label].append(female_count)
        
        # Create figure with two panels
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Define a simple function to construct the legend with both obs and sim entries
        def add_legend_entries(ax):
            # Add legend entries for line styles (sim vs obs)
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='black', linestyle='-', marker='o', markersize=5, label='Simulated'),
                Line2D([0], [0], color='black', linestyle='--', marker='s', markersize=5, label='Observed')
            ]
            # Add legend for colors (age groups)
            for i, (_, _, label) in enumerate(age_groups):
                if i < len(colors):
                    legend_elements.append(Line2D([0], [0], color=colors[i], lw=2, label=label))
            
            ax.legend(handles=legend_elements, loc='upper left', ncol=2, fontsize=10)
        
        # Initialize dictionaries for observed data
        obs_male = {}
        obs_female = {}
        obs_total_male = []
        obs_total_female = []
        obs_year_ints = []
        
        # Process observed data if available
        if observed_data is not None:
            observed_years = [str(y) for y in range(inityear, endyear + 1) if str(y) in observed_data.columns]
            
            if observed_years:
                # Convert year strings to integers for plotting
                obs_year_ints = [int(y) for y in observed_years]
                
                # Calculate total population for each year first
                for year_str in observed_years:
                    male_total = observed_data[observed_data['sex'] == 'Male'][year_str].sum()
                    female_total = observed_data[observed_data['sex'] == 'Female'][year_str].sum()
                    
                    obs_total_male.append(male_total)
                    obs_total_female.append(female_total)
                
                for start_age, end_age, label in age_groups:
                    # Get male data for this age group by summing across ages
                    male_rows = observed_data[(observed_data['sex'] == 'Male') & 
                                             (observed_data['age'] >= start_age) & 
                                             (observed_data['age'] < end_age)]
                    
                    # Get female data for this age group
                    female_rows = observed_data[(observed_data['sex'] == 'Female') & 
                                               (observed_data['age'] >= start_age) & 
                                               (observed_data['age'] < end_age)]
                    
                    # Initialize data lists
                    obs_male[label] = []
                    obs_female[label] = []
                    
                    # Extract values for each year
                    for i, year_str in enumerate(observed_years):
                        # Sum populations for this age group and year
                        male_pop = male_rows[year_str].sum()  # Raw value from data
                        female_pop = female_rows[year_str].sum()  # Raw value from data
                        
                        # Normalize if requested
                        if normalize:
                            male_pop = 100 * male_pop / obs_total_male[i] if obs_total_male[i] > 0 else 0
                            female_pop = 100 * female_pop / obs_total_female[i] if obs_total_female[i] > 0 else 0
                        else:
                            # If not normalizing, scale raw values (in thousands) to actual counts
                            male_pop *= 1000
                            female_pop *= 1000
                        
                        obs_male[label].append(male_pop)
                        obs_female[label].append(female_pop)
                
                print(f"Added observed data for years: {', '.join(observed_years)}")
            else:
                print("No matching years found in observed data")
        
        # Plot male population data
        for i, (label, counts) in enumerate(population_data['male'].items()):
            if len(counts) == len(population_data['years']):
                # Apply color consistently based on age group
                color_idx = i % len(colors)
                
                # Normalize if requested
                if normalize:
                    counts = [100 * c / t for c, t in zip(counts, population_data['total_male'])]
                
                # Plot simulated data with solid line
                ax1.plot(population_data['years'], counts, 
                        linestyle='-', marker='o', markersize=5, 
                        color=colors[color_idx], linewidth=2)
                
                # Plot observed data with the same color but dashed line
                if label in obs_male and len(obs_year_ints) > 0:
                    ax1.plot(obs_year_ints, obs_male[label],
                            linestyle='--', marker='s', markersize=5,
                            color=colors[color_idx], linewidth=2)
        
        # Plot female population data
        for i, (label, counts) in enumerate(population_data['female'].items()):
            if len(counts) == len(population_data['years']):
                # Apply color consistently based on age group
                color_idx = i % len(colors)
                
                # Normalize if requested
                if normalize:
                    counts = [100 * c / t for c, t in zip(counts, population_data['total_female'])]
                
                # Plot simulated data with solid line
                ax2.plot(population_data['years'], counts,
                        linestyle='-', marker='o', markersize=5,
                        color=colors[color_idx], linewidth=2)
                
                # Plot observed data with the same color but dashed line
                if label in obs_female and len(obs_year_ints) > 0:
                    ax2.plot(obs_year_ints, obs_female[label],
                            linestyle='--', marker='s', markersize=5,
                            color=colors[color_idx], linewidth=2)
        
        # Add legends to both panels
        add_legend_entries(ax1)
        add_legend_entries(ax2)
        
        # Configure male panel
        ax1.set_title('Male Population by Age Group', fontsize=14)
        y_label = 'Percentage of Population' if normalize else 'Population Count'
        ax1.set_ylabel(y_label, fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Configure female panel
        ax2.set_title('Female Population by Age Group', fontsize=14)
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel(y_label, fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Improve layout
        plt.tight_layout()
        title = 'Population Structure Over Time by Age Group and Sex (Percentage)' if normalize else 'Population Changes Over Time by Age Group and Sex'
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.93)
        
        return fig, (ax1, ax2) 
    
    
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
        
    def track_population(self):
        """Track population by age and sex at the current timestep"""
        if not hasattr(self, 'population_by_year'):
            self.population_by_year = {}
    
        # Get current year and population state - Fixed the year access
        year = int(self.sim.t.now())  # Use t.now() instead of t.year
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
            population[sex][age] = population[sex].get(age, 0) + 1
    
        # Store population for this year
        self.population_by_year[year] = population
    
        # Print summary statistics
        total_male = sum(population['Male'].values())
        total_female = sum(population['Female'].values())
        # print(f"Year {year} population: Male={total_male:,}, Female={total_female:,}, Total={total_male + total_female:,}")
        # print(f"Population by age and sex: {population}")

def check_child_mortality_parameters(sim):
    """Check if child mortality parameters are set correctly in the simulation"""
    print("\nChecking child mortality parameters:")
    
    # Check for parameters under deaths
    if hasattr(sim.pars, 'deaths'):
        deaths_pars = sim.pars.deaths
        # print("\nDeath parameters:")
        # if hasattr(deaths_pars, 'death_rate'):
        #     print(f"  Death rate: {deaths_pars.death_rate}")
        # if hasattr(deaths_pars, 'rel_death'):
        #     print(f"  Relative death rate: {deaths_pars.rel_death}")
            
    # Inspect the death_rate_data attribute in the deaths module
    # if hasattr(sim.demographics[1], 'death_rate_data'):
    #     print(sim.demographics[1].death_rate_data.head())
    #     print(sim.demographics[1].death_rate_data.index.names)
    #     print(sim.demographics[1].death_rate_data.columns)
    else:
        print("death_rate_data attribute not found in the deaths module.")        
    
    # Check for pregnancy parameters
    # if hasattr(sim.pars, 'pregnancy'):
    #     pregnancy_pars = sim.pars.pregnancy
    #     print("\nPregnancy parameters:")
    #     if hasattr(pregnancy_pars, 'p_neonatal_death'):
    #         print(f"  Neonatal death probability: {pregnancy_pars.p_neonatal_death}")
    #     if hasattr(pregnancy_pars, 'p_maternal_death'):
    #         print(f"  Maternal death probability: {pregnancy_pars.p_maternal_death}")
    #     if hasattr(pregnancy_pars, 'fertility_rate'):
    #         print(f"  Fertility rate: {pregnancy_pars.fertility_rate}")
    
    # Check for mortality probabilities
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
    
    # Check for disease parameters affecting children
    if hasattr(sim, 'diseases'):
        print("\nDisease parameters affecting children:")
        for disease_name, disease in sim.diseases.items():
            if hasattr(disease, 'pars'):
                child_related = {}
                for param_name, value in disease.pars.items():
                    if 'child' in param_name.lower() or 'infant' in param_name.lower() or 'age' in param_name.lower():
                        child_related[param_name] = value
                
                if child_related:
                    print(f"  {disease_name}:")
                    for param_name, value in child_related.items():
                        print(f"    {param_name}: {value}")
                    
    # Check demographic parameters that might affect children
    print("\nDemographic parameters:")
    if hasattr(sim, 'pars'):
        for key, value in sim.pars.items():
            if isinstance(key, str) and ('child' in key.lower() or 'infant' in key.lower() or 'mortality' in key.lower()):
                print(f"  {key}: {value}")

def analyze_mortality_implementation(sim, death_tracker):
    """
    Analyze how mortality is implemented in the simulation
    and compare to actual death rates
    """
    print("\n--- Mortality Implementation Analysis ---")
    
    # Check for mortality calculation functions in sim
    mortality_functions = []
    for attr_name in dir(sim):
        attr = getattr(sim, attr_name)
        if callable(attr) and ('mortality' in attr_name.lower() or 'death' in attr_name.lower()):
            mortality_functions.append(attr_name)
    
    print(f"Potential mortality functions found in sim: {mortality_functions}")
    
    # Check for mortality calculation functions in death_tracker
    mortality_functions = []
    for attr_name in dir(death_tracker):
        attr = getattr(death_tracker, attr_name)
        if callable(attr) and ('mortality' in attr_name.lower() or 'death' in attr_name.lower()):
            mortality_functions.append(attr_name)
    
    print(f"Potential mortality functions found in death_tracker: {mortality_functions}")
    
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
    
    # Look at relevant functions in death_tracker
    if hasattr(death_tracker, 'calculate_and_plot_mortality_rates'):
        print("\nFound function: calculate_and_plot_mortality_rates")
        # death_tracker.calculate_and_plot_mortality_rates()
    
    if hasattr(death_tracker, 'get_death_counts'):
        print("\nFound function: get_death_counts")
        death_counts = death_tracker.get_death_counts()
        print(f"Death counts: {death_counts}")
    
    if hasattr(death_tracker, 'record_death'):
        print("\nFound function: record_death")
 
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
    print(f"Current simulation year: {current_year}")
    
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