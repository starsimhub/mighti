import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def calculate_mortality_rates(deaths_module, year=None, max_age=100):
    """
    Calculate mortality rates from death counts and population for each year of age.
    """
    sim = deaths_module.sim

    ages = sim.people.age
    females = sim.people.female
    alive = sim.people.alive

    print(f"Alive array (first 10): {alive[:10]}")  # Debug print
    print(f"Ages array (first 10): {ages[:10]}")  # Debug print
    print(f"Females array (first 10): {females[:10]}")  # Debug print

    population = {'Male': np.zeros(max_age + 1), 'Female': np.zeros(max_age + 1)}
    alive_indices = np.where(alive)[0]

    for idx in alive_indices:
        age = int(ages[idx])
        if age > max_age:
            continue
        sex = 'Female' if females[idx] else 'Male'
        population[sex][age] += 1

    print(f"Population after counting (first 10 ages): {[(age, population['Male'][age], population['Female'][age]) for age in range(10)]}")  # Debug print

    deaths_by_age = {'Male': np.zeros(max_age + 1), 'Female': np.zeros(max_age + 1)}
    for sex in deaths_module.death_tracking:
        for age in range(max_age + 1):
            deaths_by_age[sex][age] = deaths_module.death_tracking[sex][age]

    print(f"Deaths by age after counting (first 10 ages): {[(age, deaths_by_age['Male'][age], deaths_by_age['Female'][age]) for age in range(10)]}")  # Debug print

    mortality_rates = []
    for age in range(max_age + 1):
        for sex in ['Male', 'Female']:
            pop = population[sex][age]
            deaths = deaths_by_age[sex][age]
            rate = deaths / pop if pop > 0 else 0
            mortality_rates.append({'Time': year if year is not None else int(sim.t.year), 'Age': age, 'Sex': sex, 'mx': rate})
            print(f" age: {age}, pop: {pop}, death: {deaths}, rate: {rate}")

    return pd.DataFrame(mortality_rates)

def calculate_life_table(mortality_rates, sex='Total', max_age=100):
    """
    Calculate a complete life table based on age-specific mortality rates
    
    Args:
        mortality_rates: Dictionary with age as key and mortality rate (m_x) as value
        sex: 'Male', 'Female', or 'Total' for labeling
        max_age: Maximum age to consider (default 100)
        
    Returns:
        DataFrame containing the complete life table
    """
    # Initialize lists to store life table columns
    ages = []
    l_x = []  # Survivorship function (number of persons alive at age x)
    d_x = []  # Number of deaths between age x and x+1
    q_x = []  # Probability of dying at age x
    m_x = []  # Mortality rate at age x
    L_x = []  # Person-years lived between age x and x+1
    T_x = []  # Person-years lived from age x until all have died
    e_x = []  # Life expectancy at age x
    
    # Set the radix of the life table (starting population)
    radix = 100000
    
    # Fill in any missing ages with zero mortality rate
    for age in range(max_age + 1):
        if age not in mortality_rates:
            mortality_rates[age] = 0.0
    
    # Calculate life table columns
    current_l_x = radix
    
    print(f"\nBuilding Life Table for {sex}")
    print(f"{'Age':<5}{'m(x)':<10}{'q(x)':<10}{'l(x)':<10}{'d(x)':<10}")
    print("-" * 45)
    
    for age in range(max_age + 1):
        # Get mortality rate for this age
        mort_rate = mortality_rates.get(age, 0.0)
        
        # Print warning for very high mortality rates
        # if mort_rate > 1.0:
            # print(f"WARNING: Very high mortality rate at age {age}: {mort_rate:.4f}")
        
        # Store age and mortality rate
        ages.append(age)
        m_x.append(mort_rate)
        
        # Calculate probability of dying at this age
        prob_dying = 1 - np.exp(-mort_rate)
        
        # Check for invalid probability
        if prob_dying < 0 or prob_dying > 1:
            print(f"ERROR: Invalid probability of dying at age {age}: {prob_dying:.4f}, m(x)={mort_rate:.4f}")
            prob_dying = min(max(0, prob_dying), 1)
        
        q_x.append(prob_dying)
        
        # Store current survivorship value
        l_x.append(current_l_x)
        
        # Calculate deaths in this interval
        deaths = current_l_x * prob_dying
        d_x.append(deaths)
        
        # Update survivorship for next age
        next_l_x = current_l_x * np.exp(-mort_rate)
        
        # Calculate person-years lived in this interval
        # Using the midpoint approximation except for age 0
        if age == 0:
            # For age 0, often use a different factor (e.g., 0.3 instead of 0.5)
            # due to higher mortality in the first months of life
            L_x_value = next_l_x + 0.3 * deaths
        else:
            L_x_value = next_l_x + 0.5 * deaths
        
        L_x.append(L_x_value)
        
        # Print every 10 years and first/last year
        if age % 10 == 0 or age == 1 or age == max_age:
            print(f"{age:<5}{mort_rate:<10.4f}{prob_dying:<10.4f}{current_l_x:<10.1f}{deaths:<10.1f}")
        
        # Update current_l_x for next iteration
        current_l_x = next_l_x
    
    # Calculate T_x (backwards) and life expectancy
    T_x_value = 0
    for i in range(max_age, -1, -1):
        T_x_value += L_x[i]
        T_x.insert(0, T_x_value)  # Insert at beginning since we're going backwards
        
        # Calculate life expectancy
        e_x_value = T_x_value / l_x[i] if l_x[i] > 0 else 0
        e_x.insert(0, e_x_value)
        
        # Print life expectancy for key ages
        if i % 10 == 0 or i == 1:
            print(f"Life expectancy at age {i}: {e_x_value:.2f} years")
    
    # Create data frame with life table
    life_table = pd.DataFrame({
        'Age': ages,
        'l(x)': l_x,
        'd(x)': d_x,
        'q(x)': q_x,
        'm(x)': m_x,
        'L(x)': L_x,
        'T(x)': T_x,
        'e(x)': e_x
    })
    
    return life_table

class LifeExpectancyCalculator:
    """Class to calculate life expectancy from death tracker data"""
    
    def __init__(self, death_tracker=None):
        """
        Initialize the life expectancy calculator
        
        Args:
            death_tracker: DeathTracker instance to get mortality data from
        """
        self.death_tracker = death_tracker
    
    def calculate_from_death_tracker(self, year=None, by_sex=True):
        """
        Calculate life expectancy using data from the death tracker
        
        Args:
            year: Specific year to calculate for (None for all years)
            by_sex: Whether to calculate separately for males and females
            
        Returns:
            Dictionary with life expectancy values by sex and/or a DataFrame with the full life table
        """
        if self.death_tracker is None:
            raise ValueError("Death tracker not provided")
        
        print("\n====== CALCULATING LIFE EXPECTANCY ======")
        print(f"Using data for year: {'All years (cumulative)' if year is None else year}")
        
        # Get death counts from the death tracker
        if year is None:
            # Get cumulative death counts
            death_counts = self.death_tracker.get_death_counts()
        else:
            # Get death counts for specific year
            death_counts = self.death_tracker.get_death_counts(year)
        
        # Print summary of death counts
        total_male_deaths = sum(death_counts['Male'].values())
        total_female_deaths = sum(death_counts['Female'].values())
        print(f"Total deaths in data: Male={total_male_deaths}, Female={total_female_deaths}")
        
        # Get population counts by age and sex
        population = self._get_population_counts(year)
        
        # Print summary of population
        total_male_pop = sum(population['Male'].values())
        total_female_pop = sum(population['Female'].values())
        print(f"Total population: Male={total_male_pop}, Female={total_female_pop}")
        
        # Calculate mortality rates by age and sex
        mortality_rates = self._calculate_mortality_rates(death_counts, population)
        
        # Print average mortality rates
        avg_male_rate = sum(mortality_rates['Male'].values()) / len(mortality_rates['Male']) if mortality_rates['Male'] else 0
        avg_female_rate = sum(mortality_rates['Female'].values()) / len(mortality_rates['Female']) if mortality_rates['Female'] else 0
        print(f"Average mortality rates: Male={avg_male_rate:.4f}, Female={avg_female_rate:.4f}")
        
        # Calculate life tables
        life_tables = {}
        life_expectancy = {}
        
        if by_sex:
            # Calculate separately for males and females
            for sex in ['Male', 'Female']:
                print(f"\n==== Calculating Life Table for {sex} ====")
                life_tables[sex] = calculate_life_table(mortality_rates[sex], sex)
                # Life expectancy at birth (age 0)
                life_expectancy[sex] = life_tables[sex].loc[0, 'e(x)']
                print(f"{sex} life expectancy at birth: {life_expectancy[sex]:.2f} years")
                
            # Calculate combined
            combined_rates = {}
            for age in range(101):
                male_deaths = death_counts['Male'].get(age, 0)
                female_deaths = death_counts['Female'].get(age, 0)
                male_pop = population['Male'].get(age, 0)
                female_pop = population['Female'].get(age, 0)
                
                total_deaths = male_deaths + female_deaths
                total_pop = male_pop + female_pop
                
                if total_pop > 0:
                    combined_rates[age] = total_deaths / total_pop
                else:
                    combined_rates[age] = 0
            
            print("\n==== Calculating Combined Life Table ====")
            life_tables['Total'] = calculate_life_table(combined_rates, 'Total')
            life_expectancy['Total'] = life_tables['Total'].loc[0, 'e(x)']
            print(f"Combined life expectancy at birth: {life_expectancy['Total']:.2f} years")
        else:
            # Calculate only combined
            combined_rates = {}
            for age in range(101):
                male_deaths = death_counts['Male'].get(age, 0)
                female_deaths = death_counts['Female'].get(age, 0)
                male_pop = population['Male'].get(age, 0)
                female_pop = population['Female'].get(age, 0)
                
                total_deaths = male_deaths + female_deaths
                total_pop = male_pop + female_pop
                
                if total_pop > 0:
                    combined_rates[age] = total_deaths / total_pop
                else:
                    combined_rates[age] = 0
            
            life_tables['Total'] = calculate_life_table(combined_rates, 'Total')
            life_expectancy['Total'] = life_tables['Total'].loc[0, 'e(x)']
            print(f"Combined life expectancy at birth: {life_expectancy['Total']:.2f} years")
        
        # Print summary of life expectancy values
        print("\n==== Life Expectancy Summary ====")
        for sex, le in life_expectancy.items():
            print(f"{sex}: {le:.2f} years")
        print("=================================\n")
        
        return life_expectancy, life_tables
    
    def _get_population_counts(self, year=None):
        """
        Get population counts by age and sex from the simulation
        
        Args:
            year: Specific year to get population for
            
        Returns:
            Dictionary with population counts by