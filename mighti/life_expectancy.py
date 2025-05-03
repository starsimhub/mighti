import numpy as np
import starsim as ss
import sciris as sc
import pandas as pd

# def create_results_dataframe(sim, inityear, endyear, deaths_module):
#     """
#     Create a DataFrame with simulation results.

#     Args:
#         sim: The simulation object.
#         inityear: Initial year of the simulation.
#         endyear: End year of the simulation.
#         deaths_module: Instance of the Deaths class to access death tracking data.

#     Returns:
#         DataFrame with columns: ['year', 'age', 'sex', 'pop', 'deaths']
#     """
#     data = {
#         'year': [],
#         'age': [],
#         'sex': [],
#         'pop': [],
#         'deaths': []
#     }

#     years = list(range(inityear, endyear + 1))
#     ages = list(range(101))  # Single ages from 0 to 100

#     for year in years:
#         sim.people.age = sim.people.age.astype(float)  # Ensure age is float for comparison
#         for age in ages:
#             for sex in ['Male', 'Female']:
#                 if sex == 'Male':
#                     pop = np.sum((sim.people.age >= age) & (sim.people.male < age + 1))
#                     deaths = deaths_module.death_tracking['Male'][age] if age < len(deaths_module.death_tracking['Male']) else 0
#                 else:
#                     pop = np.sum((sim.people.age >= age) & (sim.people.female < age + 1))
#                     deaths = deaths_module.death_tracking['Female'][age] if age < len(deaths_module.death_tracking['Female']) else 0
                
#                 data['year'].append(year)
#                 data['age'].append(age)
#                 data['sex'].append(sex)
#                 data['pop'].append(pop)
#                 data['deaths'].append(deaths)

#     df = pd.DataFrame(data)
#     return df



# def create_results_dataframe_agegroup(sim, inityear, endyear, deaths_module, age_groups=None):
#     """
#     Create a DataFrame with simulation results.

#     Args:
#         sim: The simulation object.
#         inityear: Initial year of the simulation.
#         endyear: End year of the simulation.
#         deaths_module: Instance of the Deaths class to access death tracking data.
#         age_groups: List of tuples defining age groups. Each tuple should have
#                     the form (start_age, end_age, label). If None, default age groups will be used.

#     Returns:
#         DataFrame with columns: ['year', 'age_group', 'sex', 'pop', 'deaths']
#     """
#     if age_groups is None:
#         # Default age groups
#         age_groups = [
#             (0, 5, "0-4"),
#             (5, 15, "5-14"),
#             (15, 25, "15-24"),
#             (25, 35, "25-34"),
#             (35, 45, "35-44"),
#             (45, 55, "45-54"),
#             (55, 65, "55-64"),
#             (65, 75, "65-74"),
#             (75, 85, "75-84"),
#             (85, 101, "85+")
#         ]

#     data = {
#         'year': [],
#         'age_group': [],
#         'sex': [],
#         'pop': [],
#         'deaths': []
#     }

#     years = list(range(inityear, endyear + 1))

#     for year in years:
#         sim.people.age = sim.people.age.astype(float)  # Ensure age is float for comparison
#         for (start_age, end_age, label) in age_groups:
#             for sex in ['Male', 'Female']:
#                 if sex == 'Male':
#                     pop = np.sum((sim.people.age >= start_age) & (sim.people.age < end_age) & (sim.people.male))
#                     deaths = np.sum(deaths_module.death_tracking['Male'][start_age:end_age])
#                 else:
#                     pop = np.sum((sim.people.age >= start_age) & (sim.people.age < end_age) & (sim.people.female))
#                     deaths = np.sum(deaths_module.death_tracking['Female'][start_age:end_age])
                
#                 data['year'].append(year)
#                 data['age_group'].append(label)
#                 data['sex'].append(sex)
#                 data['pop'].append(pop)
#                 data['deaths'].append(deaths)

#     df = pd.DataFrame(data)
#     return df


# def calculate_metrics(df):
#     """
#     Calculate metrics from the results DataFrame.

#     Args:
#         df: DataFrame with columns: ['year', 'age_group', 'sex', 'pop', 'deaths']

#     Returns:
#         DataFrame with calculated mortality rates (mx).
#     """
#     mx_values = []
#     for index, row in df.iterrows():
#         if row['pop'] == 0:
#             mx = 0  # Avoid division by zero
#         else:
#             mx = row['deaths'] / (row['pop'] + 0.5 * row['deaths'])
#         mx_values.append(mx)

#     df['mx'] = pd.Series(mx_values)
#     return df

def calculate_mortality_rates(sim, deaths_module, year=None, max_age=100, radix=100000):

    # Initialize survivorship function l(x)
    survivorship = {'Male': np.zeros(max_age + 1), 'Female': np.zeros(max_age + 1)}
    survivorship['Male'][0] = radix/2
    survivorship['Female'][0] = radix/2

    # Initialize deaths and person-years lived by age and sex
    deaths_by_age = {'Male': np.zeros(max_age + 1), 'Female': np.zeros(max_age + 1)}
    person_years = {'Male': np.zeros(max_age + 1), 'Female': np.zeros(max_age + 1)}


    # Populate deaths by age and sex using deaths_module.death_tracking
    for age in range(max_age + 1):
        for sex in ['Male', 'Female']:
            if sex == 'Male':
                deaths_by_age[sex][age] = (
                    deaths_module.death_tracking['Male'][age] 
                    if age < len(deaths_module.death_tracking['Male']) else 0
                )
            else:
                deaths_by_age[sex][age] = (
                    deaths_module.death_tracking['Female'][age] 
                    if age < len(deaths_module.death_tracking['Female']) else 0
                )
    
    # Calculate survivorship, person-years, and mortality rates
    mortality_rates = []
    for age in range(max_age):
        for sex in ['Male', 'Female']:
            # Compute d(x)
            deaths = deaths_by_age[sex][age]
            
            # Compute l(x+1) using l(x) and deaths
            survivorship[sex][age + 1] = survivorship[sex][age] - deaths
            # Compute L(x)
            Lx = survivorship[sex][age + 1] + 0.5 * deaths
            person_years[sex][age] = Lx

            # Compute m(x)
            mx = deaths / Lx if Lx > 0 else 0
            print(f"age: {age}, sex: {sex}, deaths: {deaths}, Lx: {Lx} ")
            # Record mortality rate
            current_year = year if year is not None else int(sim.t.yearvec[sim.t.ti])
            mortality_rates.append({'year': current_year, 'age': age, 'sex': sex, 'mx': mx})
            


    # Handle the last age group (open interval)
    for sex in ['Male', 'Female']:
        age = max_age
        deaths = deaths_by_age[sex][age]
        Lx = person_years[sex][age - 1]
        mx = deaths / Lx if Lx > 0 else 0
        mortality_rates.append({'year': current_year, 'age': age, 'sex': sex, 'mx': mx})

    return pd.DataFrame(mortality_rates)


def calculate_life_table(mortality_rates, n_agents = 100000, sex='Total', max_age=100):
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
    radix = n_agents
    
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
        if isinstance(mort_rate, pd.Series):
            if mort_rate.any() > 1.0:
                print(f"WARNING: Very high mortality rate at age {age}: {mort_rate.max():.4f}")
        elif mort_rate > 1.0:
            print(f"WARNING: Very high mortality rate at age {age}: {mort_rate:.4f}")
        
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
            L_x_value = next_l_x + 0.5 * deaths
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


def extract_mortality_rates(df_metrics, year, sex):
    """
    Extract mortality rates from the df_metrics DataFrame for a given year and sex.
    
    Args:
        df_metrics: DataFrame with mortality rates
        year: Year to filter the data
        sex: 'Male' or 'Female' to filter the data
        
    Returns:
        Dictionary with age as key and mortality rate (m_x) as value
    """
    # Filter the DataFrame for the given year and sex
    filtered_df = df_metrics[(df_metrics['year'] == year) & (df_metrics['sex'] == sex)]
    
    # Create a dictionary with age as key and mortality rate (m_x) as value
    mortality_rates = filtered_df.set_index('age')['mx'].to_dict()
    
    return mortality_rates

def create_life_table(df_metrics, year,  n_agents, max_age=100):
    """
    Create a life table from the df_metrics DataFrame for a given year.
    
    Args:
        df_metrics: DataFrame with mortality rates
        year: Year to filter the data
        max_age: Maximum age to consider (default 100)
        
    Returns:
        DataFrame containing the complete life table for males and females
    """
    # Extract mortality rates for males and females
    male_mortality_rates = extract_mortality_rates(df_metrics, year, 'Male')
    female_mortality_rates = extract_mortality_rates(df_metrics, year, 'Female')
    
    # Calculate life tables for males and females
    male_life_table = calculate_life_table(male_mortality_rates,  n_agents, sex='Male', max_age=max_age)
    female_life_table = calculate_life_table(female_mortality_rates,  n_agents, sex='Female', max_age=max_age)
    
    # Add columns for year and sex
    male_life_table['year'] = year
    male_life_table['sex'] = 'Male'
    female_life_table['year'] = year
    female_life_table['sex'] = 'Female'
    
    # Combine male and female life tables
    life_table = pd.concat([male_life_table, female_life_table], ignore_index=True)
    
    return life_table

# def extract_life_expectancy(life_table, sex):
#     e0 = life_table.loc[(life_table['Age'] == 0) & (life_table['sex'] == sex), 'e(x)'].values[0]
#     return e0

# def print_life_expectancy_statement(life_table):
#     male_e0 = extract_life_expectancy(life_table, 'Male')
#     female_e0 = extract_life_expectancy(life_table, 'Female')
#     general_e0 = (male_e0 + female_e0) / 2  # Assuming equal population for simplicity
    
#     statement = (f"The model predicted a life expectancy of {general_e0:.2f} years "
#                   f"({female_e0:.2f} years for females, {male_e0:.2f} years for males).")
#     print(statement)


# # # import numpy as np
# # # import pandas as pd


# # # def calculate_mortality_rates(sim, year=None, max_age=100):
# # #     """
# # #     Calculate mortality rates from death counts and population for each year of age.
# # #     """
# # #     death_counts = sim.results['new']
# # #     if year is not None:
# # #         death_counts = death_counts[sim.t.yearvec == year]

# # #     ages = sim.people.age
# # #     females = sim.people.female
# # #     alive = sim.people.alive

# # #     population = {'Male': np.zeros(max_age + 1), 'Female': np.zeros(max_age + 1)}
# # #     alive_indices = np.where(alive)[0]

# # #     for idx in alive_indices:
# # #         age = int(ages[idx])
# # #         if age > max_age:
# # #             continue
# # #         sex = 'Female' if females[idx] else 'Male'
# # #         population[sex][age] += 1

# # #     deaths_by_age = {'Male': np.zeros(max_age + 1), 'Female': np.zeros(max_age + 1)}
# # #     dead_indices = np.where(~alive)[0]

# # #     for idx in dead_indices:
# # #         age = int(ages[idx])
# # #         if age > max_age:
# # #             continue
# # #         sex = 'Female' if females[idx] else 'Male'
# # #         deaths_by_age[sex][age] += 1

# # #     mortality_rates = []
# # #     for age in range(max_age + 1):
# # #         for sex in ['Male', 'Female']:
# # #             pop = population[sex][age]
# # #             deaths = deaths_by_age[sex][age]
# # #             rate = deaths / pop if pop > 0 else 0
# # #             current_year = year if year is not None else int(sim.t.yearvec[sim.t.ti])
# # #             mortality_rates.append({'Time': current_year, 'Age': age, 'Sex': sex, 'mx': rate})

# # #     return pd.DataFrame(mortality_rates)


# # # def extract_mortality_rates(df_metrics, year, sex):
# # #     """
# # #     Extract mortality rates from the df_metrics DataFrame for a given year and sex.
# # #     """
# # #     filtered_df = df_metrics[(df_metrics['Time'] == year) & (df_metrics['Sex'] == sex)]
# # #     return filtered_df.set_index('Age')['mx'].to_dict()


# # # def calculate_life_table(mortality_rates, n_agents=100000, sex='Total', max_age=100):
# # #     ages, l_x, d_x, q_x, m_x, L_x, T_x, e_x = [], [], [], [], [], [], [], []
# # #     radix = n_agents

# # #     for age in range(max_age + 1):
# # #         if age not in mortality_rates:
# # #             mortality_rates[age] = 0.0

# # #     current_l_x = radix

# # #     for age in range(max_age + 1):
# # #         mort_rate = mortality_rates.get(age, 0.0)
# # #         prob_dying = 1 - np.exp(-mort_rate)
# # #         prob_dying = min(max(0, prob_dying), 1)

# # #         ages.append(age)
# # #         m_x.append(mort_rate)
# # #         q_x.append(prob_dying)
# # #         l_x.append(current_l_x)

# # #         deaths = current_l_x * prob_dying
# # #         d_x.append(deaths)
# # #         next_l_x = current_l_x * np.exp(-mort_rate)
# # #         L_x_value = next_l_x + 0.5 * deaths
# # #         L_x.append(L_x_value)
# # #         current_l_x = next_l_x

# # #     T_x_value = 0
# # #     for i in range(max_age, -1, -1):
# # #         T_x_value += L_x[i]
# # #         T_x.insert(0, T_x_value)
# # #         e_x.insert(0, T_x_value / l_x[i] if l_x[i] > 0 else 0)

# # #     life_table = pd.DataFrame({
# # #         'Age': ages,
# # #         'l(x)': l_x,
# # #         'd(x)': d_x,
# # #         'q(x)': q_x,
# # #         'm(x)': m_x,
# # #         'L(x)': L_x,
# # #         'T(x)': T_x,
# # #         'e(x)': e_x
# # #     })

# # #     return life_table


# # # def create_life_table(df_metrics, year, n_agents, max_age=100):
# # #     male_mx = extract_mortality_rates(df_metrics, year, 'Male')
# # #     female_mx = extract_mortality_rates(df_metrics, year, 'Female')

# # #     male_life = calculate_life_table(male_mx, n_agents, sex='Male', max_age=max_age)
# # #     female_life = calculate_life_table(female_mx, n_agents, sex='Female', max_age=max_age)

# # #     male_life['year'] = year
# # #     male_life['sex'] = 'Male'
# # #     female_life['year'] = year
# # #     female_life['sex'] = 'Female'

# # #     return pd.concat([male_life, female_life], ignore_index=True)


# # # def extract_life_expectancy(life_table, sex):
# # #     return life_table.loc[(life_table['Age'] == 0) & (life_table['sex'] == sex), 'e(x)'].values[0]


# # # def print_life_expectancy_statement(life_table):
# # #     male_e0 = extract_life_expectancy(life_table, 'Male')
# # #     female_e0 = extract_life_expectancy(life_table, 'Female')
# # #     general_e0 = (male_e0 + female_e0) / 2

# # #     print(f"The model predicted a life expectancy of {general_e0:.2f} years "
# # #           f"({female_e0:.2f} years for females, {male_e0:.2f} years for males).")

