import numpy as np
import starsim as ss
import sciris as sc
import pandas as pd


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
            deaths_by_age['Male'][age] = (
                deaths_module.results.male_deaths_by_age[age]
                    if age < len(deaths_module.results.male_deaths_by_age) else 0
            )

            deaths_by_age['Female'][age] = (
                deaths_module.results.female_deaths_by_age[age]
                    if age < len(deaths_module.results.female_deaths_by_age) else 0
            )
    
    # Calculate survivorship, person-years, and mortality rates
    mortality_rates = []
    for age in range(max_age):
        for sex in ['Male', 'Female']:
            # Compute d(x)
            deaths = deaths_by_age[sex][age]
            
            # Compute l(x+1) using l(x) and deaths
            # survivorship[sex][age + 1] = survivorship[sex][age] - deaths
            survivorship[sex][age + 1] = sim.analyzers.survivorship_analyzer.survivorship_data[sex][age]
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
        Lx = sim.analyzers.survivorship_analyzer.survivorship_data[sex][age-1]
        mx = deaths / Lx if Lx > 0 else 0
        mortality_rates.append({'year': current_year, 'age': age, 'sex': sex, 'mx': mx})

    return pd.DataFrame(mortality_rates)


def calculate_life_table(df_mortality_rates, n_agents=100000, sex='Female', max_age=100):
    """
    Calculate a complete life table based on age-specific mortality rates.
    
    Args:
        mortality_rates: Dictionary with age as key and mortality rate (m_x) as value.
        n_agents: Number of agents in the cohort (starting population, default 100,000).
        sex: 'Male', 'Female', for labeling.
        max_age: Maximum age to consider (default 100).
        
    Returns:
        DataFrame containing the complete life table.
    """

    # Filter the DataFrame by sex
    df_filtered = df_mortality_rates[df_mortality_rates['sex'] == sex]

    # Convert to a dictionary with age as the key and mortality rate (m(x)) as the value
    age_to_mx = dict(zip(df_filtered['age'], df_filtered['mx']))

    # Fill in any missing ages with zero mortality rate
    for age in range(max_age + 1):
        if age not in age_to_mx:
            age_to_mx[age] = 0.0

    # Initialize lists to store life table columns
    ages = []
    l_x = []  # Survivorship function (number of persons alive at age x)
    d_x = []  # Number of deaths between age x and x+1
    q_x = []  # Probability of dying at age x
    m_x = []  # Mortality rate at age x
    L_x = []  # Person-years lived between age x and x+1
    T_x = []  # Person-years lived from age x until all members of the cohort have died
    e_x = []  # Life expectancy at age x

    # Set the radix of the life table (starting population)
    current_l_x = n_agents

    # Calculate life table columns
    for age in range(max_age + 1):
        # Get mortality rate for this age
        mort_rate = age_to_mx[age]

        # Store age and mortality rate
        ages.append(age)
        m_x.append(mort_rate)

        # Calculate probability of dying at this age
        prob_dying = 1 - np.exp(-mort_rate)
        q_x.append(prob_dying)

        # Store current survivorship value
        l_x.append(current_l_x)

        # Calculate deaths in this interval
        deaths = current_l_x * prob_dying
        d_x.append(deaths)

        # Calculate survivorship for next age
        next_l_x = current_l_x - deaths

        # Calculate person-years lived in this interval
        if age == 0:
            # For age 0, use a factor like 0.3 instead of 0.5
            L_x_value = 0.3 * current_l_x + 0.7 * next_l_x
        else:
            L_x_value = 0.5 * (current_l_x + next_l_x)
        L_x.append(L_x_value)

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



def create_life_table(df_mortality_rates, year,  n_agents, max_age=100):
    """
    Create a life table from the df_metrics DataFrame for a given year.
    
    Args:
        df_metrics: DataFrame with mortality rates
        year: Year to filter the data
        max_age: Maximum age to consider (default 100)
        
    Returns:
        DataFrame containing the complete life table for males and females
    """
    
    # Calculate life tables for males and females
    male_life_table = calculate_life_table(df_mortality_rates,  n_agents, sex='Male', max_age=max_age)
    female_life_table = calculate_life_table(df_mortality_rates,  n_agents, sex='Female', max_age=max_age)
    
    # Add columns for year and sex
    male_life_table['year'] = year
    male_life_table['sex'] = 'Male'
    female_life_table['year'] = year
    female_life_table['sex'] = 'Female'
    
    # Combine male and female life tables
    life_table = pd.concat([male_life_table, female_life_table], ignore_index=True)
    
    return life_table



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