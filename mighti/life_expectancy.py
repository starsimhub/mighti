import numpy as np
import starsim as ss
import sciris as sc
import pandas as pd
import numpy as np
import pandas as pd

def calculate_mortality_rates(sim, deaths_module, year=None, max_age=100, radix=100000):
    """
    Calculate annual age-specific mortality rates for a given year.
    Aggregates deaths and person-years for each age and sex over all time steps in that year.
    Assumes deaths and survivorship analyzers store [age, timestep] arrays.
    """
    # Find which time steps correspond to the desired year
    yearvec = np.array(sim.t.yearvec)
    if year is None:
        # Use the latest year available
        year = int(yearvec[sim.t.ti])
    timestep_indices = np.where(yearvec == year)[0]
    if len(timestep_indices) == 0:
        raise ValueError(f"No simulation time steps found for year {year}")

    dt = float(sim.t.dt)  # timestep duration in years

    # Prepare arrays
    deaths_male = np.array(deaths_module.results.male_deaths_by_age) 
    deaths_female = np.array(deaths_module.results.female_deaths_by_age)
    surv_male = np.array(sim.analyzers.survivorship_analyzer.survivorship_data['Male'])  
    surv_female = np.array(sim.analyzers.survivorship_analyzer.survivorship_data['Female'])

    # Defensive shape checks
    n_ages = min(max_age + 1, deaths_male.shape[0], surv_male.shape[0])

    mortality_rates = []
    for sex, deaths, surv in [
        ('Male', deaths_male, surv_male),
        ('Female', deaths_female, surv_female)
    ]:
        for age in range(n_ages):
            
            # year_indices = np.where(sim.t.yearvec == year)[0]
            deaths_in_year = deaths[age, timestep_indices].sum()
            person_years = surv[age, timestep_indices].sum() * dt

            # deaths_in_year = deaths[age, timestep_indices].sum()
            # dt = float(getattr(sim.t, 'dt', 1.0))
            # person_years = surv[age, timestep_indices].sum() * dt
            
            mx = deaths_in_year / person_years if person_years > 0 else 0
            mortality_rates.append({'year': year, 'age': age, 'sex': sex, 'mx': mx})
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
        mort_rate = age_to_mx[age] /12

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
            L_x_value = 0.2 * current_l_x + 0.8 * next_l_x
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

