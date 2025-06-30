"""
Calculates and analyzes mortality rates and life expectancy from simulation data
"""


import numpy as np
import pandas as pd


def calculate_mortality_rates(sim, deaths_module, year=None, max_age=100, radix=100000):
    """
    Compute age-specific mortality rates (m(x)) using simulated death tracking and survivorship.

    Args:
        sim (ss.Sim): The simulation object.
        deaths_module: Module tracking male/female deaths by age (e.g., mi.DeathsByAgeSexAnalyzer).
        year (int, optional): Simulation year for labeling output. If None, uses current sim year.
        max_age (int): Maximum age to include in calculations.
        radix (int): Reference population size used for initial survivorship (typically 100000).

    Returns:
        pd.DataFrame: A table with columns ['year', 'age', 'sex', 'mx'].
    """
    survivorship = {'Male': np.zeros(max_age + 1), 'Female': np.zeros(max_age + 1)}
    survivorship['Male'][0] = sim.analyzers.survivorship_analyzer.survivorship_data['Male'][0]
    survivorship['Female'][0] = sim.analyzers.survivorship_analyzer.survivorship_data['Female'][0]

    deaths_by_age = {'Male': np.zeros(max_age + 1), 'Female': np.zeros(max_age + 1)}
    person_years = {'Male': np.zeros(max_age + 1), 'Female': np.zeros(max_age + 1)}

    for age in range(max_age + 1):
            deaths_by_age['Male'][age] = (
                deaths_module.results.male_deaths_by_age[age]
                    if age < len(deaths_module.results.male_deaths_by_age) else 0
            )

            deaths_by_age['Female'][age] = (
                deaths_module.results.female_deaths_by_age[age]
                    if age < len(deaths_module.results.female_deaths_by_age) else 0
            )
    
    mortality_rates = []
    for age in range(max_age):
        for sex in ['Male', 'Female']:
            # Compute d(x)
            deaths = deaths_by_age[sex][age]
            
            # Compute l(x+1) using l(x) and deaths
            survivorship[sex][age + 1] = sim.analyzers.survivorship_analyzer.survivorship_data[sex][age]
            
            # Compute L(x)
            Lx = survivorship[sex][age + 1] + 0.5 * deaths
            person_years[sex][age] = Lx

            # Compute m(x)
            mx = deaths / Lx if Lx > 0 else 0

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


def calculate_life_table_from_mx(m_x, max_age=100, radix=100000):
    """
    Calculate a life table from a vector of mortality rates m(x).

    Args:
        m_x: 1D numpy array of age-specific mortality rates for ages 0 to max_age.
        max_age: maximum age.
        radix: starting cohort size, typically 100000.

    Returns:
        pd.DataFrame with columns ['Age', 'l(x)', 'd(x)', 'q(x)', 'm(x)', 'L(x)', 'T(x)', 'e(x)']
    """
    ages = np.arange(max_age + 1)
    l_x = [radix]

    # Step 1: compute l(x)
    for age in range(max_age):
        l_next = l_x[-1] * np.exp(-m_x[age])
        l_x.append(l_next)
    l_x = np.array(l_x)

    # Step 2: compute d(x), q(x)
    d_x = l_x[:-1] - l_x[1:]
    d_x = np.append(d_x, l_x[-1])  # All remaining die at last age
    q_x = 1 - np.exp(-m_x)

    # Step 3: compute L(x)
    L_x = 0.5 * (l_x[:-1] + l_x[1:])
    L_x = np.append(L_x, l_x[-1] / m_x[-1] if m_x[-1] > 0 else 0)  # Terminal age open interval

    # Step 4: compute T(x), e(x)
    T_x = np.zeros_like(L_x)
    T_accum = 0
    for i in reversed(range(max_age + 1)):
        T_accum += L_x[i]
        T_x[i] = T_accum
    e_x = T_x / l_x

    df = pd.DataFrame({
        'Age': ages,
        'l(x)': l_x,
        'd(x)': d_x,
        'q(x)': q_x,
        'm(x)': m_x,
        'L(x)': L_x,
        'T(x)': T_x,
        'e(x)': e_x
    })
    return df


def calculate_life_table_from_lx_mx(l_x, m_x, max_age=100, radix=100000):
    """
    Calculate a life table from a vector of mortality rates m(x).

    Args:
        m_x: 1D numpy array of age-specific mortality rates for ages 0 to max_age.
        max_age: maximum age.
        radix: starting cohort size, typically 100000.

    Returns:
        pd.DataFrame with columns ['Age', 'l(x)', 'd(x)', 'q(x)', 'm(x)', 'L(x)', 'T(x)', 'e(x)']
    """
    ages = np.arange(max_age + 1)

    d_x = l_x[:-1] - l_x[1:]
    d_x = np.append(d_x, l_x[-1]) 
    
    q_x = 1 - np.exp(-m_x)

    L_x = np.zeros(max_age + 1)
    L_x[:max_age] = l_x[1:] + 0.5 * d_x[:max_age]
    L_x[max_age] = l_x[max_age] / m_x[max_age] if m_x[max_age] > 0 else 0

    T_x = np.zeros_like(L_x)
    T_accum = 0
    for i in reversed(range(max_age + 1)):
        T_accum += L_x[i]
        T_x[i] = T_accum
        
    with np.errstate(divide='ignore', invalid='ignore'):
        e_x = np.divide(T_x, l_x, out=np.zeros_like(T_x), where=l_x > 0)

    df = pd.DataFrame({
        'Age': ages,
        'l(x)': l_x,
        'd(x)': d_x,
        'q(x)': q_x,
        'm(x)': m_x,
        'L(x)': L_x,
        'T(x)': T_x,
        'e(x)': e_x
    })
    
    # Print a few key values
    for i in [0, 10, 30, 50, 70, 90, 99]:
        print(f"Age {i:>2}: l(x)={l_x[i]:,.0f}, m(x)={m_x[i]:.5f}, L(x)={L_x[i]:,.0f}, T(x)={T_x[i]:,.0f}")
    print(f"Sum of L(x): {np.sum(L_x):,.0f}")
    print(f"DEBUG: l(0) = {l_x[0]:,.0f}")
    print(f"DEBUG: L(0) = {L_x[0]:,.0f}")
    print(f"DEBUG: T(0) = {T_x[0]:,.0f}")
    print(f"DEBUG: e(0) = {T_x[0]/l_x[0]:.2f}")
    return df


def create_life_table(sim, df_mx_male, df_mx_female, max_age=100, radix = 100000):
    """
    Create life tables for both sexes using mortality (m_x) and survivorship (l_x) from simulation.

    Args:
        sim: Simulation object with `sim.analyzers.survivorship_analyzer.survivorship_data`.
        df_mx_male: DataFrame with 'age' and 'mx' for males.
        df_mx_female: DataFrame with 'age' and 'mx' for females.
        max_age: Maximum age to include in the life table.

    Returns:
        Concatenated DataFrame with life tables for males and females.
    """
    # Get survivorship arrays from sim (already aligned by age)
    # l_x_male = sim.analyzers.survivorship_analyzer.survivorship_data['Male'][:max_age + 1]
    # l_x_female = sim.analyzers.survivorship_analyzer.survivorship_data['Female'][:max_age + 1]
    
    l_x_male = sim.analyzers.survivorship_analyzer.survivorship_data['Male']
    if len(l_x_male) < max_age + 1:
        max_age = len(l_x_male) - 1
        print(f"Adjusted max_age to {max_age} based on l_x length")
    
    l_x_female = sim.analyzers.survivorship_analyzer.survivorship_data['Female']
    if len(l_x_female) < max_age + 1:
        max_age = len(l_x_female) - 1
        print(f"Adjusted max_age to {max_age} based on l_x length")
    
    l_x_male = l_x_male[:max_age + 1]
    l_x_female = l_x_female[:max_age + 1]
    
    # Normalize to radix
    radix = radix
    l_x_male = (l_x_male / l_x_male[0]) * radix
    l_x_female = (l_x_female / l_x_female[0]) * radix

    # Get mortality rates m(x) as array aligned by age
    m_x_male = df_mx_male.set_index('age').reindex(range(max_age + 1)).fillna(0)['mx'].values
    m_x_female = df_mx_female.set_index('age').reindex(range(max_age + 1)).fillna(0)['mx'].values

    # Use updated function that consumes l(x) and m(x)
    lt_male = calculate_life_table_from_lx_mx(l_x_male, m_x_male, max_age=max_age)
    lt_female = calculate_life_table_from_lx_mx(l_x_female, m_x_female, max_age=max_age)

    lt_male['sex'] = 'Male'
    lt_female['sex'] = 'Female'

    return pd.concat([lt_male, lt_female], ignore_index=True)
