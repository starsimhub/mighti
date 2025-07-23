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


def calculate_life_table_from_mx(sim, df_mx_male, df_mx_female, max_age=100):
    """
    Compute life tables for males and females using m(x) and simulated l(0) from survivorship analyzer.

    Args:
        sim: Simulation object with a survivorship analyzer.
        df_mx_male, df_mx_female: DataFrames with columns ['age', 'mx'].
        max_age: Maximum age to compute.

    Returns:
        pd.DataFrame with columns ['sex', 'Age', 'l(x)', 'd(x)', 'q(x)', 'm(x)', 'L(x)', 'T(x)', 'e(x)']
    """
    def compute_life_table(sex, l0, m_x):
        l_x = [l0]
        for age in range(max_age):
            l_next = l_x[-1] * np.exp(-m_x[age])
            l_x.append(l_next)
        l_x = np.array(l_x)

        d_x = l_x[:-1] - l_x[1:]
        d_x = np.append(d_x, l_x[-1])  # all die at terminal age

        q_x = 1 - np.exp(-m_x)

        L_x = 0.5 * (l_x[:-1] + l_x[1:])
        L_x = np.append(L_x, l_x[-1] / m_x[-1] if m_x[-1] > 0 else 0)

        T_x = np.zeros_like(L_x)
        T_accum = 0
        for i in reversed(range(max_age + 1)):
            T_accum += L_x[i]
            T_x[i] = T_accum

        e_x = T_x / l_x

        return pd.DataFrame({
            'sex': sex,
            'Age': np.arange(max_age + 1),
            'l(x)': l_x,
            'd(x)': d_x,
            'q(x)': q_x,
            'm(x)': m_x,
            'L(x)': L_x,
            'T(x)': T_x,
            'e(x)': e_x
        })

    # Extract initial survivorship
    l0_male = sim.analyzers.survivorship_analyzer.survivorship_data['Male'][0]
    l0_female = sim.analyzers.survivorship_analyzer.survivorship_data['Female'][0]

    # Align and extract m(x)
    m_x_male = df_mx_male.set_index('age').reindex(range(max_age + 1)).fillna(0)['mx'].values
    m_x_female = df_mx_female.set_index('age').reindex(range(max_age + 1)).fillna(0)['mx'].values

    # Compute life tables
    lt_male = compute_life_table('Male', l0_male, m_x_male)
    lt_female = compute_life_table('Female', l0_female, m_x_female)

    return pd.concat([lt_male, lt_female], ignore_index=True)

