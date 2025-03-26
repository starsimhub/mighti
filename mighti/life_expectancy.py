import numpy as np
import pandas as pd

def extract_mortality_rates(sim, cumulative_deaths_year):
    mortality_rates = {'male': {}, 'female': {}}
    ages = np.arange(0, 101)
    alive_counts = {'male': np.zeros(len(ages)), 'female': np.zeros(len(ages))}
    person_years = {'male': np.zeros(len(ages)), 'female': np.zeros(len(ages))}

    print("Extracting mortality rates...")
    
    # Calculate alive_counts for each age
    for age in ages:
        alive_counts['male'][age] = np.sum((sim.people.age >= age) & sim.people.male)
        alive_counts['female'][age] = np.sum((sim.people.age >= age) & sim.people.female)

    # Calculate person_years and mortality_rates for each age
    for age in ages:
        if age < 100:  # To avoid index error for the last age group
            deaths_male = min(cumulative_deaths_year['male'].get(age, 0), alive_counts['male'][age])
            deaths_female = min(cumulative_deaths_year['female'].get(age, 0), alive_counts['female'][age])
            next_alive_male = alive_counts['male'][age + 1]
            next_alive_female = alive_counts['female'][age + 1]

            person_years['male'][age] = next_alive_male + 0.5 * deaths_male
            person_years['female'][age] = next_alive_female + 0.5 * deaths_female

            mortality_rates['male'][age] = deaths_male / person_years['male'][age] if person_years['male'][age] > 0 else 0
            mortality_rates['female'][age] = deaths_female / person_years['female'][age] if person_years['female'][age] > 0 else 0
    
    print("Mortality rates (male):", mortality_rates['male'])
    print("Mortality rates (female):", mortality_rates['female'])
    
    return mortality_rates

def smooth_mortality_rates(mortality_rates, window_size=5):
    smoothed_mortality_rates = {'male': {}, 'female': {}}
    
    print("Smoothing mortality rates...")

    for gender in ['male', 'female']:
        mortality_series = pd.Series(mortality_rates[gender])
        smoothed_series = mortality_series.rolling(window=window_size, min_periods=1, center=True).mean()

        for age in range(101):
            smoothed_mortality_rates[gender][age] = smoothed_series.get(age, 0)

    print("Smoothed mortality rates (male):", smoothed_mortality_rates['male'])
    print("Smoothed mortality rates (female):", smoothed_mortality_rates['female'])

    return smoothed_mortality_rates

def calculate_life_table(mortality_rates, n_agents):
    ages = sorted(mortality_rates.keys())
    n = len(ages)
    
    l = np.zeros(n)
    d = np.zeros(n)
    q = np.zeros(n)
    m = np.zeros(n)
    L = np.zeros(n)
    T = np.zeros(n)
    e = np.zeros(n)

    l[0] = n_agents  # Starting with the number of agents

    print("Calculating life table...")
    
    for i in range(n):
        age = ages[i]
        m[i] = mortality_rates[age]
        
        if i < n - 1:
            l[i + 1] = l[i] * np.exp(-m[i])
            d[i] = l[i] - l[i + 1]
            q[i] = d[i] / l[i]
            L[i] = l[i + 1] + 0.5 * d[i]
        else:
            d[i] = l[i]
            q[i] = 1
            L[i] = l[i] / m[i] if m[i] > 0 else l[i] / 1e-10

        print(f"Age: {age}, l(x): {l[i]}, d(x): {d[i]}, q(x): {q[i]}, m(x): {m[i]}, L(x): {L[i]}")

    T[n - 1] = L[n - 1]
    for i in range(n - 2, -1, -1):
        T[i] = T[i + 1] + L[i]
        e[i] = T[i] / l[i]

        print(f"Age: {ages[i]}, T(x): {T[i]}, e(x): {e[i]}")

    life_table = pd.DataFrame({
        'Age': ages,
        'l(x)': l,
        'd(x)': d,
        'q(x)': q,
        'm(x)': m,
        'L(x)': L,
        'T(x)': T,
        'e(x)': e
    })

    print("Life table:")
    print(life_table)

    return life_table

def calculate_life_expectancy(sim, cumulative_deaths, n_agents):
    mortality_rates = extract_mortality_rates(sim, cumulative_deaths)
    mortality_rates = smooth_mortality_rates(mortality_rates)
    life_table_male = calculate_life_table(mortality_rates['male'], n_agents)
    life_table_female = calculate_life_table(mortality_rates['female'], n_agents)
    
    # Export life tables to CSV
    life_table_male.to_csv('life_table_male.csv', index=False)
    life_table_female.to_csv('life_table_female.csv', index=False)
    print("Life tables exported to life_table_male.csv and life_table_female.csv")

    return life_table_male[['Age', 'e(x)']], life_table_female[['Age', 'e(x)']]

