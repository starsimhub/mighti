import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_mortality_rates(sim, cumulative_deaths):
    mortality_rates = {}
    ages = np.arange(0, 101)
    alive_counts = np.zeros(len(ages))

    for age in ages:
        alive_counts[age] = np.sum(sim.people.age >= age)
        mortality_rates[age] = cumulative_deaths.get(age, 0) / alive_counts[age] if alive_counts[age] > 0 else 0

    return mortality_rates

def calculate_life_table(mortality_rates):
    ages = sorted(mortality_rates.keys())
    n = len(ages)
    
    l = np.zeros(n)
    d = np.zeros(n)
    q = np.zeros(n)
    m = np.zeros(n)
    L = np.zeros(n)
    T = np.zeros(n)
    e = np.zeros(n)

    l[0] = 100000

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

    T[n - 1] = L[n - 1]
    for i in range(n - 2, -1, -1):
        T[i] = T[i + 1] + L[i]
        e[i] = T[i] / l[i]

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

    return life_table

def calculate_life_expectancy(sim, cumulative_deaths):
    mortality_rates = extract_mortality_rates(sim, cumulative_deaths)
    life_table = calculate_life_table(mortality_rates)
    
    # Print intermediate results for debugging
    print("Mortality Rates:")
    for age, rate in mortality_rates.items():
        print(f"Age {age}: {rate}")
    
    print("Life Table:")
    print(life_table[['Age', 'l(x)', 'd(x)', 'q(x)', 'm(x)', 'L(x)', 'T(x)', 'e(x)']])

    return life_table[['Age', 'e(x)']]

def compare_life_expectancy(predicted, actual):
    predicted_life_expectancy = np.array([predicted['e(x)'].iloc[0], predicted['e(x)'].iloc[0], predicted['e(x)'].iloc[0]])
    actual_life_expectancy_values = np.array(list(actual.values()))
    sse = np.sum((predicted_life_expectancy - actual_life_expectancy_values) ** 2)
    return sse

def plot_survival_curves(predicted, actual):
    plt.figure(figsize=(12, 8))
    plt.plot(predicted['Age'], predicted['e(x)'], label='Predicted Life Expectancy')
    plt.axhline(y=actual['men'], color='b', linestyle='--', label='Actual Men')
    plt.axhline(y=actual['women'], color='r', linestyle='--', label='Actual Women')
    plt.axhline(y=actual['all'], color='g', linestyle='--', label='Actual All')
    plt.xlabel('Age')
    plt.ylabel('Life Expectancy')
    plt.title('Survival Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()