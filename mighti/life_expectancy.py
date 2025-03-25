import numpy as np
import pandas as pd
import starsim as ss
import matplotlib.pyplot as plt
from collections import defaultdict


class CustomDeaths(ss.Deaths):
    def __init__(self, pars=None, metadata=None, **kwargs):
        super().__init__(pars, metadata, **kwargs)
        self.cumulative_deaths = 0

    def step(self):
        """ Select people to die and track cumulative deaths """
        death_uids = self.pars.death_rate.filter()
        self.sim.people.request_death(death_uids)
        self.n_deaths = len(death_uids)
        self.cumulative_deaths += self.n_deaths
        return self.n_deaths

    def get_cumulative_deaths(self):
        return self.cumulative_deaths

    def finalize(self):
        super().finalize()
        self.results.cumulative[:] = np.cumsum(self.results.new)
        return
    
class CustomPeople(ss.People):
    def __init__(self, n_agents, age_data):
        super().__init__(n_agents, age_data)  # Initialize the parent class

    def calculate_life_expectancy(self, cumulative_deaths):
        """
        Calculate life expectancy based on cumulative deaths.
        """
        age_groups = np.arange(0, 101)
        l_x = np.zeros_like(age_groups, dtype=float)  # Number of people alive at the beginning of each age group
        d_x = np.zeros_like(age_groups, dtype=float)  # Number of deaths in each age group

        for age in age_groups:
            l_x[age] = np.sum(self.age >= age)
            d_x[age] = cumulative_deaths.get(age, 0)

        L_x = (l_x[:-1] + l_x[1:]) / 2  # Total number of person-years lived within the age interval
        T_x = np.cumsum(L_x[::-1])[::-1]  # Total number of person-years lived by the cohort after age x
        e_x = T_x / l_x[:-1]  # Life expectancy at each age

        life_table = pd.DataFrame({
            'Age': age_groups[:-1],
            'l(x)': l_x[:-1],
            'd(x)': d_x[:-1],
            'L(x)': L_x,
            'T(x)': T_x,
            'e(x)': e_x
        })

        return life_table

# Helper functions
def calculate_life_table(mortality_rates):
    ages = sorted(mortality_rates.keys())
    n = len(ages)
    
    # Initialize life table columns
    l = np.zeros(n)
    d = np.zeros(n)
    q = np.zeros(n)
    m = np.zeros(n)
    L = np.zeros(n)
    T = np.zeros(n)
    e = np.zeros(n)

    # Set initial population (radix)
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
            if m[i] > 0:
                L[i] = l[i] / m[i]
            else:
                L[i] = l[i] / 1e-10  # Avoid division by zero

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

def extract_mortality_rates(sim, prevalence_analyzer):
    mortality_rates = {}
    ages = np.arange(0, 101)  # Assuming age range from 0 to 100

    # Initialize data structures to store life and death counts at each time step
    alive_counts = defaultdict(lambda: np.zeros(len(ages)))
    dead_counts = defaultdict(lambda: np.zeros(len(ages)))

    # Determine the number of time steps using timevec
    num_time_steps = len(sim.timevec)

    # Track life and death counts for each time step
    for t in range(num_time_steps):
        for age in ages:
            age_mask = (sim.people.age >= age) & (sim.people.age < age + 1)
            alive_counts[t][age] = np.sum(age_mask)
            dead_counts[t][age] = sum(1 for uid, (death_age, death_year) in prevalence_analyzer.cumulative_deaths.items() if death_age == age and death_year == t)

    # Calculate mortality rates for each age
    for age in ages:
        num_alive = np.sum([alive_counts[t][age] for t in range(num_time_steps)])
        num_dead = np.sum([dead_counts[t][age] for t in range(num_time_steps)])
        if num_alive > 0:
            mortality_rate = num_dead / num_alive
        else:
            mortality_rate = 0
        mortality_rates[age] = mortality_rate

        # Print detailed debugging information
        print(f"Age {age}: num_alive = {num_alive}, num_dead = {num_dead}, mortality_rate = {mortality_rate}")

    return mortality_rates

def calculate_life_expectancy(sim, prevalence_analyzer):
    mortality_rates = extract_mortality_rates(sim, prevalence_analyzer)
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