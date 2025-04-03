import pandas as pd

class SimulationResults:
    def __init__(self):
        self.results = {
            'year': [],
            'births': [],
            'deaths': [],
            'population_0_4': [],
            'total_population': []
        }

    def add_result(self, year, births, deaths, population_0_4, total_population):
        self.results['year'].append(year)
        self.results['births'].append(births)
        self.results['deaths'].append(deaths)
        self.results['population_0_4'].append(population_0_4)
        self.results['total_population'].append(total_population)

    def to_dataframe(self):
        return pd.DataFrame(self.results)