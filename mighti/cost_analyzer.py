import starsim as ss
import numpy as np

class CostAnalyzer(ss.Analyzer):
    def __init__(self, interventions=None):
        """
        Initialize the cost analyzer.
        :param interventions: Dict of interventions with cost and target populations.
        """
        super().__init__()
        self.interventions = interventions if interventions else {}
        self.costs = []
        self.time_points = []

    def initialize(self, sim):
        """
        Initialize the analyzer for simulation.
        :param sim: The simulation object.
        """
        self.costs = np.zeros(sim.npts)
        self.time_points = np.arange(sim.npts)
        return

    def apply(self, sim):
        """
        Apply cost calculations at each simulation step.
        :param sim: The simulation object.
        """
        step_cost = 0
        for intervention, data in self.interventions.items():
            coverage = data['coverage']  # Proportion of population covered
            unit_cost = data['cost']  # Cost per person
            affected_uids = sim.people.uid[data['target']]  # Target group
            step_cost += len(affected_uids) * unit_cost * coverage

        self.costs[sim.ti] = step_cost  # Store cost for this timestep
        return

    def finalize(self, sim):
        """
        Finalize the analyzer results after simulation.
        """
        print("Total Costs Over Time:", self.costs.sum())
        return