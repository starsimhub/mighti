
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
        self.costs = None

    def init_pre(self, sim):
        """
        Initialize the analyzer for simulation.
        :param sim: The simulation object.
        """
        self.costs = np.zeros(sim.t.npts)
        return

    def step(self):
        """
        Apply cost calculations at each simulation step.
        :param sim: The simulation object.
        """
        sim = self.sim  # Access the simulation object from the analyzer
        step_cost = 0
        for intervention, data in self.interventions.items():
            coverage = data['coverage']  # Proportion of population covered
            unit_cost = data['cost']  # Cost per person
            affected_uids = data['target'](sim)  # Target group, assuming it's a callable
            step_cost += len(affected_uids) * unit_cost * coverage

        self.costs[sim.t.ti] = step_cost  # Store cost for this timestep
        return

    def finalize(self):
        """
        Finalize the analyzer results after simulation.
        """
        print("Total Costs Over Time:", self.costs.sum())
        return
