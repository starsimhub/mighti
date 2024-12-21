
import starsim as ss
import numpy as np

class CostAnalyzer(ss.Analyzer):
    def __init__(self, interventions=None):
        """
        Initialize the cost analyzer.
        
        Args:
            interventions (dict): Dictionary of interventions with cost and target populations.
        """
        super().__init__()
        self.interventions = interventions if interventions else {}
        self.costs = ss.Result(name='costs', dtype=float, scale=False, label='Intervention Costs')
        return

    def step(self):
        """
        Apply cost calculations at each simulation step.
        """
        sim = self.sim  # Access the simulation object from the analyzer
        step_cost = 0
        for intervention, data in self.interventions.items():
            coverage = data['coverage']  # Proportion of population covered
            unit_cost = data['cost']  # Cost per person
            target = data['target']  # Target group as a callable or a boolean array
            if callable(target):
                affected_uids = target(sim)
            else:
                affected_uids = sim.people.uid[target]
            step_cost += len(affected_uids) * unit_cost * coverage

        self.costs[sim.t.ti] = step_cost  # Store cost for this timestep
        return

    def finalize(self):
        """
        Finalize the analyzer results after simulation.
        """
        print("Total Costs Over Time:", self.costs.sum())
        return
