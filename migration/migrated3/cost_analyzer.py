
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
        self.costs = None
        self.time_points = None

    def init_pre(self, sim):
        """
        Initialize the analyzer for the simulation.
        
        Args:
            sim (Sim): The simulation object.
        """
        self.costs = np.zeros(sim.t.npts)
        self.time_points = sim.t.tvec.copy()
        return

    def step(self):
        """
        Apply cost calculations at each simulation step.
        """
        step_cost = 0
        for intervention, data in self.interventions.items():
            coverage = data['coverage']  # Proportion of population covered
            unit_cost = data['cost']  # Cost per person
            affected_uids = data['target'](self.sim)  # Target group, assuming it's a callable returning UIDs
            step_cost += len(affected_uids) * unit_cost * coverage

        self.costs[self.sim.t.ti] = step_cost  # Store cost for this timestep
        return

    def finalize(self):
        """
        Finalize the analyzer results after the simulation.
        """
        total_cost = self.costs.sum()
        print(f"Total Costs Over Time: {total_cost}")
        return
