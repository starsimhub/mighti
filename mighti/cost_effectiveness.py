import numpy as np

class CostEffectivenessAnalyzer:
    def __init__(self, interventions, utilities):
        """
        Initialize the cost-effectiveness analyzer.
        :param interventions: Dict containing cost and coverage for each intervention.
        :param utilities: Dict containing utility weights (e.g., QALY, DALY).
        """
        self.interventions = interventions
        self.utilities = utilities
        self.results = {
            "total_costs": {},
            "health_outcomes": {},
            "icer": {}
        }

    def calculate_costs(self, sim):
        """
        Calculate total intervention costs during the simulation.
        :param sim: The simulation object containing time-series data.
        """
        total_cost = 0
        for intervention, data in self.interventions.items():
            coverage = data['coverage']  # Proportion of population covered
            unit_cost = data['cost']  # Cost per person
            target_condition = data['target']  # E.g., 'hiv_infected' or 'type2diabetes_affected'
    
            # Split the target into condition and state (e.g., 'hiv_infected' -> 'hiv', 'infected')
            condition, state = target_condition.split('_')
            
            # Access the disease object and the corresponding state
            if hasattr(sim.diseases, condition):
                disease_obj = getattr(sim.diseases, condition)
                if hasattr(disease_obj, state):
                    state_uids = getattr(disease_obj, state).uids  # Get UIDs of the target state
                    step_cost = len(state_uids) * unit_cost * coverage
                    total_cost += step_cost
                else:
                    raise AttributeError(f"State '{state}' not found for disease '{condition}'.")
            else:
                raise AttributeError(f"Disease '{condition}' not found in the simulation.")
            
        self.results["total_costs"] = total_cost
        return total_cost

    def calculate_outcomes(self, sim):
        """
        Calculate the health outcomes (e.g., QALYs) for each condition.
        :param sim: The simulation object containing time-series data.
        """
        total_qalys = 0
        health_outcomes = {}  # Dictionary to store QALYs for each condition
        
        for condition, utility_data in self.utilities.items():
            # Dynamically determine whether to use 'infected' or 'affected'
            disease_obj = getattr(sim.diseases, condition.lower())
            if hasattr(disease_obj, 'affected'):
                state_array = disease_obj.affected  # For NCDs
            elif hasattr(disease_obj, 'infected'):
                state_array = disease_obj.infected  # For infectious diseases
            else:
                raise AttributeError(f"Neither 'affected' nor 'infected' found for disease '{condition}'.")
            
            # Calculate prevalence and QALYs
            prevalence = state_array.sum() / len(sim.people)
            qaly = utility_data['qaly'] * prevalence
            
            total_qalys += qaly
            health_outcomes[condition] = qaly  # Store QALY for each condition
        
        # Save results
        self.results["health_outcomes"] = {"total_qalys": total_qalys, **health_outcomes}
        return total_qalys

    def calculate_icer(self, baseline_cost, baseline_qaly):
        """
        Calculate Incremental Cost-Effectiveness Ratio (ICER).
        :param baseline_cost: Cost of baseline intervention.
        :param baseline_qaly: QALY of baseline intervention.
        """
        incremental_cost = self.results["total_costs"] - baseline_cost
        incremental_qaly = self.results["health_outcomes"]["total_qalys"] - baseline_qaly
        self.results["icer"] = incremental_cost / incremental_qaly if incremental_qaly != 0 else np.inf
        return self.results["icer"]

    def summarize_results(self):
        """
        Summarize and print the results for cost-effectiveness analysis.
        """
        print("Total Costs:", self.results["total_costs"])
        print("Health Outcomes (QALY/DALY):", self.results["health_outcomes"])
        print("ICER:", self.results["icer"])
        return self.results