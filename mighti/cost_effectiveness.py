import numpy as np

class CostEffectivenessAnalyzer:
    def __init__(self, interventions, qols):
        """
        Initialize the cost-effectiveness analyzer.
        :param interventions: Dict containing cost, coverage, and QOL improvement for each intervention.
        :param qols: Dict containing baseline QOL for each disease.
        """
        self.interventions = interventions
        self.qols = qols
        self.results = {
            "costs": {"total_costs": 0},
            "utilities": {"total_qalys": 0},
            "icer": {},
            "increments": {}
        }

    def calculate_costs_and_utilities(self, sim):
        """
        Calculate total costs and utilities for each intervention.
        :param sim: The simulation object containing disease and intervention data.
        """
        total_costs = 0
        total_qalys = 0
        costs = {}
        utilities = {}

        for disease, data in self.interventions.items():
            # Extract intervention parameters
            cost_per_person = data["cost"]
            coverage = data["coverage"]
            qaly_improvement = data["qaly_improvement"]  # QALY improvement due to the intervention
            baseline_qol = self.qols[disease]  # Baseline QOL for the condition

            # Access the disease object
            disease_obj = getattr(sim.diseases, disease.lower())
            if hasattr(disease_obj, "affected"):
                state_array = disease_obj.affected
            elif hasattr(disease_obj, "infected"):
                state_array = disease_obj.infected
            else:
                raise AttributeError(f"No valid state found for disease: {disease}")

            # Calculate prevalence, costs, and QALYs
            prevalence = state_array.sum() / len(sim.people)  # Proportion of population affected
            num_affected = len(state_array.uids)  # Total number of affected individuals

            # Calculate intervention cost and QALY gain
            intervention_cost = cost_per_person * coverage * num_affected
            qaly_gain = prevalence * (baseline_qol + qaly_improvement)  # Adjusted QALYs due to intervention

            # Store individual intervention results
            costs[disease] = intervention_cost
            utilities[disease] = qaly_gain

            # Update totals
            total_costs += intervention_cost
            total_qalys += qaly_gain

        # Save results
        self.results["costs"] = {"total_costs": total_costs, **costs}
        self.results["utilities"] = {"total_qalys": total_qalys, **utilities}
        return total_costs, total_qalys

    def calculate_increments(self, baseline_results):
        """
        Calculate incremental costs and QALYs relative to the baseline results.
        :param baseline_results: Dictionary of results from the baseline scenario.
        """
        baseline_cost = baseline_results["costs"]["total_costs"]
        baseline_qaly = baseline_results["utilities"]["total_qalys"]

        scenario_cost = self.results["costs"]["total_costs"]
        scenario_qaly = self.results["utilities"]["total_qalys"]

        # Calculate increments
        incremental_cost = scenario_cost - baseline_cost
        incremental_qaly = scenario_qaly - baseline_qaly

        # Store increments and ICER
        self.results["increments"] = {
            "incremental_cost": incremental_cost,
            "incremental_qaly": incremental_qaly,
        }
        self.results["icer"] = incremental_cost / incremental_qaly if incremental_qaly != 0 else np.inf
        return self.results["increments"]

    def summarize_results(self):
        """
        Summarize and print the results for cost-effectiveness analysis.
        """
        print("Costs:", self.results["costs"])
        print("Utilities (QALYs):", self.results["utilities"])
        print("ICER:", self.results["icer"])
        print("Increments:", self.results["increments"])
        return self.results