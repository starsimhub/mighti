import matplotlib.pyplot as plt
import numpy as np

def plot_cost_vs_utility(results):
    """
    Plot Cost vs Utility (QALY) for each intervention.
    :param results: Dictionary containing total costs and utilities (QALYs) for each intervention.
    """
    interventions = list(results["utilities"].keys())
    total_costs = [results["costs"].get(intv, 0) for intv in interventions]
    total_qalys = [results["utilities"].get(intv, 0) for intv in interventions]

    plt.figure(figsize=(10, 6))
    plt.scatter(total_qalys, total_costs, color='blue', s=100, label='Interventions')
    plt.axhline(y=50000, color='r', linestyle='--', label='Cost-effectiveness threshold ($50,000/QALY)')

    for i, intv in enumerate(interventions):
        plt.text(total_qalys[i], total_costs[i], intv, fontsize=10, ha='right')

    plt.title('Cost vs. Utility (QALY)', fontsize=16)
    plt.xlabel('Total QALYs', fontsize=14)
    plt.ylabel('Total Costs ($)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()


def plot_icer_bars(scenario_results, baseline_results):
    """
    Plot ICER bar chart for each intervention relative to the baseline.
    :param scenario_results: Dictionary of scenario results.
    :param baseline_results: Baseline results for comparison.
    """
    interventions = list(scenario_results.keys())
    baseline_cost = baseline_results["costs"]["total_costs"]
    baseline_qaly = baseline_results["utilities"]["total_qalys"]

    incremental_costs = [scenario_results[intv]["costs"]["total_costs"] - baseline_cost for intv in interventions]
    incremental_qalys = [scenario_results[intv]["utilities"]["total_qalys"] - baseline_qaly for intv in interventions]
    icers = [cost / qaly if qaly > 0 else np.inf for cost, qaly in zip(incremental_costs, incremental_qalys)]

    plt.figure(figsize=(10, 6))
    plt.bar(interventions, icers, color='blue')
    plt.axhline(y=50000, color='r', linestyle='--', label='Cost-effectiveness threshold ($50,000/QALY)')

    plt.title('Incremental Cost-Effectiveness Ratios (ICER)', fontsize=16)
    plt.xlabel('Interventions', fontsize=14)
    plt.ylabel('ICER ($/QALY)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_cost_breakdown(results):
    """
    Plot cost breakdown for each intervention.
    :param results: Dictionary containing cost components for each intervention.
    """
    interventions = list(results["costs"].keys())
    total_costs = results["costs"]["total_costs"]

    # Extract cost data for each intervention
    cost_data = [results["costs"].get(intv, 0) for intv in interventions]

    plt.figure(figsize=(10, 6))
    plt.bar(interventions, cost_data, color="blue", alpha=0.7, label="Costs")
    plt.title('Cost Breakdown by Intervention', fontsize=16)
    plt.xlabel('Interventions', fontsize=14)
    plt.ylabel('Total Costs ($)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_cost_qaly_bar(scenario_results):
    """
    Plot bar charts for incremental costs and QALYs for each scenario.
    :param scenario_results: Dictionary of scenario results from CEA.
    """
    diseases = list(scenario_results.keys())
    incremental_costs = [scenario_results[d]["increments"]["incremental_cost"] for d in diseases]
    incremental_qalys = [scenario_results[d]["increments"]["incremental_qaly"] for d in diseases]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot for costs
    ax1.bar(diseases, incremental_costs, color="blue", alpha=0.7, label="Incremental Cost ($)")
    ax1.set_xlabel("Disease Added to HBP", fontsize=14)
    ax1.set_ylabel("Incremental Cost ($)", fontsize=14, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Secondary axis for QALYs
    ax2 = ax1.twinx()
    ax2.plot(diseases, incremental_qalys, color="green", marker="o", label="Incremental QALYs")
    ax2.set_ylabel("Incremental QALYs", fontsize=14, color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    plt.title("Incremental Costs and QALYs by Disease", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_icer_scatter(scenario_results, baseline_results):
    """
    Plot ICER scatter plot comparing interventions relative to a baseline.
    :param scenario_results: Results of the scenario simulations.
    :param baseline_results: Baseline results for comparison.
    """
    # Extract interventions from scenario results
    interventions = list(scenario_results.keys())
    
    # Initialize lists to store values
    costs = []
    qalys = []
    labels = []
    icers = []

    # Extract baseline costs and QALYs
    baseline_cost = baseline_results["costs"].get("total_costs", None)
    baseline_qaly = baseline_results["utilities"].get("total_qalys", None)

    # Debugging: Print baseline values
    print(f"Baseline Cost: {baseline_cost}")
    print(f"Baseline QALY: {baseline_qaly}")

    # Ensure baseline values are valid
    if baseline_cost is None or baseline_qaly is None:
        raise ValueError("Baseline results are missing 'total_costs' or 'total_qalys'. Check the input data.")

    # Process each intervention
    for intervention in interventions:
        scenario_cost = scenario_results[intervention]["costs"]["total_costs"]
        scenario_qaly = scenario_results[intervention]["utilities"]["total_qalys"]

        # Calculate incremental costs and QALYs
        incremental_cost = scenario_cost - baseline_cost
        incremental_qaly = scenario_qaly - baseline_qaly

        # Calculate ICER
        icer = incremental_cost / incremental_qaly if incremental_qaly > 0 else np.inf
        icers.append(icer)
        costs.append(scenario_cost)
        qalys.append(scenario_qaly)
        labels.append(intervention)

        # Debugging: Print each intervention's data
        print(f"Intervention: {intervention}")
        print(f"Scenario Cost: {scenario_cost}, Scenario QALY: {scenario_qaly}")
        print(f"Incremental Cost: {incremental_cost}, Incremental QALY: {incremental_qaly}, ICER: {icer}")

    # Ensure there is data to plot
    if not costs or not qalys:
        raise ValueError("No valid data found to plot. Check scenario_results and baseline_results.")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(costs, qalys, c=icers, cmap="viridis", s=100, edgecolors="k")

    # Add annotations for each point
    for i, label in enumerate(labels):
        ax.annotate(label, (costs[i], qalys[i]), fontsize=10, ha="right")

    # Add labels, title, and colorbar
    ax.set_xlabel("Total Costs ($)", fontsize=14)
    ax.set_ylabel("Total QALYs", fontsize=14)
    ax.set_title("Cost vs. QALYs with ICER", fontsize=16)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("ICER ($/QALY)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()