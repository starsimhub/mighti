import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_cea(prevalence_analyzer, cost_data, utility_data):
    """
    Perform cost-effectiveness analysis on the simulation results.

    Parameters:
    - prevalence_analyzer: The PrevalenceAnalyzer object with stored results
    - cost_data: DataFrame with cost information for each disease
    - utility_data: DataFrame with utility information for each disease

    Returns:
    - cea_results: DataFrame with cost-effectiveness analysis results
    """

    # Initialize an empty DataFrame for storing CEA results
    cea_results = pd.DataFrame(columns=['Disease', 'Cost', 'Utility', 'ICER'])

    for disease in prevalence_analyzer.diseases:
        # Extract prevalence data for the disease
        num_with_disease = prevalence_analyzer.results[f'{disease}_num_total'].sum()
        den_with_disease = prevalence_analyzer.results[f'{disease}_den_total'].sum()
        prevalence = num_with_disease / den_with_disease if den_with_disease > 0 else 0

        # Calculate total cost and utility
        cost = cost_data.loc[cost_data['Disease'] == disease, 'Cost'].values[0]
        utility = utility_data.loc[utility_data['Disease'] == disease, 'Utility'].values[0]

        # Assuming current cost and utility are zero
        current_cost = 0
        current_utility = 0

        # Calculate ICER (Incremental Cost-Effectiveness Ratio)
        delta_cost = cost - current_cost
        delta_utility = utility - current_utility
        icer = delta_cost / delta_utility if delta_utility > 0 else float('inf')

        # Store the results in a DataFrame
        result_df = pd.DataFrame([{'Disease': disease, 'Cost': cost, 'Utility': utility, 'ICER': icer}])
        cea_results = pd.concat([cea_results, result_df], ignore_index=True)

    return cea_results

def plot_cea_results(cea_results):
    """
    Plot the cost-effectiveness analysis results.

    Parameters:
    - cea_results: DataFrame with cost-effectiveness analysis results
    """
    plt.figure(figsize=(14, 10))  # Wider width for the plot
    sns.barplot(data=cea_results, y='Disease', x='ICER', color='lightblue')
    plt.title('Cost-Effectiveness Analysis (ICER) by Disease')
    plt.xlabel('ICER')
    plt.ylabel('Disease')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
