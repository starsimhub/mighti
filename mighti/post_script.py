import numpy as np
import matplotlib.pyplot as plt

def analyze_hiv_ncd_prevalence(sim, prevalence_analyzer, ncd):
    """
    Analyze the prevalence of agents with both HIV and an NCD, only the NCD, or only HIV.

    Parameters:
    - sim: The simulation object
    - prevalence_analyzer: The prevalence analyzer with stored results
    - ncd: Name of the NCD (e.g., 'Type2Diabetes')
    """
    # Extract HIV and NCD prevalence data
    hiv_male_data = prevalence_analyzer.results.get('HIV_prevalence_male', None)
    hiv_female_data = prevalence_analyzer.results.get('HIV_prevalence_female', None)
    ncd_male_data = prevalence_analyzer.results.get(f'{ncd}_prevalence_male', None)
    ncd_female_data = prevalence_analyzer.results.get(f'{ncd}_prevalence_female', None)

    # Ensure data exists
    if hiv_male_data is None or hiv_female_data is None or ncd_male_data is None or ncd_female_data is None:
        print(f"[ERROR] No prevalence data available for HIV or {ncd}.")
        return

    # Compute mean prevalence across all age groups
    mean_hiv_male = np.mean(hiv_male_data, axis=1) * 100
    mean_hiv_female = np.mean(hiv_female_data, axis=1) * 100
    mean_ncd_male = np.mean(ncd_male_data, axis=1) * 100
    mean_ncd_female = np.mean(ncd_female_data, axis=1) * 100

    # Plot mean prevalence for each group
    plt.figure(figsize=(15, 8))
    plt.plot(sim.timevec, mean_hiv_male, label=f'Male with HIV', linewidth=2, color='blue')
    plt.plot(sim.timevec, mean_hiv_female, label=f'Female with HIV', linewidth=2, color='red', linestyle='dashed')
    plt.plot(sim.timevec, mean_ncd_male, label=f'Male with {ncd}', linewidth=2, color='cyan')
    plt.plot(sim.timevec, mean_ncd_female, label=f'Female with {ncd}', linewidth=2, color='magenta', linestyle='dashed')

    # Labels and title
    plt.xlabel('Year')
    plt.ylabel('Prevalence (%)')
    plt.title(f'Mean Prevalence Over Time')
    plt.legend()
    plt.grid()

    plt.show()
