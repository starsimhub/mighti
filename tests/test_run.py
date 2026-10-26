"""
MIGHTI simulation test.

This test verifies that a basic MIGHTI simulation can:
- Initialize with MIGHTI modules (e.g., deaths, analyzers)
- Run successfully through all time steps
- Exit cleanly without raising exceptions
"""

import os
import starsim as ss
import mighti as mi
import pandas as pd


def test_basic_mighti_run():
    n_agents = 500

    # --- Minimal mortality table ---
    death_rates = {
        'death_rate': pd.DataFrame({
            'Time': [2000] * 10,
            'AgeGrpStart': list(range(5)) * 2,
            'Sex': ['Male'] * 5 + ['Female'] * 5,
            'mx': [0.01] * 10,
        }),
        'rate_units': 1,
    }

    deaths = ss.Deaths(death_rates)

    # --- Create people ---
    ppl = ss.People(n_agents=n_agents)

    # --- Define a simple disease ---
    csv_path = os.path.join(os.path.dirname(__file__), 'test_data', 'eswatini_parameters.csv')
    init_prev = ss.bernoulli(p=lambda sim, uids: 0.1)

    disease = mi.diseases.AlcoholUseDisorder(csv_path=csv_path, pars={'init_prev': init_prev})

    # --- Define analyzers ---
    prevalence_analyzer = mi.analyzers.PrevalenceAnalyzer(prevalence_data={}, diseases=['AlcoholUseDisorder'])
    survivorship_analyzer = mi.analyzers.SurvivorshipAnalyzer()
    deaths_analyzer = mi.analyzers.DeathsByAgeSexAnalyzer()

    analyzers = [deaths_analyzer, survivorship_analyzer, prevalence_analyzer]

    # --- Build simulation ---
    sim = ss.Sim(
        n_agents=n_agents,
        start=2000,
        stop=2002,
        people=ppl,
        demographics=[deaths],
        diseases=[disease],
        analyzers=analyzers,
        label='test_basic_mighti_run',
    )

    # --- Run simulation ---
    sim.run()

    # --- Assertions ---
    assert sim.ti == len(sim.t.tvec) - 1, "Simulation did not complete full timeline."
    prevalence_analyzer = sim.analyzers.prevalence_analyzer
    assert prevalence_analyzer.results.alcoholusedisorder_prev_total[-1] > 0, "Prevalence analyzer recorded no prevalence"

    print("✅ Basic MIGHTI run completed successfully.")


# --- Run as a standalone script ---
if __name__ == '__main__':
    test_basic_mighti_run()
    