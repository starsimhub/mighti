"""
MIGHTI simulation test.

This test verifies that a basic MIGHTI simulation can:
- Initialize with MIGHTI modules (e.g., deaths, analyzers)
- Run successfully through all time steps
- Exit cleanly without raising exceptions
Compatible with Starsim 3.x
"""

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
    csv_path = 'tests/test_data/eswatini_parameters.csv'
    init_prev = ss.bernoulli(p=lambda sim, size=None: 0.1)  # ✅ Starsim 3.x-compatible callable

    disease = mi.AlcoholUseDisorder(csv_path=csv_path, pars={'init_prev': init_prev})

    # --- Define analyzers ---
    prevalence_analyzer = mi.PrevalenceAnalyzer(prevalence_data={}, diseases=['AlcoholUseDisorder'])
    survivorship_analyzer = mi.SurvivorshipAnalyzer()
    deaths_analyzer = mi.DeathsByAgeSexAnalyzer()

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
    assert hasattr(prevalence_analyzer, 'results'), "Prevalence analyzer missing results attribute."
    assert isinstance(prevalence_analyzer.results, dict) or hasattr(prevalence_analyzer.results, 'keys'), \
        "Analyzer results not properly initialized."

    print("✅ Basic MIGHTI run completed successfully.")


# --- Run as a standalone script ---
if __name__ == '__main__':
    test_basic_mighti_run()
    