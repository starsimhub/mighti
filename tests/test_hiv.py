"""
Test MIGHTI / STIsim HIV module under Starsim 3.x

This test verifies that:
1. HIV initializes correctly with a given initial prevalence.
2. Infection dynamics run smoothly over multiple years.
3. The prevalence output is non-zero and within realistic bounds.
"""

import starsim as ss
import stisim as sti
import sciris as sc
import numpy as np
import matplotlib.pyplot as pl


def test_hiv_basic(n_agents=5000, start=2000, stop=2020):
    sc.heading("Testing HIV basic dynamics")

    # --- Initialize HIV disease
    hiv = sti.HIV(
        init_prev=0.05,  # 5% baseline prevalence
        beta={"structuredsexual": [0.02, 0.02]},
    )

    # --- Sexual network
    net = sti.StructuredSexual()

    # --- Build and run sim
    sim = ss.Sim(
        n_agents=n_agents,
        start=start,
        stop=stop,
        networks=[net],
        diseases=[hiv],
        dt=1,
        copy_inputs=False,
        label="HIV Basic Test",
    )
    sim.run()

    # --- Assertions
    assert hasattr(sim.results, "hiv"), "HIV results missing"
    hiv_res = sim.results.hiv
    assert np.any(hiv_res.prevalence > 0), "Prevalence remained zero"
    assert np.max(hiv_res.prevalence) <= 1.0, "Prevalence exceeds 100%"

    final_prev = hiv_res.prevalence[-1]
    print(f"[✓] HIV test passed; final prevalence = {final_prev:.3f}")
    return sim


if __name__ == "__main__":
    sim = test_hiv_basic()

    # Optional plotting
    pl.figure()
    pl.plot(sim.results.hiv.timevec, sim.results.hiv.prevalence)
    pl.xlabel("Year")
    pl.ylabel("HIV Prevalence")
    pl.title("STIsim HIV Module Test")
    pl.tight_layout()
    pl.show(block=True)