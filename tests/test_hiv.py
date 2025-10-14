"""
Test MIGHTI / STIsim HIV module under Starsim 3.x
"""

import starsim as ss
import stisim as sti
import sciris as sc
import numpy as np
import matplotlib.pyplot as pl


def test_hiv_basic(n_agents=5000, start=2000, stop=2020, do_plot=False):
    sc.heading("Testing HIV basic dynamics")

    # --- Initialize HIV disease properly ---
    hiv = sti.HIV(
        init_prev=ss.bernoulli(p=0.05),  # FIXED: must be a Starsim distribution
        beta={"structuredsexual": [0.02, 0.02]},
    )

    hiv.pars.include_aids_deaths = True
    hiv.pars.include_care = False
    hiv.pars.art_efficacy = 0.0
    hiv.pars.p_hiv_death = ss.bernoulli(p=0.0005)

    net = sti.StructuredSexual()

    sim = ss.Sim(
        n_agents=n_agents,
        start=start,
        stop=stop,
        networks=[net],
        diseases=[hiv],
        dt=1,
        copy_inputs=False,
        label="HIV Mortality Test",
    )

    # Explicitly initialize before running
    sim.init()
    n_init_inf = hiv.infected.sum()
    print(f"Initial infections after init_post: {n_init_inf:,} ({n_init_inf/n_agents:.2%})")

    sim.run()

    # --- Prevalence and mortality checks ---
    hiv_res = sim.results.hiv
    assert np.any(hiv_res.prevalence > 0), "Prevalence remained zero"
    assert np.max(hiv_res.prevalence) <= 1.0, "Prevalence exceeds 100%"

    cum_deaths = getattr(hiv_res, "cum_deaths", None)
    total_deaths = cum_deaths[-1] if cum_deaths is not None else 0
    assert total_deaths > 0, "No AIDS-related deaths recorded"

    print(f"[✓] {int(total_deaths):,} cumulative AIDS-related deaths recorded.")

    if do_plot:
        pl.figure()
        pl.plot(hiv_res.timevec, hiv_res.prevalence, label="Prevalence")
        pl.plot(hiv_res.timevec, hiv_res.cum_deaths / n_agents, label="Cumulative deaths (per capita)")
        pl.xlabel("Year")
        pl.ylabel("Fraction")
        pl.legend()
        pl.title("HIV Dynamics with AIDS Mortality")
        pl.tight_layout()
        pl.show(block=True)

    return sim


if __name__ == "__main__":
    sim = test_hiv_basic(do_plot=True)
    