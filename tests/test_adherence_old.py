"""
Unit tests for Adherence connectors (e.g., AdherenceFromDepression)
Ensures that CASM-linked adherence logic behaves as expected.
"""

import pytest
import numpy as np
import starsim as ss
import stisim as sti
import mighti as mi


# ---------------------------------------------------------------------
# Helper: minimal simulation
# ---------------------------------------------------------------------
def make_minisim(n_agents=500, start=2007, stop=2010, depression_prevalence=0.3):
    """Create a minimal Starsim simulation with HIV, Depression, ART, and adherence connector."""
    ppl = ss.People(n_agents)
    maternal = ss.MaternalNet()
    sexual = sti.StructuredSexual()
    networks = [maternal, sexual]

    death = ss.Deaths()
    pregnancy = ss.Pregnancy()

    # HIV with ART enabled
    hiv = sti.HIV()
    hiv.pars.include_care = True
    hiv.pars.art_efficacy = 0.9

    # MajorDepressiveDisorder with fixed prevalence
    csv_path = "mighti/data/eswatini_parameters.csv"  # valid placeholder
    dep = mi.MajorDepressiveDisorder(
        csv_path=csv_path,
        pars=dict(init_prev=ss.bernoulli(p=depression_prevalence))
    )
    diseases = [hiv, dep]

    art = sti.ART(pars={"init_prob": ss.bernoulli(p=0.9)})
    interventions = [art]

    adherence_conn = mi.AdherenceFromDepression()
    prevalence_analyzer = mi.PrevalenceAnalyzer(diseases=["HIV", "MajorDepressiveDisorder"])

    sim = ss.Sim(
        n_agents=n_agents,
        start=start,
        stop=stop,
        people=ppl,
        networks=networks,
        demographics=[death, pregnancy],
        diseases=diseases,
        interventions=interventions,
        connectors=[adherence_conn],
        analyzers=[prevalence_analyzer],
        label="AdherenceTest",
    )

    return sim, adherence_conn


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_adherence_connector_records_timesteps():
    """Connector should record adherence over time (using sim-managed connector)."""
    sim, _ = make_minisim()
    sim.run()

    # Fetch connector actually used by Starsim
    conn = next(c for name, c in sim.connectors.items() if "adherence" in name.lower())

    assert len(conn.time) > 0, "Connector did not record any timesteps."
    assert len(conn.time) == len(conn.mean_adherence), "Mismatch between time and adherence lengths."
    mean_val = np.mean(conn.mean_adherence)
    assert 0.0 < mean_val <= 1.0, f"Unexpected adherence mean: {mean_val:.3f}"


def test_adherence_reduces_with_higher_depression():
    """Mean adherence should be lower with higher depression prevalence."""
    sim_low, _ = make_minisim(depression_prevalence=0.3)
    sim_low.run()
    conn_low = next(c for name, c in sim_low.connectors.items() if "adherence" in name.lower())
    mean_low = np.mean(conn_low.mean_adherence)

    sim_high, _ = make_minisim(depression_prevalence=0.8)
    sim_high.run()
    conn_high = next(c for name, c in sim_high.connectors.items() if "adherence" in name.lower())
    mean_high = np.mean(conn_high.mean_adherence)

    assert not np.isnan(mean_low), "Low depression adherence values are NaN."
    assert not np.isnan(mean_high), "High depression adherence values are NaN."
    assert mean_high < mean_low, (
        f"Adherence did not decrease with higher depression prevalence "
        f"({mean_high:.3f} >= {mean_low:.3f})"
    )


def test_connector_is_attached_to_sim():
    """Connector should be attached and accessible via sim.connectors."""
    sim, _ = make_minisim()
    sim.run()

    connector_labels = list(sim.connectors.keys())
    matches = [c for name, c in sim.connectors.items() if "adherence" in name.lower()]

    assert matches, f"No adherence connector found; available: {connector_labels}"
    conn = matches[0]

    assert len(conn.mean_adherence) > 0, "Connector did not record adherence values."
    assert np.mean(conn.mean_adherence) < 1.0, "Adherence reduction not applied."


def test_adherence_improves_with_depressioncare():
    """
    Verify that adding DepressionCare intervention increases adherence
    compared with no treatment, and reduces depression prevalence.
    """
    # --- Baseline: no depression care ---
    sim_no_tx, _ = make_minisim(depression_prevalence=0.5)
    sim_no_tx.run()
    conn_no_tx = next(c for name, c in sim_no_tx.connectors.items() if "adherence" in name.lower())
    mean_adherence_no_tx = np.mean(conn_no_tx.mean_adherence)
    dep_prev_no_tx = sim_no_tx.results["majordepressivedisorder"].prevalence.mean()

    # --- With DepressionCare intervention ---
    depression_tx = mi.DepressionCare(
        product=ss.Product(name="Tx"),
        prob=0.9,               # 90% of depressed treated
        remission_boost=2.0,    # faster recovery
        adherence_boost=1.3,    # 30% adherence improvement
    )

    # Recreate full mini-sim but include the intervention
    ppl = ss.People(500)
    maternal = ss.MaternalNet()
    sexual = sti.StructuredSexual()
    networks = [maternal, sexual]
    death = ss.Deaths()
    pregnancy = ss.Pregnancy()

    csv_path = "mighti/data/eswatini_parameters.csv"
    hiv = sti.HIV()
    hiv.pars.include_care = True
    hiv.pars.art_efficacy = 0.9
    dep = mi.MajorDepressiveDisorder(csv_path=csv_path, pars=dict(init_prev=ss.bernoulli(p=0.5)))
    diseases = [hiv, dep]

    art = sti.ART(pars={"init_prob": ss.bernoulli(p=0.9)})
    adherence_conn = mi.AdherenceFromDepression()
    prevalence_analyzer = mi.PrevalenceAnalyzer(diseases=["HIV", "MajorDepressiveDisorder"])

    sim_tx = ss.Sim(
        n_agents=500,
        start=2007,
        stop=2010,
        people=ppl,
        networks=networks,
        demographics=[death, pregnancy],
        diseases=diseases,
        interventions=[art, depression_tx],  # include DepressionCare here
        connectors=[adherence_conn],
        analyzers=[prevalence_analyzer],
        label="AdherenceTestWithCare",
    )

    sim_tx.run()

    # --- Results ---
    conn_tx = next(c for name, c in sim_tx.connectors.items() if "adherence" in name.lower())
    mean_adherence_tx = np.mean(conn_tx.mean_adherence)
    dep_prev_tx = sim_tx.results["majordepressivedisorder"].prevalence.mean()

    # --- Assertions ---
    assert not np.isnan(mean_adherence_tx), "Adherence with treatment is NaN."
    assert mean_adherence_tx > mean_adherence_no_tx - 0.01, (
        f"Adherence did not meaningfully improve with DepressionCare "
        f"({mean_adherence_tx:.3f} vs {mean_adherence_no_tx:.3f})"
    )
    assert dep_prev_tx <= dep_prev_no_tx + 0.005, (
        f"Depression prevalence did not meaningfully decrease with DepressionCare "
        f"({dep_prev_tx:.3f} vs {dep_prev_no_tx:.3f})"
    )

# ---------------------------------------------------------------------
# Command line entry (manual run)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
