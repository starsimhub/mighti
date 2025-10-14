"""
Test MIGHTI disease prevalence initialization and age-sex assignment.

Ensures that initialize_prevalence_data() and age_sex_dependent_prevalence()
produce valid, non-negative prevalence probabilities for all diseases with
available data. Skips incomplete or NaN-only conditions automatically.
"""

import os
import pandas as pd
import numpy as np
import starsim as ss
import mighti as mi


def test_disease_prevalence_from_data(n_agents=500, inityear=2007, tol=0.02):
    """Run MIGHTI prevalence sanity test with robust diagnostics."""
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    thisdir = os.path.dirname(__file__)
    param_path = os.path.join(thisdir, "test_data", "eswatini_parameters.csv")
    prevalence_path = os.path.join(thisdir, "test_data", "eswatini_prevalence.csv")

    assert os.path.exists(param_path), f"Missing {param_path}"
    assert os.path.exists(prevalence_path), f"Missing {prevalence_path}"

    params_df = pd.read_csv(param_path)
    prevalence_df = pd.read_csv(prevalence_path)
    params_df.columns = params_df.columns.str.strip()
    prevalence_df.columns = prevalence_df.columns.str.strip()

    diseases = params_df.query("condition != 'HIV'")["condition"].unique().tolist()
    print(f"\n🧪 Testing {len(diseases)} diseases for year {inityear}...")

    # ------------------------------------------------------------------
    # 2. Initialize prevalence data
    # ------------------------------------------------------------------
    prevalence_data, age_bins = mi.initialize_prevalence_data(
        diseases=diseases, prevalence_data=prevalence_df, inityear=inityear
    )
    assert isinstance(prevalence_data, dict) and isinstance(age_bins, dict)

    # ------------------------------------------------------------------
    # 3. Initialize a simple Starsim population
    # ------------------------------------------------------------------
    sim = ss.Sim(n_agents=n_agents).init()
    uids = sim.people.uid.raw

    # ------------------------------------------------------------------
    # 4. Evaluate each disease
    # ------------------------------------------------------------------
    any_nonzero = False
    for disease in diseases:
        bins = age_bins.get(disease, [])
        if not bins or len(bins) < 2:
            print(f"  [⚠] Skipping {disease}: insufficient or empty age bins.")
            continue
        if disease not in prevalence_data:
            print(f"  [⚠] Skipping {disease}: missing in prevalence_data.")
            continue

        # Compute age-sex dependent prevalence
        try:
            probs = mi.age_sex_dependent_prevalence(disease, prevalence_data, age_bins, sim, uids)
        except Exception as e:
            raise RuntimeError(f"  [❌] Error computing prevalence for {disease}: {e}")

        probs = np.nan_to_num(np.asarray(probs), nan=0.0)
        assert len(probs) == n_agents, f"{disease}: length mismatch ({len(probs)} vs {n_agents})"
        assert np.all((probs >= 0) & (probs <= 1)), f"{disease}: probability out of bounds"

        mean_val = probs.mean()
        nonzero = np.sum(probs > 0)
        any_nonzero = any_nonzero or (mean_val > 0)

        print(f"  {disease:20s} bins={len(bins):2d}  mean={mean_val:.3f}  "
              f"nonzero={nonzero}/{n_agents}  min={probs.min():.3f}  max={probs.max():.3f}")

    # ------------------------------------------------------------------
    # 5. Global assertions
    # ------------------------------------------------------------------
    assert any_nonzero, "All diseases returned zero prevalence!"
    print("\n[✓] All prevalence functions returned valid, bounded probabilities.")


# ----------------------------------------------------------------------
# 6. Run as standalone script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    test_disease_prevalence_from_data()
    