import starsim as ss
import pandas as pd
import numpy as np

# Debug log function
def debug_log(message):
    print(f"[DEBUG] {message}")

try:
    # Load Eswatini demographic data
    debug_log("Loading Eswatini demographic data...")
    fertility_rates = pd.read_csv("mighti/data/eswatini_asfr.csv")
    death_rates = pd.read_csv("mighti/data/eswatini_deaths.csv")
    age_data = pd.read_csv("mighti/data/eswatini_age_2023.csv")

    # Initialize pregnancy and death models
    debug_log("Initializing pregnancy and death modules...")
    pregnancy = ss.Pregnancy(pars={"fertility_rate": fertility_rates})
    death = ss.Deaths(pars={"death_rate": death_rates, "rate_units": 1})

    demographics = [pregnancy, death]

    # Create People object
    n_agents = 5000
    debug_log(f"Creating `People` object with {n_agents} agents...")
    ppl = ss.People(n_agents=n_agents, age_data=age_data)

    # Debug `male` and `female` attributes
    debug_log("Checking `ppl.female.raw`...")
    if hasattr(ppl, "female"):
        female_vals = ppl.female.raw
        unique_female_vals, female_counts = np.unique(female_vals, return_counts=True)
        debug_log(f"`ppl.female.raw` unique values: {unique_female_vals}")
        debug_log(f"`ppl.female.raw` counts: {female_counts}")
    else:
        debug_log("[ERROR] `ppl` does not have `female` attribute!")

    debug_log("Checking `ppl.male.raw`...")
    if hasattr(ppl, "male"):
        male_vals = ppl.male.raw
        unique_male_vals, male_counts = np.unique(male_vals, return_counts=True)
        debug_log(f"`ppl.male.raw` unique values: {unique_male_vals}")
        debug_log(f"`ppl.male.raw` counts: {male_counts}")
    else:
        debug_log("[ERROR] `ppl` does not have `male` attribute!")

    # Check if total male + female count matches `n_agents`
    if hasattr(ppl, "female") and hasattr(ppl, "male"):
        total_gender_count = len(female_vals) + len(male_vals)
        debug_log(f"Total gender count: {total_gender_count} (Expected: {n_agents})")
        if total_gender_count != n_agents:
            debug_log("[ERROR] Gender assignment mismatch!")

    debug_log("Finished gender distribution checks.")

except Exception as e:
    print(f"[ERROR] Exception occurred: {e}")