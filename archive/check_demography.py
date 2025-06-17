import starsim as ss
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Load demographic data for Eswatini
# ---------------------------------------------------------------------
# print("[DEBUG] Loading Eswatini demographic data...")

# Paths to demographic data files
csv_path_fertility = 'mighti/data/eswatini_asfr.csv'
csv_path_death = 'mighti/data/eswatini_deaths.csv'
csv_path_age = 'mighti/data/eswatini_age_2023.csv'

# Read data
fertility_rates = pd.read_csv(csv_path_fertility)
death_rates = pd.read_csv(csv_path_death)
age_data = pd.read_csv(csv_path_age)

# Initialize Pregnancy and Death modules
# print("[DEBUG] Initializing pregnancy and death modules...")
pregnancy = ss.Pregnancy(pars={'fertility_rate': fertility_rates})
death = ss.Deaths(pars={'death_rate': death_rates, 'rate_units': 1})

# Create People object
n_agents = 5000
# print(f"[DEBUG] Creating `People` object with {n_agents} agents...")
ppl = ss.People(n_agents=n_agents, age_data=age_data)

# ---------------------------------------------------------------------
# Initialize `Sim` BEFORE checking values
# ---------------------------------------------------------------------
# print("[DEBUG] Initializing Simulation (`sim`) before checking gender values...")
sim = ss.Sim(people=ppl, demographics=[pregnancy, death],copy_inputs=False).init()

# ---------------------------------------------------------------------
# Verify Gender Attributes
# ---------------------------------------------------------------------
print("[DEBUG] Checking `ppl.female`...")
try:
    unique_female_vals, female_counts = np.unique(ppl.female, return_counts=True)
    print(f"[DEBUG] `ppl.female` unique values: {unique_female_vals}")
    print(f"[DEBUG] `ppl.female` counts: {female_counts}")
except Exception as e:
    print(f"[ERROR] Exception occurred when checking `ppl.female`: {e}")

print("[DEBUG] Checking `ppl.male`...")
try:
    unique_male_vals, male_counts = np.unique(ppl.male, return_counts=True)
    print(f"[DEBUG] `ppl.male` unique values: {unique_male_vals}")
    print(f"[DEBUG] `ppl.male` counts: {male_counts}")
except Exception as e:
    print(f"[ERROR] Exception occurred when checking `ppl.male`: {e}")

# ---------------------------------------------------------------------
# Final Debug Message
# ---------------------------------------------------------------------
print("[DEBUG] Finished running `check_demography.py`.")