"""
Figure 3: Multimorbidity & overlap (Shapley decomposition)
- Decomposes total LE loss vs. idealized into condition attributions (Shapley values)
- Quantifies pairwise overlap with Shapley interaction indices
"""


import logging
import os
import matplotlib.pyplot as plt
import starsim as ss
import stisim as sti
import mighti as mi
import pandas as pd
import numpy as np
import prepare_data_for_year
import random
import json
from tqdm import tqdm



# ---------------------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------------------
logger = logging.getLogger("MIGHTI")
logger.setLevel(logging.INFO)

n_agents = 100_000
inityear = 2007
endyear = 2024
region = "eswatini"
CACHE_DIR = "results/shapley_cache"
PERMUTATION = 500
os.makedirs(CACHE_DIR, exist_ok=True)



# ---------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------
csv_path_params       = f"mighti/data/{region}_parameters.csv"
csv_path_interactions = "mighti/data/rel_sus.csv"
csv_prevalence        = f"mighti/data/{region}_prevalence.csv"
csv_path_fertility    = f"mighti/data/{region}_asfr.csv"
csv_path_death        = f"mighti/data/{region}_mortality_rates.csv"
csv_path_age          = f"mighti/data/{region}_age_distribution.csv"

# Ensure required demographic files exist
prepare_data_for_year.prepare_data_for_year(region, inityear)
prepare_data_for_year.prepare_data(region)

# ---------------------------------------------------------------------
# Load parameters & define which diseases to include
# ---------------------------------------------------------------------
df = pd.read_csv(csv_path_params)
df.columns = df.columns.str.strip()

healthconditions = [
    "HIV",
    "CardiovascularDiseases", "ChronicKidneyDisease", "Type1Diabetes", "Type2Diabetes",
    "RoadInjuries", "InterpersonalViolence", "COPD", "Asthma", 
    "ProstateCancer", "LungCancer", "CervicalCancer", "BreastCancer"
]


# ---------------------------------------------------------------------
# HELPER: cached run of subset of diseases
# ---------------------------------------------------------------------
def run_subset(disease_subset):
    CACHE_SALT = f"v3_{endyear}"
    subset_key = f"{CACHE_SALT}|" + (",".join(sorted(disease_subset)) if disease_subset else "None")
    cache_file = os.path.join(CACHE_DIR, f"e0_{subset_key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    # Demography
    death = ss.Deaths({"death_rate": pd.read_csv(csv_path_death), "rate_units": 1})
    pregnancy = ss.Pregnancy(pars={"fertility_rate": pd.read_csv(csv_path_fertility)})
    ppl = mi.make_people_with_age_sex(csv_path=csv_path_age, init_year=inityear, n_agents=n_agents)

    # Build disease modules (HIV via stisim; others via mighti)
    modules = []
    for cond in disease_subset:
        if cond.lower() == "hiv":
            hiv = sti.HIV()
            hiv.pars.include_aids_deaths = True
            hiv.pars.p_hiv_death = ss.bernoulli(p=0.00015)
            modules.append(hiv)
        else:
            cls = getattr(mi, cond, None)
            if cls is None:
                print(f"Skipping missing disease in subset: {cond}")
                continue
            modules.append(cls(csv_path=csv_path_params))

    sim = ss.Sim(
        n_agents=n_agents,
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        networks=[ss.MaternalNet(), sti.StructuredSexual()],
        diseases=modules,
        analyzers=[mi.DeathsByAgeSexAnalyzer(), mi.SurvivorshipAnalyzer()],
        label=f"Subset: {subset_key}",
        copy_inputs=False,
    )

    sim.run()

    e0_dict = mi.calculate_life_expectancy(sim, year=endyear)
    if not e0_dict:
        raise RuntimeError(f"calc_e0 returned None for subset {subset_key}")

    with open(cache_file, "w") as f:
        json.dump(e0_dict, f)
    return e0_dict


# ---------------------------------------------------------------------
# SHAPLEY DECOMPOSITION
# ---------------------------------------------------------------------
def shapley_decomposition(conditions, n_permutations=500):
    """
    Approximate Shapley values for each condition using Monte Carlo permutations.
    """
    shapley_vals = {sex: {cond: 0.0 for cond in conditions} for sex in ["Both", "Male", "Female"]}

    for perm in tqdm(range(n_permutations), desc="Computing Shapley permutations"):
        perm_list = random.sample(conditions, len(conditions))
        current_set = set()

        # life expectancy for the empty set (idealized)
        e0_prev = run_subset(set())

        for cond in perm_list:
            current_set.add(cond)
            e0_curr = run_subset(current_set)

            # Marginal contribution = current - previous
            for sex in ["Both", "Male", "Female"]:
                delta = e0_curr[sex] - e0_prev[sex]
                shapley_vals[sex][cond] += delta / n_permutations

            e0_prev = e0_curr

    # Convert to DataFrame
    records = []
    for sex in shapley_vals:
        for cond, val in shapley_vals[sex].items():
            records.append({"sex": sex, "condition": cond, "Shapley_value": val})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# RUN AND SAVE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df_shapley = shapley_decomposition(healthconditions, n_permutations=PERMUTATION)
    EPS = 0.02  # years
    df_shapley.loc[df_shapley["Shapley_value"].abs() < EPS, "Shapley_value"] = 0.0
    df_shapley.to_csv("Figures/shapley_e0_values.csv", index=False)
    print(df_shapley.head())



    # ---------------------------------------------------------------------
    # Load precomputed Shapley results
    # ---------------------------------------------------------------------
    df_shapley = pd.read_csv("Figures/shapley_e0_values.csv")

    # Flip sign if you want positive = life expectancy loss
    df_shapley["Shapley_value"] *= -1

    # ---------------------------------------------------------------------
    # Plot: add vertical reference line at x=0
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    sexes = ["Both", "Female", "Male"]
    titles = ["Both sexes", "Female", "Male"]

    for ax, sex, title in zip(axes, sexes, titles):
        df_sub = df_shapley[df_shapley["sex"] == sex].sort_values("Shapley_value", ascending=True)
        ax.barh(df_sub["condition"], df_sub["Shapley_value"], color="#377eb8", alpha=0.8)
        
        # vertical line at 0
        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
        
        ax.set_title(title)
        ax.set_xlabel("Shapley contribution to Δe₀ (years)")
        ax.invert_yaxis()

    fig.suptitle(f"Figure 3A. Shapley decomposition of life expectancy loss, {region.capitalize()} {endyear}")
    plt.tight_layout()
    plt.savefig("Figures/Fig3A_shapley_contributions_vline.png", dpi=600)
    plt.show()
