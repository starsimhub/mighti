"""
Figure 2: Population-attributable life expectancy loss (waterfall plot)
----------------------------------------------------------------------
Compares the life expectancy gap between 'Idealized (No diseases)'
and 'Observed (With diseases)' simulations, and decomposes the loss
by individual health conditions modeled in MIGHTI.
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


# ---------------------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------------------
logger = logging.getLogger("MIGHTI")
logger.setLevel(logging.INFO)

n_agents = 100_000
inityear = 2007
endyear = 2024
region = "eswatini"


# ---------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------
csv_path_params       = f"mighti/data/{region}_parameters.csv"
csv_path_interactions = "mighti/data/rel_sus.csv"
csv_prevalence        = f"mighti/data/{region}_prevalence.csv"
csv_path_fertility    = f"mighti/data/{region}_asfr.csv"
csv_path_death        = f"mighti/data/{region}_mortality_rates.csv"
csv_path_age          = f"mighti/data/{region}_age_distribution_{inityear}.csv"
csv_path_intervention = f"mighti/data/{region}_intervention.csv"

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
    "RoadInjuries", "InterpersonalViolence", "COPD", "Asthma", "Influenza",
    "ProstateCancer", "LungCancer", "CervicalCancer", "BreastCancer"
]


# ---------------------------------------------------------------------
# Read prevalence table and build callable prevalence data
# ---------------------------------------------------------------------
prevalence_data_df = pd.read_csv(csv_prevalence)
prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases=healthconditions, prevalence_data=prevalence_data_df, inityear=inityear
)

def get_prevalence_function(disease):
    def prevalence_func(sim, uids, size=None):
        return mi.age_sex_dependent_prevalence(
            disease=disease, prevalence_data=prevalence_data,
            age_bins=age_bins, sim=sim, size=size,
        )
    return prevalence_func


# ---------------------------------------------------------------------
# Helper: single-condition simulation builder
# ---------------------------------------------------------------------
def make_sim_for_condition(condition=None, label=None):

    death_rates = {"death_rate": pd.read_csv(csv_path_death), "rate_units": 1}
    death = ss.Deaths(death_rates)
    fertility_rate = {"fertility_rate": pd.read_csv(csv_path_fertility)}
    pregnancy = ss.Pregnancy(pars=fertility_rate)

    ppl = mi.make_people_with_age_sex(
        csv_path="mighti/data/eswatini_age_distribution.csv",
        init_year=inityear,
        n_agents=n_agents,
    )

    maternal = ss.MaternalNet()
    structuredsexual = sti.StructuredSexual()
    networks = [maternal, structuredsexual]

    disease_module = None
    if condition is not None:
        if condition.lower() == "hiv":
            disease_module = sti.HIV()
            prev_func = get_prevalence_function("HIV")
            disease_module.pars.init_prev = ss.bernoulli(
                p=lambda sim, uids, size=None: prev_func(sim, uids, size)
            )
            disease_module.pars.beta = {
                "structuredsexual": [0.0296, 0.0296],
                "maternal": [0.00112, 0.00112],
            }
            disease_module.pars.include_aids_deaths = True
            disease_module.pars.p_hiv_death = ss.bernoulli(p=0.00015)
            disease_module.pars.include_care = True
            disease_module.pars.art_efficacy = 0.9
        else:
            cls = getattr(mi, condition, None)
            if cls is not None:
                disease_module = cls(csv_path=csv_path_params)
            else:
                print(f"Warning: condition '{condition}' not found in MIGHTI.")

    diseases = [disease_module] if disease_module is not None else []

    analyzers = [
        mi.DeathsByAgeSexAnalyzer(),
        mi.SurvivorshipAnalyzer(),
    ]

    sim = ss.Sim(
        n_agents=n_agents,
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        networks=networks,
        diseases=diseases,
        analyzers=analyzers,
        label=label or condition or "Idealized (No diseases)",
        copy_inputs=False,
    )
    return sim


# ---------------------------------------------------------------------
# Run all conditions
# ---------------------------------------------------------------------
records = []

# Baseline (Idealized)
baseline_sim = make_sim_for_condition(condition=None, label="Idealized (No diseases)")
baseline_sim.run()
baseline_e0_dict = mi.calculate_life_expectancy(baseline_sim, year=endyear)
for sex, e0_val in baseline_e0_dict.items():
    records.append({"condition": "Idealized (No diseases)", "e0": e0_val, "year": endyear, "sex": sex})

# Each condition (one at a time)
for cond in healthconditions:
    print(f"Running condition: {cond}")
    sim = make_sim_for_condition(condition=cond, label=cond)
    sim.run()
    e0_dict = mi.calculate_life_expectancy(sim, year=endyear)
    for sex, e0_val in e0_dict.items():
        records.append({"condition": cond, "e0": e0_val, "year": endyear, "sex": sex})

# Save results
df = pd.DataFrame(records)
df.to_csv("result_conditionwise_e0.csv", index=False)
print("Saved to result_conditionwise_e0.csv")


# ---------------------------------------------------------------------
# Postprocessing & plotting
# ---------------------------------------------------------------------
df = pd.read_csv("result_conditionwise_e0.csv")
plot_year = 2024
df = df[df["year"] == plot_year].copy()

# Baseline and disease-only splits
e0_ideal = df[df["condition"] == "Idealized (No diseases)"]
df = df[df["condition"] != "Idealized (No diseases)"].copy()

# Pivot to condition × sex
df_pivot = df.pivot(index="condition", columns="sex", values="e0")
e0_ideal = e0_ideal.set_index("sex")["e0"].to_dict()

# Compute losses
for sex in ["Both", "Female", "Male"]:
    df_pivot[sex] = e0_ideal[sex] - df_pivot[sex]

df_pivot = df_pivot.sort_values("Both", ascending=False)

color_map = {
    "HIV": "#E41A1C", "CardiovascularDiseases": "#009E73",
    "ChronicKidneyDisease": "#56B4E9", "Type1Diabetes": "#0072B2",
    "Type2Diabetes": "#E69F00", "COPD": "#D55E00", "Asthma": "#CC79A7",
    "LungCancer": "#882255", "BreastCancer": "#AA4499", "CervicalCancer": "#332288",
    "ProstateCancer": "#44AA99", "Influenza": "#117733",
    "RoadInjuries": "#999933", "InterpersonalViolence": "#661100"
}
df_pivot["color"] = [color_map.get(cond, "#BBBBBB") for cond in df_pivot.index]

# ---------------------------------------------------------------------
# Plot: stacked loss relative to idealized
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
sexes = ["Both", "Female", "Male"]
titles = ["Both sexes", "Female", "Male"]

for ax, sex, title in zip(axes, sexes, titles):
    ideal = e0_ideal[sex]
    total_loss = df_pivot[sex].sum()
    observed = ideal - total_loss

    current_top = ideal
    for cond, row in df_pivot.iterrows():
        loss = row[sex]
        ax.bar(
            ["Realized"],
            -loss,
            bottom=current_top,
            color=row["color"],
            edgecolor="black",
            linewidth=0.3,
        )
        current_top -= loss

    # Idealized vs Realized reference bars
    ax.bar(["Idealized"], ideal, color="#CCCCCC", edgecolor="black", label="Idealized (No diseases)")
    ax.bar(["Realized"], observed, color="#333333", edgecolor="black", label="Realized (With diseases)")

    ax.set_title(title)
    ax.set_ylim(0, 80)
    ax.set_ylabel("Life expectancy at birth (years)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(frameon=False, loc="upper right")
fig.suptitle(f"Figure 2. Population-attributable life expectancy loss by condition, Eswatini {plot_year}", y=0.98)

plt.tight_layout()
plt.savefig("Figures/Fig2_stacked_LE_loss_corrected.png", dpi=600)
plt.show()
