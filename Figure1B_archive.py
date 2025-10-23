"""
figure1.py — Produce Figure 1 for the MIGHTI–Eswatini paper
Observed vs. idealized survival (panel A) and life expectancy at birth, 2007–2022 (panel B).
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import mighti as mi
import starsim as ss
import prepare_data_for_year
import stisim as sti
import numpy as np


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

# Post-process targets
mx_path = f"mighti/data/{region}_mx.csv"
ex_path = f"mighti/data/{region}_ex.csv"

# Ensure required demographic files exist
prepare_data_for_year.prepare_data_for_year(region, inityear)
prepare_data_for_year.prepare_data(region)

# ---------------------------------------------------------------------
# Load parameters & define which diseases to include
# ---------------------------------------------------------------------
df = pd.read_csv(csv_path_params)
df.columns = df.columns.str.strip()

# healthconditions = [condition for condition in df.condition if condition != "HIV"]
healthconditions = ["CardiovascularDiseases", "ChronicKidneyDisease","Type1Diabetes", "Type2Diabetes",
                     "RoadInjuries","InterpersonalViolence","COPD","Asthma","Influenza",
                     "ProstateCancer","LungCancer","CervicalCancer","BreastCancer"]
    
diseases = ["HIV"] + healthconditions

# ---------------------------------------------------------------------
# Read prevalence table and build callable prevalence data
# ---------------------------------------------------------------------
prevalence_data_df = pd.read_csv(csv_prevalence)
prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases=diseases, prevalence_data=prevalence_data_df, inityear=inityear
)

def get_prevalence_function(disease):
    def prevalence_func(sim, uids, size=None):
        return mi.age_sex_dependent_prevalence(
            disease=disease, prevalence_data=prevalence_data,
            age_bins=age_bins, sim=sim, size=size,
        )
    return prevalence_func


def make_sim(year):
    survivorship_analyzer = mi.SurvivorshipAnalyzer()
    deaths_analyzer = mi.DeathsByAgeSexAnalyzer()

    death_cause_analyzer = mi.ConditionAtDeathAnalyzer(
            conditions=healthconditions)

    analyzers = [deaths_analyzer, survivorship_analyzer, death_cause_analyzer]

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
    
    disease_objects = []
    hiv = sti.HIV()

    # Assign prevalence
    prev_func = get_prevalence_function('HIV')
    hiv.pars.init_prev = ss.bernoulli(
        p=lambda sim, uids, size=None: prev_func(sim, uids, size)
    )
    hiv.pars.beta = {
    'structuredsexual': [0.029594299274445842, 0.029594299274445842],
    'maternal': [0.0011249414706988527, 0.0011249414706988527],
    }

    # --- AIDS mortality ---
    hiv.pars.include_aids_deaths = True
    hiv.pars.p_hiv_death = ss.bernoulli(p=0.00015)
    hiv.pars.include_care = True
    hiv.pars.art_efficacy = 0.9

    disease_objects.append(hiv)


    def make_init_prev_func(disease):
        prev_func = get_prevalence_function(disease)
        return lambda sim, uids, size=None: prev_func(sim, uids, size)

    # Other diseases
    for disease in healthconditions:
        disease_class = getattr(mi, disease, None)
        if disease_class:
            init_prev = ss.bernoulli(p=make_init_prev_func(disease))
            disease_obj = disease_class(csv_path=csv_path_params, pars={"init_prev": init_prev})
            disease_objects.append(disease_obj)

    ncd_hiv_rel_sus = df.set_index("condition")["rel_sus"].to_dict()
    ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
    connectors = [ncd_hiv_connector]

    ncd_interactions = mi.read_interactions(csv_path_interactions)
    connectors.extend(mi.create_connectors(ncd_interactions))

    sim_with = ss.Sim(
        n_agents=n_agents,
        start=inityear,
        stop=endyear,
        people=ppl,
        networks=networks,
        demographics=[pregnancy, death],
        diseases=disease_objects,      
        connectors=connectors,
        analyzers=analyzers,
        copy_inputs=False,
        label="Observed (With diseases)"
    )

    sim_without = ss.Sim(
        n_agents=n_agents,
        start=inityear,
        stop=endyear,
        people=ppl,
        networks=networks,
        demographics=[pregnancy, death],
        analyzers=analyzers,
        copy_inputs=False,
        label="Idealized (No diseases)"
    )    

    msim = ss.MultiSim([sim_with, sim_without])

    return msim

# ---------------------------------------------------------------------
# Utility: helpers
# ---------------------------------------------------------------------
def get_deaths_module(sim):
    """
    Return the DeathsByAgeSexAnalyzer or any analyzer with 'death' in its name.
    Works with dict-style sim.analyzers (MIGHTI default).
    """
    if hasattr(sim, "analyzers") and isinstance(sim.analyzers, dict):
        for a in sim.analyzers.values():
            if isinstance(a, mi.DeathsByAgeSexAnalyzer) or "death" in a.label.lower():
                return a
    raise ValueError(
        f"Deaths analyzer not found. Available analyzers: {list(sim.analyzers.keys())}"
    )


    
years = list(range(2008, 2050))  # or any range you like


life_expectancy_by_year = []

# Run MultiSim
for year in years:
    msim = make_sim(year)
    msim.run()

    for sim in msim.sims:
        label = sim.label
        deaths_module = get_deaths_module(sim)
        df_mx = mi.calculate_mortality_rates(sim, deaths_module, year=year, max_age=100, radix=n_agents)

        df_male = df_mx[df_mx['sex'] == 'Male']
        df_female = df_mx[df_mx['sex'] == 'Female']
        lt = mi.calculate_life_table_from_mx(sim, df_male, df_female)

        # Male and female life expectancy at birth
        for sex in ['Male', 'Female']:
            e0 = lt[(lt['sex'] == sex) & (lt['Age'] == 0)]['e(x)'].values[0]
            life_expectancy_by_year.append({
                'year': year,
                'scenario': label,
                'sex': sex,
                'e0': e0
            })

        # Both sexes (weighted average)
        lt0 = lt[lt['Age'] == 0].copy()
        total_l0 = lt0['l(x)'].sum()
        lt0['weight'] = lt0['l(x)'] / total_l0
        weighted_e0 = (lt0['e(x)'] * lt0['weight']).sum()
        life_expectancy_by_year.append({
            'year': year,
            'scenario': label,
            'sex': 'Both',
            'e0': weighted_e0
        })

# Convert to DataFrame
le_df = pd.DataFrame(life_expectancy_by_year)

# Pivot to view as year × scenario × sex
pivot_df = le_df.pivot_table(index='year', columns=['scenario', 'sex'], values='e0').reset_index()

# Display result
pivot_df.to_csv("result_hivtest.csv", index=False)
 

# ---------------------------------------------------------------------
# Load and combine 2 header rows properly
# ---------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = "result_hivtest.csv"
assert os.path.exists(csv_path), f"File not found: {csv_path}"

raw = pd.read_csv(csv_path, header=None)
print("Raw head:\n", raw.head())

# Combine first two header rows
header_top = raw.iloc[0].astype(str).tolist()
header_bottom = raw.iloc[1].astype(str).tolist()

combined_cols = []
for top, bottom in zip(header_top, header_bottom):
    top = top.strip()
    bottom = bottom.strip()
    if top.lower() == "year":
        combined_cols.append("year")
    elif bottom.lower() in ["both", "male", "female"]:
        combined_cols.append(f"{top}.{bottom}")
    else:
        combined_cols.append(top)

# Create clean DataFrame
df = raw.iloc[2:].copy()
df.columns = combined_cols

# Convert numeric
df["year"] = pd.to_numeric(df["year"], errors="coerce")
for c in df.columns:
    if c != "year":
        df[c] = pd.to_numeric(df[c], errors="coerce")

print("\nFinal combined header columns:")
print(df.columns.tolist())
print(df.head())

# Reshape wide → tidy
scenario_cols = [c for c in df.columns if c != "year"]
df_long = df.melt(id_vars="year", value_vars=scenario_cols,
                  var_name="scenario_sex", value_name="e0")

df_long["scenario"] = df_long["scenario_sex"].str.extract(r"^(.*?)\.")[0]
df_long["sex"] = df_long["scenario_sex"].str.extract(r"\.([A-Za-z]+)$")[0]
print("\nTidy preview:\n", df_long.head())


# ---------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------
colors = {'Male': '#538DD5', 'Female': '#B23A48', 'Both': 'gray'}
linestyles = {'Idealized (No diseases)': '--', 'Observed (With diseases)': '-'}

# ---------------------------------------------------------------------
# Panel 1 — split by sex (Figure 1B style)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for i, sex in enumerate(['Male', 'Female']):
    ax = axes[i]
    sub = df_long[df_long['sex'] == sex]
    for scenario, grp in sub.groupby('scenario'):
        color = colors.get(sex, 'black')
        ls = linestyles.get(scenario, '-')
        ax.plot(grp['year'], grp['e0'], lw=2.5, ls=ls, color=color, label=scenario)
    ax.set_title(sex)
    ax.set_xlabel("Year")
    if i == 0:
        ax.set_ylabel("Life expectancy at birth (years)")
    ax.grid(alpha=0.3)
axes[0].legend(frameon=False)
fig.suptitle("Life Expectancy at Birth (e₀) — Eswatini", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs("Figures", exist_ok=True)
plt.savefig("Figures/Fig1B_life_expectancy_by_sex.png", dpi=300)
plt.show()

# ---------------------------------------------------------------------
# Panel 2 — combined (Both sexes)
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
for scenario, sub in df_long[df_long['sex'] == 'Both'].groupby('scenario'):
    ls = linestyles.get(scenario, '-')
    plt.plot(sub['year'], sub['e0'], lw=3, ls=ls, label=scenario)
plt.xlabel("Year")
plt.ylabel("Life expectancy at birth (years)")
plt.title("Life Expectancy (Both sexes) — Eswatini")
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Figures/Fig1B_life_expectancy_both.png", dpi=300)
plt.show()



