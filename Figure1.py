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


# ---------------------------------------------------------------------
# Analyzers
# ---------------------------------------------------------------------
survivorship_analyzer = mi.SurvivorshipAnalyzer()
deaths_analyzer = mi.DeathsByAgeSexAnalyzer()

death_cause_analyzer = mi.ConditionAtDeathAnalyzer(
    conditions=healthconditions)

analyzers = [deaths_analyzer, survivorship_analyzer, death_cause_analyzer]

survivorship_analyzer2 = mi.SurvivorshipAnalyzer()
deaths_analyzer2 = mi.DeathsByAgeSexAnalyzer()

death_cause_analyzer2 = mi.ConditionAtDeathAnalyzer(
    conditions=healthconditions)

analyzers2 = [deaths_analyzer2, survivorship_analyzer2, death_cause_analyzer2]

# ---------------------------------------------------------------------
# Demographics & networks
# ---------------------------------------------------------------------
death_rates = {"death_rate": pd.read_csv(csv_path_death), "rate_units": 1}
death = ss.Deaths(death_rates)

fertility_rate = {"fertility_rate": pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rate)

ppl1 = mi.make_people_with_age_sex(
    csv_path="mighti/data/eswatini_age_distribution.csv",
    init_year=inityear,
    n_agents=n_agents,
)
ppl2 = mi.make_people_with_age_sex(
    csv_path="mighti/data/eswatini_age_distribution.csv",
    init_year=inityear,
    n_agents=n_agents,
)

maternal = ss.MaternalNet()
structuredsexual = sti.StructuredSexual()
networks = [maternal, structuredsexual]

# ---------------------------------------------------------------------
# Diseases 
# ---------------------------------------------------------------------
disease_objects = []

# --- HIV ---
hiv = sti.HIV()

# Assign prevalence
prev_func = get_prevalence_function('HIV')
hiv.pars.init_prev = ss.bernoulli(
    p=lambda sim, uids, size=None: prev_func(sim, uids, size)
)

# Transmission parameters
# Best pars: {'hiv_beta_m2f': 0.09553835265049065, 'hiv_beta_m2c': 0.003895160642773216}
# Best pars: {'hiv_beta_m2f': 0.041126225026336546, 'hiv_beta_m2c': 0.02313161100759324}
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


# ---------------------------------------------------------------------
# Connectors (HIV ↔ NCD, plus other NCD interactions)
# ---------------------------------------------------------------------
ncd_hiv_rel_sus = df.set_index("condition")["rel_sus"].to_dict()
ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
connectors = [ncd_hiv_connector]

ncd_interactions = mi.read_interactions(csv_path_interactions)
connectors.extend(mi.create_connectors(ncd_interactions))

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

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sim_with = ss.Sim(
        n_agents=n_agents,
        start=inityear,
        stop=endyear,
        people=ppl1,
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
        people=ppl2,
        networks=networks,
        demographics=[pregnancy, death],
        # diseases=disease_objects,      
        # connectors=connectors,
        analyzers=analyzers2,
        copy_inputs=False,
        label="Idealized (No diseases)"
    )    

    # Run
    msim = ss.MultiSim([sim_with, sim_without])
    msim.run()

    for sim in msim.sims:
        df_test = sim.analyzers['survivorship_analyzer'].to_df()
        print(f"\n{sim.label} survivorship DataFrame:")
        print(df_test.head(10))
        print(df_test.tail(5))

    # ---------------------------------------------------------------------
    # Figure 1 – Observed vs Idealized survival and life expectancy
    # ---------------------------------------------------------------------
    # Extract survivorship data from both sims
    dfs_surv = []
    for sim in msim.sims:
        surv_df = sim.analyzers['survivorship_analyzer'].to_df()
        surv_df['scenario'] = sim.label
        dfs_surv.append(surv_df)
    df_surv = pd.concat(dfs_surv)


    # ---------------------------------------------------------------------
    # Panel A – Survivorship by sex with UN data overlay
    # ---------------------------------------------------------------------
    male_c, female_c = "#538DD5", "#B23A48"
    plot_year = endyear

    # Convert observed UN/WPP mortality to life table lx (scaled 0–1)
    def life_table_from_mx(mx_df, year):
        """
        Build a life table l(x) by sex for the given year from an observed mx DataFrame.
        Accepts either 'year' or 'time' column; case-insensitive.
        """
        mx_df = mx_df.copy()
        mx_df.columns = [c.strip().lower() for c in mx_df.columns]

        # rename possible variants
        if 'time' in mx_df.columns and 'year' not in mx_df.columns:
            mx_df = mx_df.rename(columns={'time': 'year'})
        if 'sex' not in mx_df.columns or 'mx' not in mx_df.columns:
            raise ValueError(f"obs_mx must contain 'sex' and 'mx' columns. Found: {mx_df.columns.tolist()}")

        # restrict to target year if column exists
        if 'year' in mx_df.columns:
            mx_df = mx_df[mx_df['year'].astype(float).round() == float(year)]

        out = []
        for sex in ['male', 'female']:
            sub = mx_df[mx_df['sex'].str.lower() == sex].set_index('age').reindex(range(101)).fillna(0)
            mx = sub['mx'].values
            qx = 1 - np.exp(-mx)
            lx = np.zeros_like(mx)
            lx[0] = 1.0
            for a in range(100):
                lx[a+1] = lx[a] * (1 - qx[a])
            out.append(pd.DataFrame({'age': np.arange(101), 'sex': sex.title(), 'lx': lx}))
        return pd.concat(out, ignore_index=True)

    obs_mx = mi.load_un_mx_from_wide(mx_csv_path=mx_path, year=inityear, max_age=100)

    obs_lt = life_table_from_mx(obs_mx, year=plot_year)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sexes = ['Male', 'Female']
    colors = {'Male': male_c, 'Female': female_c}

    for ax, sex in zip(axes, sexes):
        # Observed data (UN) dots
        un_df = obs_lt[obs_lt['sex'] == sex]
        ax.scatter(un_df['age'], un_df['lx'], color=colors[sex],
                s=25, alpha=0.8, marker='o', label='Observed data (UN)')

        # Simulation: observed (solid) and idealized (dashed)
        for scenario, style in [('Observed (With diseases)', '-'),
                                ('Idealized (No diseases)', '--')]:
            sim_df = df_surv[(df_surv['sex'] == sex) &
                            (df_surv['year'] == plot_year) &
                            (df_surv['scenario'] == scenario)]
            if len(sim_df):
                ax.plot(sim_df['age'], sim_df['survival'], lw=3,
                        color=colors[sex], ls=style,
                        label=scenario.replace('(With diseases)', 'Sim Observed')
                                    .replace('(No diseases)', 'Sim Idealized'))

        ax.set_title(sex, fontsize=16)
        ax.set_xlabel('Age (years)', fontsize=13)
        ax.grid(alpha=0.3, linestyle=':')
    axes[0].set_ylabel('Survival probability $l(x)$', fontsize=13)
    axes[0].legend(frameon=False, loc='upper right')
    fig.suptitle(f'Survivorship $l(x)$ — Eswatini {plot_year}', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('Figures/Fig1A_survivorship_by_sex_with_UN.png', dpi=300)
    plt.show()
