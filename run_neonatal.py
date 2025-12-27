"""
figure1.py — Produce Figure 1 for the MIGHTI–Eswatini paper
Realized vs. idealized survival (panel A) and life expectancy at birth, 2007–2022 (panel B).
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
# healthconditions = [condition for condition in df.condition if condition not in ["HIV", "TB", "HPV", "Flu", "ViralHepatitis"]]
healthconditions = ['Type2Diabetes']
# Neonatal/congenital diseases
neonatal_disease = [
    'NeonatalSepsis',
    'NeonatalEncephalopathy', 
    'NeonatalJaundice',
    'NeonatalPretermBirth',
    'CongenitalHeartAnomalies',
    'CongenitalMusculoskeletal',
    'DigestiveCongenitalAnomalies',
    'ChromosomalAbnormalities',
]
diseases = ["HIV"] + healthconditions + neonatal_disease

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
infant_deaths_analyzer = mi.InfantDeathsAnalyzer()

death_cause_analyzer = mi.ConditionAtDeathAnalyzer(
    conditions=healthconditions + neonatal_disease)

analyzers = [deaths_analyzer, survivorship_analyzer, death_cause_analyzer, infant_deaths_analyzer]

survivorship_analyzer2 = mi.SurvivorshipAnalyzer()
deaths_analyzer2 = mi.DeathsByAgeSexAnalyzer()
infant_deaths_analyzer2 = mi.InfantDeathsAnalyzer()

death_cause_analyzer2 = mi.ConditionAtDeathAnalyzer(
    conditions=healthconditions + neonatal_disease)

analyzers2 = [deaths_analyzer2, survivorship_analyzer2, death_cause_analyzer2, infant_deaths_analyzer2]

# ---------------------------------------------------------------------
# Demographics & networks
# ---------------------------------------------------------------------
death_rates = {"death_rate": pd.read_csv(csv_path_death), "rate_units": 1}
death = ss.Deaths(death_rates)
death2 = ss.Deaths(death_rates)

# death_rates = pd.read_csv(csv_path_death)
# death = DeathsExtended(death_rate=death_rates, rate_units=1)
# death2 = DeathsExtended(death_rate=death_rates, rate_units=1)

fertility_rate = {"fertility_rate": pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rate)
pregnancy2 = ss.Pregnancy(pars=fertility_rate)

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
networks2 = [maternal, structuredsexual]

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

# Neonatal/congenital diseases (acquired at birth, age-dependent mortality)
for disease in neonatal_disease:
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
        dt=1/12,  # Monthly timesteps (1/12 year) to capture neonatal deaths within 28 days
        people=ppl1,
        networks=networks,
        demographics=[pregnancy, death],
        diseases=disease_objects,      
        connectors=connectors,
        analyzers=analyzers,
        copy_inputs=False,
        label="Realized (With diseases)"
    )

    sim_without = ss.Sim(
        n_agents=n_agents,
        start=inityear,
        stop=endyear,
        dt=1/12,  # Monthly timesteps (1/12 year) to match sim_with
        people=ppl2,
        networks=networks2,
        demographics=[pregnancy2, death2],
        # diseases=disease_objects,      
        # connectors=connectors,
        analyzers=analyzers2,
        copy_inputs=False,
        label="Idealized (No diseases)"
)    

    # Run
    msim = ss.MultiSim([sim_with, sim_without])
    msim.run()  # Run both simulations for comparison

    # Use sim_with for detailed analysis
    sim_with = msim.sims[0]  # Get the first sim from msim after running
    sim_without = msim.sims[1]  # Get the second sim from msim after running
    
    ppl = sim_with.people
    n_people = len(ppl)
    print(f"Total population size: {n_people}")
    print("Deaths requested:", np.isfinite(ppl.ti_dead.raw).sum())
    print("\nNOTE: Starsim removes dead people from the active population.")
    print("      So ppl.dead will show 0 after simulation, but deaths ARE being finalized correctly.")
    print("      Check the analyzer results to see deaths that were tracked during simulation.\n")
    
    # Check dead state - try both .raw and direct access
    dead_raw = ppl.dead.raw if hasattr(ppl.dead, 'raw') else None
    if dead_raw is not None:
        # Only check within valid population bounds
        dead_raw_valid = dead_raw[:n_people] if len(dead_raw) >= n_people else dead_raw
        print("Deaths finalized (raw, valid range):", np.sum(dead_raw_valid))
        print("Deaths finalized (raw, total):", np.sum(dead_raw))
    else:
        print("Deaths finalized (raw): N/A - no .raw attribute")
        dead_raw_valid = None
    
    # Try direct access
    try:
        dead_direct = np.array(ppl.dead, dtype=bool)
        print("Deaths finalized (direct array):", np.sum(dead_direct))
    except Exception as e:
        print(f"Deaths finalized (direct array): Error - {e}")
        dead_direct = None
    
    # Try .sum() method
    try:
        dead_sum = ppl.dead.sum()
        print("Deaths finalized (.sum()):", dead_sum)
    except Exception as e:
        print(f"Deaths finalized (.sum()): Error - {e}")
    
    # Try to get dead UIDs
    try:
        if hasattr(ppl.dead, 'uids'):
            dead_uids = ppl.dead.uids
            print("Dead UIDs count (via .uids):", len(dead_uids))
        elif dead_raw_valid is not None:
            dead_uids = np.where(dead_raw_valid)[0]
            print("Dead UIDs count (via np.where on raw):", len(dead_uids))
        else:
            dead_uids = np.array([], dtype=int)
            print("Dead UIDs count: 0 (no valid access method)")
        
        if len(dead_uids) > 0:
            # Ensure UIDs are within bounds
            dead_uids = dead_uids[dead_uids < n_people]
            if len(dead_uids) > 0:
                early = ppl.age[dead_uids]
                print("Deaths <1 year:", np.sum(early < 1))
                print("Deaths <5 years:", np.sum(early < 5))
            else:
                print("Deaths <1 year: 0 (no valid UIDs)")
                print("Deaths <5 years: 0")
        else:
            print("Deaths <1 year: 0")
            print("Deaths <5 years: 0")
    except Exception as e:
        print(f"Error accessing dead UIDs: {e}")
        import traceback
        traceback.print_exc()

    # Print neonatal disease information
    print("\n" + "="*80)
    print("NEONATAL/CONGENITAL DISEASES")
    print("="*80)
    for d in sim_with.diseases():
        name = getattr(d, "disease_name", d.__class__.__name__)
        if name in neonatal_disease or "Neonatal" in name or "Congenital" in name:
            n_affected = np.count_nonzero(d.affected) if hasattr(d, 'affected') else 0
            print(f"\n{name}:")
            print(f"  - Affected: {n_affected}")
            if hasattr(d, 'results'):
                total_deaths = d.results.new_deaths.sum() if 'new_deaths' in d.results else 0
                neonatal_deaths = d.results.neonatal_deaths.sum() if 'neonatal_deaths' in d.results else 0
                infant_deaths = d.results.infant_deaths.sum() if 'infant_deaths' in d.results else 0
                later_deaths = d.results.later_deaths.sum() if 'later_deaths' in d.results else 0
                print(f"  - Total deaths: {total_deaths}")
                print(f"  - Neonatal deaths (<28 days): {neonatal_deaths}")
                print(f"  - Infant deaths (28d-1yr): {infant_deaths}")
                print(f"  - Later deaths (>=1yr): {later_deaths}")

    # Print analyzer results
    print("\n" + "="*80)
    print("DEATHS BY AGE/SEX ANALYZER")
    print("="*80)
    # Get deaths analyzer using the helper function
    a = get_deaths_module(sim_with)
    print("Male deaths by age (first 5):", a.results.male_deaths_by_age[:5])
    print("Female deaths by age (first 5):", a.results.female_deaths_by_age[:5])
    print("Total deaths tracked by analyzer:", a.results.male_deaths_by_age.sum() + a.results.female_deaths_by_age.sum())
    print("Infant deaths (age < 1):", a.results.infant_deaths[-1] if len(a.results.infant_deaths) > 0 else 0)
    
    # Print infant deaths analyzer results
    print("\n" + "="*80)
    print("INFANT DEATHS ANALYZER")
    print("="*80)
    # Try different access patterns for the analyzer
    infant_an = None
    if hasattr(sim_with, 'analyzers'):
        if isinstance(sim_with.analyzers, dict):
            # Try to find by name or label
            for key, analyzer in sim_with.analyzers.items():
                if isinstance(analyzer, mi.InfantDeathsAnalyzer) or 'infant' in key.lower():
                    infant_an = analyzer
                    break
        elif hasattr(sim_with.analyzers, 'infantdeathsanalyzer'):
            infant_an = sim_with.analyzers.infantdeathsanalyzer
    
    if infant_an is not None:
        print(f"Total neonatal deaths (male): {infant_an.results.neonatal_deaths_male.sum() if 'neonatal_deaths_male' in infant_an.results else 'N/A'}")
        print(f"Total neonatal deaths (female): {infant_an.results.neonatal_deaths_female.sum() if 'neonatal_deaths_female' in infant_an.results else 'N/A'}")
        print(f"Total infant deaths (male): {infant_an.results.infant_deaths_male.sum() if 'infant_deaths_male' in infant_an.results else 'N/A'}")
        print(f"Total infant deaths (female): {infant_an.results.infant_deaths_female.sum() if 'infant_deaths_female' in infant_an.results else 'N/A'}")
    else:
        print("InfantDeathsAnalyzer not found in analyzers")
        print(f"Available analyzers: {list(sim_with.analyzers.keys()) if isinstance(sim_with.analyzers, dict) else 'N/A'}")

    print("Survivorship (lx_male, first 5):", sim_with.analyzers.survivorship_analyzer.results['lx_male'][:5])
    print("Survivorship (lx_female, first 5):", sim_with.analyzers.survivorship_analyzer.results['lx_female'][:5] if 'lx_female' in sim_with.analyzers.survivorship_analyzer.results else 'N/A')

    # Calculate life expectancy
    print("\n" + "="*80)
    print("LIFE EXPECTANCY CALCULATION")
    print("="*80)
    try:
        le_results = mi.calculate_life_expectancy(sim_with, year=endyear, max_age=100, radix=n_agents)
        print(f"Life expectancy at birth (Male): {le_results['Male']:.2f} years")
        print(f"Life expectancy at birth (Female): {le_results['Female']:.2f} years")
        print(f"Life expectancy at birth (Both): {le_results['Both']:.2f} years")
    except Exception as e:
        print(f"Error calculating life expectancy: {e}")
        import traceback
        traceback.print_exc()
    
    # Calculate mortality rates for plotting
    # Get deaths analyzer from the simulation (after msim.run())
    deaths_analyzer_actual = get_deaths_module(sim_with)
    df_mx = mi.calculate_mortality_rates(sim_with, deaths_analyzer_actual)
    
    for d in sim_with.diseases():
        name = getattr(d, "disease_name", d.__class__.__name__)
        print(
            f"{name:<30} "
            f"step_die={hasattr(d, 'step_die')}, "
            f"ti_dead={hasattr(d, 'ti_dead')}"
        )

    # ppl = sim_with.people
    ti_dead_raw = ppl.ti_dead.raw
    print("Deaths requested:", np.count_nonzero(np.isfinite(ti_dead_raw)))
    print("Deaths finalized:", int(np.sum(ppl.dead)))

    # # print([k for k in dir(ppl) if "death" in k.lower() or "pending" in k.lower()])  
    
    # # print("Alive before manual call:", ppl.alive.sum())
    # # ppl.step_die()
    # # print("Alive after manual call:", ppl.alive.sum())
    # # print("Dead after manual call:", ppl.dead.sum())

    # a = (ppl.ti_dead <= ppl.sim.ti)
    # print(type(a), getattr(a, "uids", None))
    # print("Example:", np.count_nonzero(a), "vs expected", np.count_nonzero(np.isfinite(ppl.ti_dead.raw)))
    
    
    # # After sim.run()
    # print("Total people dead:", sim_with.people.dead.sum())
    # print("Infant deaths recorded by analyzer:", sim_with.analyzers.deathsbyagesexanalyzer.results.infant_deaths[-1])
    # print("Sample ConditionAtDeath records (first 10):")
    # print(sim_with.analyzers.condition_at_death_analyzer.to_df().head(10))
    # df_test = sim_with.analyzers['survivorship_analyzer'].to_df()
    # print(f"\n{sim_with.label} survivorship DataFrame:")
    # print(df_test.head(10))
    # print(df_test.tail(5))
    # print(msim.sims[0].analyzers.condition_at_death_analyzer.to_df().head(10))

    #     # After sim.run()
    # print("Total people dead:", msim.sims[1].people.dead.sum())
    # print("Infant deaths recorded by analyzer:", msim.sims[1].analyzers.deathsbyagesexanalyzer.results.infant_deaths[-1])
    # print("Sample ConditionAtDeath records (first 10):")
    # print(msim.sims[1].analyzers.condition_at_death_analyzer.to_df().head(10))

    # for sim in msim.sims:
    #     df_test = sim.analyzers['survivorship_analyzer'].to_df()
    #     print(f"\n{sim.label} survivorship DataFrame:")
    #     print(df_test.head(10))
    #     print(df_test.tail(5))

    # ---------------------------------------------------------------------
    # Figure 1 – Realized vs Idealized survival and life expectancy
    # ---------------------------------------------------------------------
    # Extract survivorship data from both sims
    dfs_surv = []
    for sim in msim.sims:
        surv_df = sim.analyzers['survivorship_analyzer'].to_df()
        surv_df['scenario'] = sim.label
        dfs_surv.append(surv_df)
    df_surv = pd.concat(dfs_surv)


    # ---------------------------------------------------------------------
    # Panel A – Survivorship by sex
    # ---------------------------------------------------------------------
    male_c, female_c = "#538DD5", "#B23A48"
    plot_year = endyear

    # Try to load observed data if available
    try:
        obs_mx = mi.load_un_mx_from_wide(mx_csv_path=mx_path, year=inityear, max_age=100)
        
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

        obs_lt = life_table_from_mx(obs_mx, year=plot_year)
        has_obs_data = True
    except Exception as e:
        print(f"Could not load observed data: {e}")
        has_obs_data = False
        obs_lt = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sexes = ['Male', 'Female']
    colors = {'Male': male_c, 'Female': female_c}

    for ax, sex in zip(axes, sexes):
        # Observed data (UN) dots if available
        if has_obs_data and obs_lt is not None:
            un_df = obs_lt[obs_lt['sex'] == sex]
            ax.scatter(un_df['age'], un_df['lx'], color=colors[sex],
                    s=25, alpha=0.8, marker='o', label='Observed data (UN)')

        # Simulation: realized (solid) and idealized (dashed)
        for scenario, style in [('Realized (With diseases)', '-'),
                                ('Idealized (No diseases)', '--')]:
            sim_df = df_surv[(df_surv['sex'] == sex) &
                            (df_surv['year'] == plot_year) &
                            (df_surv['scenario'] == scenario)]
            if len(sim_df):
                ax.plot(sim_df['age'], sim_df['survival'], lw=3,
                        color=colors[sex], ls=style,
                        label=scenario.replace('(With diseases)', 'Sim Realized')
                                    .replace('(No diseases)', 'Sim Idealized'))

        ax.set_title(sex, fontsize=16)
        ax.set_xlabel('Age (years)', fontsize=13)
        ax.grid(alpha=0.3, linestyle=':')
    axes[0].set_ylabel('Survival probability $l(x)$', fontsize=13)
    axes[0].legend(frameon=False, loc='upper right')
    fig.suptitle(f'Survivorship $l(x)$ — Eswatini {plot_year}', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Create Figures directory if it doesn't exist
    import os
    os.makedirs('Figures', exist_ok=True)
    plt.savefig('Figures/Fig1A_survivorship_by_sex_with_UN.png', dpi=300)
    print("\nSaved: Figures/Fig1A_survivorship_by_sex_with_UN.png")
    
    # ---------------------------------------------------------------------
    # Panel B – Life Expectancy over time
    # ---------------------------------------------------------------------
    print("\n" + "="*80)
    print("CALCULATING LIFE EXPECTANCY OVER TIME")
    print("="*80)
    
    # Calculate life expectancy for both simulations at the final year
    # Note: For a time series, we'd need to calculate at multiple time points,
    # but for now we'll show a comparison at the final year
    le_by_year = []
    for sim in msim.sims:
        try:
            deaths_mod = get_deaths_module(sim)
            le = mi.calculate_life_expectancy(sim, year=sim.t.year, max_age=100, radix=n_agents)
            le_by_year.append({
                'year': sim.t.year,
                'scenario': sim.label,
                'Male': le['Male'],
                'Female': le['Female'],
                'Both': le['Both']
            })
            print(f"{sim.label}: Male={le['Male']:.2f}, Female={le['Female']:.2f}, Both={le['Both']:.2f}")
        except Exception as e:
            print(f"Error calculating LE for {sim.label}: {e}")
            import traceback
            traceback.print_exc()
    
    if len(le_by_year) > 0:
        le_df = pd.DataFrame(le_by_year)
        
        # Plot life expectancy as a bar chart or comparison plot
        # Since we only have one time point, use a bar chart or comparison plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Prepare data for plotting
        scenarios = le_df['scenario'].unique()
        x_pos = np.arange(len(scenarios))
        width = 0.35
        
        male_values = [le_df[le_df['scenario'] == sc]['Male'].values[0] for sc in scenarios]
        female_values = [le_df[le_df['scenario'] == sc]['Female'].values[0] for sc in scenarios]
        
        # Create bar chart
        bars1 = ax.bar(x_pos - width/2, male_values, width, label='Male', color=male_c, alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, female_values, width, label='Female', color=female_c, alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Scenario', fontsize=13)
        ax.set_ylabel('Life Expectancy at Birth $e_0$ (years)', fontsize=13)
        ax.set_title(f'Life Expectancy at Birth — Eswatini {endyear}', fontsize=16)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([sc.replace('Realized (With diseases)', 'Realized').replace('Idealized (No diseases)', 'Idealized') 
                            for sc in scenarios], rotation=15, ha='right')
        ax.grid(alpha=0.3, linestyle=':', axis='y')
        ax.legend(frameon=False, loc='upper left')
        plt.tight_layout()
        plt.savefig('Figures/Fig1B_life_expectancy_over_time.png', dpi=300)
        print("Saved: Figures/Fig1B_life_expectancy_over_time.png")
        
        # Also save to CSV
        le_df.to_csv('Figures/life_expectancy_results.csv', index=False)
        print("Saved: Figures/life_expectancy_results.csv")
    
    plt.show()
