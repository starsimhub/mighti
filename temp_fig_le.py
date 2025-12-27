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
healthconditions = [condition for condition in df.condition if condition not in ["HIV", "TB", "HPV", "Flu", "ViralHepatitis"]]
# healthconditions = ['MajorDepressiveDisorder', 'Type2Diabetes', 'ChronicKidneyDisease']
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

# Create separate network objects for sim_without
maternal2 = ss.MaternalNet()
structuredsexual2 = sti.StructuredSexual()
networks2 = [maternal2, structuredsexual2]

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
# All diseases (including neonatal/congenital) should use age/sex dependent prevalence data
# via make_init_prev_func() - init_prev is never read from CSV
for disease in healthconditions:
    disease_class = getattr(mi, disease, None)
    if disease_class:
        # All diseases use age/sex dependent prevalence data for init_prev
        init_prev = ss.bernoulli(p=make_init_prev_func(disease))
        disease_obj = disease_class(csv_path=csv_path_params, pars={"init_prev": init_prev})
        disease_objects.append(disease_obj)

# TEST: Override neonatal disease parameters with extreme values to verify mechanism works
# This is a temporary test - remove after verification
# COMMENTED OUT: Using real parameters from CSV instead
# neonatal_diseases = ['NeonatalEncephalopathy', 'NeonatalPretermBirth', 'NeonatalSepsis', 'NeonatalJaundice']
# for disease_obj in disease_objects:
#     if hasattr(disease_obj, 'disease_name') and disease_obj.disease_name in neonatal_diseases:
#         print(f"[TEST] Overriding {disease_obj.disease_name} with extreme test values: init_prev=0.1 (10%), p_death=0.5 (50%)")
#         disease_obj.pars.init_prev = ss.bernoulli(p=0.1)  # 10% of babies affected (very high)
#         disease_obj.pars.p_death = ss.bernoulli(p=0.5)   # 50% death rate (very high)


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
        label="Realized (With diseases)"
    )

    sim_without = ss.Sim(
        n_agents=n_agents,
        start=inityear,
        stop=endyear,
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
    msim.run()

    # Manually finalize analyzers (fixes the RuntimeWarning and ensures normalization)
    for sim in msim.sims:
        for analyzer in sim.analyzers.values():
            if hasattr(analyzer, 'finalize'):
                analyzer.finalize()
            if hasattr(analyzer, 'finalize_results'):
                analyzer.finalize_results()

    ppl = sim_with.people
    print("Deaths requested:", np.isfinite(ppl.ti_dead.raw).sum())
    # Use .raw to access dead array (BoolState might filter dead people)
    dead_raw = ppl.dead.raw if hasattr(ppl.dead, 'raw') else ppl.dead
    print("Deaths finalized (using .raw):", int(np.sum(dead_raw)))
    
    # Get ages for dead people using .raw
    age_raw = ppl.age.raw if hasattr(ppl.age, 'raw') else ppl.age
    dead_uids = np.where(dead_raw)[0]
    if len(dead_uids) > 0:
        early_ages = age_raw[dead_uids] if len(age_raw) > max(dead_uids) else np.array([])
        print("Deaths <1 year:", np.sum(early_ages < 1) if len(early_ages) > 0 else 0)
        print("Deaths <5 years:", np.sum(early_ages < 5) if len(early_ages) > 0 else 0)
    else:
        print("Deaths <1 year: 0")
        print("Deaths <5 years: 0")
    print("Deaths total:", int(np.sum(dead_raw)))

    for d in sim_with.modules.values():
        if "Neonatal" in d.name or "Congenital" in d.name:
            print(d.disease_name, d.pars.p_death)

    a = sim_with.analyzers.deathsbyagesexanalyzer
    print(a.results.male_deaths_by_age[:5])
    print(a.results.female_deaths_by_age[:5])

    print(sim_with.analyzers.survivorship_analyzer.results['lx_male'][:5])

    # df_mx = mi.calculate_mortality_rates(sim_with, deaths_analyzer)
    # plt.plot(df_mx[df_mx.sex=="Male"].age, df_mx[df_mx.sex=="Male"].mx)
    # sim_with.run()

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

    # print([k for k in dir(ppl) if "death" in k.lower() or "pending" in k.lower()])  
    
    # print("Alive before manual call:", ppl.alive.sum())
    # ppl.step_die()
    # print("Alive after manual call:", ppl.alive.sum())
    # print("Dead after manual call:", ppl.dead.sum())

    a = (ppl.ti_dead <= ppl.sim.ti)
    print(type(a), getattr(a, "uids", None))
    print("Example:", np.count_nonzero(a), "vs expected", np.count_nonzero(np.isfinite(ppl.ti_dead.raw)))
    
    
    # After sim.run()
    # Use .raw to access dead array (BoolState might filter dead people)
    dead_raw_with = sim_with.people.dead.raw if hasattr(sim_with.people.dead, 'raw') else sim_with.people.dead
    print("Total people dead (using .raw):", int(np.sum(dead_raw_with)))
    print("Infant deaths recorded by analyzer:", sim_with.analyzers.deathsbyagesexanalyzer.results.infant_deaths[-1])
    print("Sample ConditionAtDeath records (first 10):")
    print(sim_with.analyzers.condition_at_death_analyzer.to_df().head(10))
    df_test = sim_with.analyzers['survivorship_analyzer'].to_df()
    print(f"\n{sim_with.label} survivorship DataFrame:")
    print(df_test.head(10))
    print(df_test.tail(5))
    print(msim.sims[0].analyzers.condition_at_death_analyzer.to_df().head(10))

    # After sim.run()
    # Use .raw to access dead array (BoolState might filter dead people)
    dead_raw_without = sim_without.people.dead.raw if hasattr(sim_without.people.dead, 'raw') else sim_without.people.dead
    print("Total people dead (using .raw):", int(np.sum(dead_raw_without)))
    print("Infant deaths recorded by analyzer:", sim_without.analyzers.deathsbyagesexanalyzer.results.infant_deaths[-1])
    print("Sample ConditionAtDeath records (first 10):")
    print(sim_without.analyzers.condition_at_death_analyzer.to_df().head(10))

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

    # ---------------------------------------------------------------------
    # Back-calculate required infant mortality from observed data
    # ---------------------------------------------------------------------
    print("\n" + "="*80)
    print("INFANT MORTALITY ANALYSIS: Observed vs. Simulated")
    print("="*80)
    
    for sex in ['Male', 'Female']:
        obs_sex = obs_lt[obs_lt['sex'] == sex]
        obs_l0 = obs_sex[obs_sex['age'] == 0]['lx'].values[0]
        obs_l1 = obs_sex[obs_sex['age'] == 1]['lx'].values[0]
        
        # Calculate observed infant mortality rate (q0)
        obs_q0 = 1.0 - obs_l1  # Deaths per 1000 live births
        obs_imr_per_1000 = obs_q0 * 1000
        
        # For a cohort of 100,000 (typical simulation size)
        cohort_size = 100000
        required_infant_deaths = obs_q0 * cohort_size
        
        # Get simulated values
        sim_realized = df_surv[(df_surv['sex'] == sex) & 
                              (df_surv['scenario'] == 'Realized (With diseases)') &
                              (df_surv['year'] == plot_year)]
        sim_idealized = df_surv[(df_surv['sex'] == sex) & 
                               (df_surv['scenario'] == 'Idealized (No diseases)') &
                               (df_surv['year'] == plot_year)]
        
        sim_realized_l1 = sim_realized[sim_realized['age'] == 1]['survival'].values[0] if len(sim_realized) > 0 else None
        sim_idealized_l1 = sim_idealized[sim_idealized['age'] == 1]['survival'].values[0] if len(sim_idealized) > 0 else None
        
        # Get actual infant deaths from simulation
        sim_with = msim.sims[1] if len(msim.sims) > 1 else msim.sims[0]
        actual_infant_deaths = sim_with.analyzers.deathsbyagesexanalyzer.results.infant_deaths[-1] if hasattr(sim_with.analyzers, 'deathsbyagesexanalyzer') else 0
        
        print(f"\n{sex}:")
        print(f"  Observed data (UN):")
        print(f"    l(0) = {obs_l0:.6f}, l(1) = {obs_l1:.6f}")
        print(f"    Infant mortality rate (q0) = {obs_q0:.6f} ({obs_imr_per_1000:.2f} per 1000)")
        print(f"    Required infant deaths (cohort of {cohort_size:,}) = {required_infant_deaths:.0f}")
        
        if sim_realized_l1 is not None:
            sim_realized_q0 = 1.0 - sim_realized_l1
            sim_realized_imr_per_1000 = sim_realized_q0 * 1000
            print(f"  Simulated (Realized):")
            print(f"    l(1) = {sim_realized_l1:.6f}")
            print(f"    Infant mortality rate (q0) = {sim_realized_q0:.6f} ({sim_realized_imr_per_1000:.2f} per 1000)")
            print(f"    Actual infant deaths recorded = {actual_infant_deaths:.0f}")
            print(f"    Gap: Need {required_infant_deaths - actual_infant_deaths:.0f} more infant deaths")
            print(f"    Gap as % of required: {(required_infant_deaths - actual_infant_deaths) / required_infant_deaths * 100:.1f}%")
        
        if sim_idealized_l1 is not None:
            sim_idealized_q0 = 1.0 - sim_idealized_l1
            sim_idealized_imr_per_1000 = sim_idealized_q0 * 1000
            print(f"  Simulated (Idealized):")
            print(f"    l(1) = {sim_idealized_l1:.6f}")
            print(f"    Infant mortality rate (q0) = {sim_idealized_q0:.6f} ({sim_idealized_imr_per_1000:.2f} per 1000)")
    
    # ---------------------------------------------------------------------
    # Calculate required parameters for neonatal diseases
    # ---------------------------------------------------------------------
    print("\n" + "="*80)
    print("REQUIRED PARAMETERS TO MATCH OBSERVED INFANT MORTALITY")
    print("="*80)
    
    # Get average across sexes for simplicity
    avg_required_deaths = 0
    avg_actual_deaths = 0
    for sex in ['Male', 'Female']:
        obs_sex = obs_lt[obs_lt['sex'] == sex]
        obs_l1 = obs_sex[obs_sex['age'] == 1]['lx'].values[0]
        obs_q0 = 1.0 - obs_l1
        avg_required_deaths += obs_q0 * 100000 / 2
        
        sim_realized = df_surv[(df_surv['sex'] == sex) & 
                              (df_surv['scenario'] == 'Realized (With diseases)') &
                              (df_surv['year'] == plot_year)]
        if len(sim_realized) > 0:
            sim_realized_l1 = sim_realized[sim_realized['age'] == 1]['survival'].values[0]
            sim_realized_q0 = 1.0 - sim_realized_l1
            avg_actual_deaths += sim_realized_q0 * 100000 / 2
    
    gap_deaths = avg_required_deaths - avg_actual_deaths
    
    print(f"\nAverage across sexes (cohort of 100,000):")
    print(f"  Required infant deaths: {avg_required_deaths:.0f}")
    print(f"  Current infant deaths: {avg_actual_deaths:.0f}")
    print(f"  Gap: {gap_deaths:.0f} additional deaths needed")
    
    # Get actual births and infant deaths from simulation
    sim_with = msim.sims[1] if len(msim.sims) > 1 else msim.sims[0]
    n_agents = len(sim_with.people) if hasattr(sim_with, 'people') else 100000
    
    # Try to get actual births from pregnancy module
    actual_annual_births = None
    if hasattr(sim_with, 'demographics'):
        for demo in sim_with.demographics:
            if hasattr(demo, 'name') and 'pregnancy' in demo.name.lower():
                # Try to get births from pregnancy module
                if hasattr(demo, 'results') and hasattr(demo.results, 'births'):
                    total_births = np.sum(demo.results.births) if hasattr(demo.results.births, '__len__') else 0
                    n_years = len(sim_with.yearvec) - 1 if hasattr(sim_with, 'yearvec') else 18
                    actual_annual_births = total_births / n_years if n_years > 0 else 0
                    break
    
    if actual_annual_births is None:
        # Estimate annual births (roughly 2-3% of population for Eswatini)
        annual_birth_rate = 0.025  # 2.5% annual birth rate (approximate for Eswatini)
        estimated_annual_births = n_agents * annual_birth_rate
        print(f"\nEstimated annual births in simulation: {estimated_annual_births:.0f}")
        print(f"  (Assuming {annual_birth_rate*100:.1f}% annual birth rate)")
        annual_births = estimated_annual_births
    else:
        print(f"\nActual annual births in simulation: {actual_annual_births:.0f}")
        annual_births = actual_annual_births
    
    # Get all infant deaths from ConditionAtDeath analyzer
    all_infant_deaths_by_cause = {}
    if hasattr(sim_with, 'analyzers') and 'condition_at_death_analyzer' in sim_with.analyzers:
        cond_at_death_df = sim_with.analyzers['condition_at_death_analyzer'].to_df()
        infant_deaths_df = cond_at_death_df[cond_at_death_df['age'] < 1.0].copy()
        
        print(f"\nInfant deaths (<1 year) by cause (from ConditionAtDeath analyzer):")
        print(f"  Total infant deaths recorded: {len(infant_deaths_df)}")
        
        # Count by primary cause
        if 'primary_cause' in infant_deaths_df.columns:
            cause_counts = infant_deaths_df['primary_cause'].value_counts()
            for cause, count in cause_counts.items():
                print(f"    {cause}: {count}")
                all_infant_deaths_by_cause[cause] = count
        
        # Also check all disease columns
        disease_cols = [col for col in infant_deaths_df.columns if col.startswith('cause_')]
        for col in disease_cols:
            disease_name = col.replace('cause_', '')
            deaths_from_disease = infant_deaths_df[infant_deaths_df[col] == True]
            if len(deaths_from_disease) > 0:
                if disease_name not in all_infant_deaths_by_cause:
                    all_infant_deaths_by_cause[disease_name] = 0
                all_infant_deaths_by_cause[disease_name] += len(deaths_from_disease)
    
    # Calculate required neonatal disease parameters
    # We have 4 neonatal diseases: NeonatalEncephalopathy, NeonatalPretermBirth, NeonatalSepsis, NeonatalJaundice
    neonatal_disease_names = ['NeonatalEncephalopathy', 'NeonatalPretermBirth', 'NeonatalSepsis', 'NeonatalJaundice']
    
    print(f"\nCurrent neonatal disease parameters:")
    current_total_affected = 0
    current_total_deaths = 0
    
    for disease_name in neonatal_disease_names:
        for disease_obj in disease_objects:
            if hasattr(disease_obj, 'disease_name') and disease_obj.disease_name == disease_name:
                try:
                    # Extract init_prev value
                    if hasattr(disease_obj.pars.init_prev, 'pars'):
                        init_prev_val = disease_obj.pars.init_prev.pars.get("p", 0)
                    elif callable(disease_obj.pars.init_prev):
                        # It's a function/lambda - can't extract value, skip this disease
                        print(f"  {disease_name}: init_prev is a function, skipping parameter calculation")
                        break
                    else:
                        init_prev_val = float(disease_obj.pars.init_prev)
                    
                    # Extract p_death value
                    if hasattr(disease_obj.pars.p_death, 'pars'):
                        p_death_val = disease_obj.pars.p_death.pars.get("p", 0)
                    elif callable(disease_obj.pars.p_death):
                        # It's a function/lambda - can't extract value, skip this disease
                        print(f"  {disease_name}: p_death is a function, skipping parameter calculation")
                        break
                    else:
                        p_death_val = float(disease_obj.pars.p_death)
                    
                    # Ensure we have numeric values
                    init_prev_val = float(init_prev_val) if not callable(init_prev_val) else 0.0
                    p_death_val = float(p_death_val) if not callable(p_death_val) else 0.0
                    
                    expected_affected = annual_births * init_prev_val
                    expected_deaths = expected_affected * p_death_val
                    current_total_affected += expected_affected
                    current_total_deaths += expected_deaths
                    
                    print(f"  {disease_name}:")
                    print(f"    init_prev = {init_prev_val:.6f} ({init_prev_val*100:.4f}%)")
                    print(f"    p_death = {p_death_val:.6f} ({p_death_val*100:.4f}%)")
                    print(f"    Expected affected per year: {expected_affected:.0f}")
                    print(f"    Expected deaths per year: {expected_deaths:.0f}")
                except Exception as e:
                    print(f"  {disease_name}: Could not read parameters ({e})")
                break
    
    print(f"\nCurrent total expected neonatal deaths per year: {current_total_deaths:.0f}")
    
    # Calculate what fraction of infant mortality should come from neonatal diseases
    # Based on GBD data, neonatal causes typically account for ~40-50% of infant mortality
    # The rest comes from other causes (diarrhea, respiratory infections, malnutrition, etc.)
    neonatal_fraction = 0.45  # Assume 45% of infant deaths are from neonatal causes
    required_neonatal_deaths = gap_deaths * neonatal_fraction
    required_other_infant_deaths = gap_deaths * (1 - neonatal_fraction)
    
    print(f"\nInfant mortality breakdown (based on typical GBD patterns):")
    print(f"  Total required infant deaths: {gap_deaths:.0f}")
    print(f"  Required from neonatal diseases (~{neonatal_fraction*100:.0f}%): {required_neonatal_deaths:.0f}")
    print(f"  Required from other causes (~{(1-neonatal_fraction)*100:.0f}%): {required_other_infant_deaths:.0f}")
    print(f"  Current neonatal deaths: {current_total_deaths:.0f}")
    print(f"  Gap in neonatal deaths: {required_neonatal_deaths - current_total_deaths:.0f}")
    
    # Calculate what parameters would be needed for neonatal diseases only
    if current_total_affected > 0:
        # Option 1: Keep init_prev, adjust p_death (for neonatal portion only)
        required_p_death_avg = required_neonatal_deaths / current_total_affected
        print(f"\nTo match observed neonatal mortality (keeping current init_prev):")
        print(f"  Required average p_death = {required_p_death_avg:.6f} ({required_p_death_avg*100:.4f}%)")
        print(f"  (Current average p_death ≈ {current_total_deaths/current_total_affected:.6f})")
        
        # Option 2: Keep p_death, adjust init_prev
        avg_p_death = current_total_deaths / current_total_affected if current_total_affected > 0 else 0.5
        required_init_prev_avg = required_neonatal_deaths / (annual_births * avg_p_death)
        print(f"\nTo match observed neonatal mortality (keeping current p_death):")
        print(f"  Required average init_prev = {required_init_prev_avg:.6f} ({required_init_prev_avg*100:.4f}%)")
        print(f"  (Current average init_prev ≈ {current_total_affected/annual_births:.6f})")
    
    print(f"\nNOTE: The remaining {required_other_infant_deaths:.0f} infant deaths should come from:")
    print(f"  - DiarrhealDisease")
    print(f"  - LowerRespiratoryInfections")
    print(f"  - ProteinEnergyMalnutrition")
    print(f"  - Other congenital diseases")
    print(f"  - Other causes not yet modeled")
    
    print("\n" + "="*80 + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sexes = ['Male', 'Female']
    colors = {'Male': male_c, 'Female': female_c}

    for ax, sex in zip(axes, sexes):
        # Observed data (UN) - use line plot for better visibility
        un_df = obs_lt[obs_lt['sex'] == sex]
        ax.plot(un_df['age'], un_df['lx'], color=colors[sex],
                lw=2, alpha=0.7, linestyle=':', marker='o', markersize=3,
                label='Observed data (UN)', zorder=3)

        # Simulation: observed (solid) and idealized (dashed)
        for scenario, style in [('Realized (With diseases)', '-'),
                                ('Idealized (No diseases)', '--')]:
            sim_df = df_surv[(df_surv['sex'] == sex) &
                            (df_surv['year'] == plot_year) &
                            (df_surv['scenario'] == scenario)]
            if len(sim_df):
                ax.plot(sim_df['age'], sim_df['survival'], lw=3,
                        color=colors[sex], ls=style, alpha=0.8,
                        label=scenario.replace('(With diseases)', 'Sim Realized')
                                    .replace('(No diseases)', 'Sim Idealized'),
                        zorder=2 if style == '-' else 1)

        # Zoom in on early ages to better see infant mortality
        ax.set_xlim(-0.5, 10)  # Focus on ages 0-10 to see infant mortality clearly
        ax.set_ylim(0.85, 1.02)  # Focus on survival range where infant mortality is visible
        
        ax.set_title(sex, fontsize=16)
        ax.set_xlabel('Age (years)', fontsize=13)
        ax.grid(alpha=0.3, linestyle=':')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    axes[0].set_ylabel('Survival probability $l(x)$', fontsize=13)
    axes[0].legend(frameon=False, loc='lower left', fontsize=10)
    fig.suptitle(f'Survivorship $l(x)$ — Eswatini {plot_year} (Ages 0-10)', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('Figures/Fig1A_survivorship_by_sex_with_UN.png', dpi=300)
    plt.show()
