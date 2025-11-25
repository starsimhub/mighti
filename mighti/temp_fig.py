"""
Figure 7 — Adherence and CASM Demonstration (HIV–AUD Interaction)

This script compares 4 scenarios:
1. No interaction (HIV+AUD, no CASM effects)
2. With interaction (HIV+AUD, CASM reduces adherence)
3. No interaction + AUD care
4. With interaction + AUD care

Plots ART coverage over time stratified by AUD status.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import starsim as ss
import stisim as sti
import mighti as mi
import prepare_data_for_year
import logging

# ---------------------------------------------------------------------
# Logging and seeding
# ---------------------------------------------------------------------
logger = logging.getLogger("MIGHTI")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------
# Simulation setup
# ---------------------------------------------------------------------
n_agents = 100_000
inityear = 2007
endyear = 2020
region = "eswatini"

# File paths
csv_path_params     = f"mighti/data/{region}_parameters.csv"
csv_path_prevalence = f"mighti/data/{region}_prevalence.csv"
csv_path_fertility  = f"mighti/data/{region}_asfr.csv"
csv_path_death      = f"mighti/data/{region}_mortality_rates.csv"
csv_path_age        = f"mighti/data/{region}_age_distribution_{inityear}.csv"
csv_path_intervention = f"mighti/data/{region}_intervention.csv"

# Prepare demographic data (safe even if cached)
prepare_data_for_year.prepare_data_for_year(region, inityear)
prepare_data_for_year.prepare_data(region)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def make_people():
    """Create People object with adherence state."""
    return ss.People(
        n_agents,
        age_data=pd.read_csv(csv_path_age),
        extra_states=[ss.FloatArr("adherence", default=1.0)],
    )


# ---------------------------------------------------------------------
# Load prevalence data
# ---------------------------------------------------------------------
prevalence_data_df = pd.read_csv(csv_path_prevalence)
diseases = ["HIV", "AlcoholUseDisorder"]

prevalence_data, age_bins = mi.initialize_prevalence_data(
    diseases=diseases, prevalence_data=prevalence_data_df, inityear=inityear
)


def make_init_prev_func(disease):
    def prevalence_func(sim, uids, size=None):
        # In Starsim 3.x, uids are the agent IDs, size is optional
        # Convert uids to indices if needed
        if size is None:
            # Use uids directly as indices (they should be 0-indexed)
            size = uids if hasattr(uids, '__len__') else np.arange(len(sim.people))
        
        prev_vals = mi.age_sex_dependent_prevalence(
            disease=disease,
            prevalence_data=prevalence_data,
            age_bins=age_bins,
            sim=sim,
            size=size,
        )
        
        # Debug: print prevalence stats for first call
        if not hasattr(prevalence_func, '_called'):
            print(f"[DEBUG] {disease} prevalence: mean={prev_vals.mean():.4f}, "
                  f"min={prev_vals.min():.4f}, max={prev_vals.max():.4f}, "
                  f"non-zero={np.sum(prev_vals > 0)}/{len(prev_vals)}")
            prevalence_func._called = True
        
        return prev_vals
    return lambda sim, uids, size=None: prevalence_func(sim, uids, size)


# ---------------------------------------------------------------------
# Demography, networks
# ---------------------------------------------------------------------
death = ss.Deaths({"death_rate": pd.read_csv(csv_path_death), "rate_units": 1})
pregnancy = ss.Pregnancy({"fertility_rate": pd.read_csv(csv_path_fertility)})

maternal = ss.MaternalNet()
sexual = sti.StructuredSexual()
networks = [maternal, sexual]


# ---------------------------------------------------------------------
# Disease modules
# ---------------------------------------------------------------------
# Initialize HIV with constant prevalence (HIV data may not be in prevalence CSV)
hiv = sti.HIV(
    beta_m2f=0.955,
    beta_m2c=0.0039,
    init_prev=ss.bernoulli(p=0.15),  # Constant 15% prevalence
)
hiv.pars.include_care = True
hiv.pars.art_efficacy = 0.9
hiv.pars.beta = {
    "structuredsexual": [0.0296, 0.0296],
    "maternal": [0.00112, 0.00112],
}
print(f"[DEBUG] HIV initialized with constant prevalence: ss.bernoulli(p=0.15)")

aud = mi.AlcoholUseDisorder(
    csv_path=csv_path_params,
    pars=dict(init_prev=ss.bernoulli(p=make_init_prev_func("AlcoholUseDisorder")))
)


# ---------------------------------------------------------------------
# AUD Care Intervention (similar to DepressionCare)
# ---------------------------------------------------------------------
class AUDCare(ss.treat_num):
    """AUD treatment intervention (adapted from DepressionCare)."""
    def __init__(self, *args, product=None, prob=1.0, remission_boost=1.5, 
                 eligibility=None, disease="alcoholusedisorder", **kwargs):
        self.disease = disease
        self.remission_boost = remission_boost
        if product is not None and hasattr(product, "df"):
            df = product.df
            if "disease" in df.columns:
                df = df[df["disease"].str.lower() == self.disease]
            product.df = df
        super().__init__(*args, product=product, prob=prob, eligibility=eligibility, **kwargs)
    
    def initialize(self, sim):
        super().initialize(sim)
        if self.eligibility is None:
            if not hasattr(sim.diseases, self.disease):
                raise ValueError(f"[{self.label}] Disease '{self.disease}' not found.")
            self.eligibility = lambda sim: sim.diseases[self.disease].affected.uids
    
    def step(self):
        cur_year = float(self.sim.now)
        condition = self.sim.diseases[self.disease]
        if self.eligibility is None:
            self.eligibility = lambda sim: sim.diseases[self.disease].affected.uids
        eligible = self.eligibility(self.sim)
        n_eligible = len(eligible)
        chooser = (np.random.rand(n_eligible) < self.prob)
        treated = eligible[chooser]
        self.treated_inds = ss.uids(treated)
        if len(treated):
            try:
                condition.pars.remission_mult = float(self.remission_boost)
            except:
                pass
        return self.treated_inds


# ---------------------------------------------------------------------
# Analyzers
# ---------------------------------------------------------------------
prevalence_analyzer = mi.PrevalenceAnalyzer(
    diseases=["HIV", "AlcoholUseDisorder"]
)

art_analyzer = mi.OnARTByConditionAnalyzer(
    condition_key="alcoholusedisorder.affected"
)
art_analyzer_by_sex = mi.OnARTByConditionAndSexAnalyzer(
    condition_key="alcoholusedisorder.affected"
)


# ---------------------------------------------------------------------
# ART and Testing Interventions
# ---------------------------------------------------------------------
art_coverage_data = pd.DataFrame(
    {"p_art": [0.10, 0.34, 0.50, 0.65, 0.74, 0.85, 0.95]},
    index=[2003, 2010, 2013, 2014, 2016, 2022, 2050],
)

print("[DEBUG] ART coverage data:")
print(art_coverage_data)
print(f"[DEBUG] Simulation years: {inityear} to {endyear}")

hiv_test = sti.HIVTest(
    test_prob_data=[0.10, 0.25, 0.60, 0.70, 0.80, 0.95, 0.95],
    years=[2003, 2005, 2007, 2010, 2014, 2016, 2050],
)

# Create ART interventions for each scenario
art1 = mi.ARTNoAutoAdjust(coverage_data=art_coverage_data, label="art1")
art2 = mi.ARTNoAutoAdjust(coverage_data=art_coverage_data, label="art2")
art3 = mi.ARTNoAutoAdjust(coverage_data=art_coverage_data, label="art3")
art4 = mi.ARTNoAutoAdjust(coverage_data=art_coverage_data, label="art4")

# Create HIV test interventions for each scenario
hiv_test1 = sti.HIVTest(
    test_prob_data=[0.10, 0.25, 0.60, 0.70, 0.80, 0.95, 0.95],
    years=[2003, 2005, 2007, 2010, 2014, 2016, 2050],
    label="hiv_test1"
)
hiv_test2 = sti.HIVTest(
    test_prob_data=[0.10, 0.25, 0.60, 0.70, 0.80, 0.95, 0.95],
    years=[2003, 2005, 2007, 2010, 2014, 2016, 2050],
    label="hiv_test2"
)
hiv_test3 = sti.HIVTest(
    test_prob_data=[0.10, 0.25, 0.60, 0.70, 0.80, 0.95, 0.95],
    years=[2003, 2005, 2007, 2010, 2014, 2016, 2050],
    label="hiv_test3"
)
hiv_test4 = sti.HIVTest(
    test_prob_data=[0.10, 0.25, 0.60, 0.70, 0.80, 0.95, 0.95],
    years=[2003, 2005, 2007, 2010, 2014, 2016, 2050],
    label="hiv_test4"
)

# ---------------------------------------------------------------------
# Adherence system components
# ---------------------------------------------------------------------
# Adherence engine (calculates per-agent adherence based on CASM)
adherence_engine = mi.AdherenceEngine(
    casm_rel={"AlcoholUseDisorder": 0.709},  # AUD reduces adherence to 70.9%
    sdoh_rel={}
)

# Intervention adherence disruptor (scales intervention effectiveness)
intervention_disruptor = mi.InterventionAdherenceDisruptor(scale_art_efficacy=False)

# ART adherence disruptor (drops people from ART based on adherence)
# Requires people to be on ART for 2+ timesteps (2 months) before allowing dropout
# This minimal delay avoids the HIV module bug but allows people with low adherence
# (e.g., due to AUD) to drop out relatively quickly
art_dropout1 = mi.ARTAdherenceDisruptor(base_dropout=0.50, label="artadherencedisruptor")
art_dropout2 = mi.ARTAdherenceDisruptor(base_dropout=0.50, label="artadherencedisruptor")

# ---------------------------------------------------------------------
# AUD Care interventions
# ---------------------------------------------------------------------
intervention_df = pd.read_csv(csv_path_intervention)
unified_product = ss.Tx(df=intervention_df, label="UnifiedTx")

aud_care1 = AUDCare(
    product=unified_product,
    prob=0.1,
    remission_boost=1.5,
    label="aud_care1"
)

aud_care2 = AUDCare(
    product=unified_product,
    prob=0.1,
    remission_boost=1.5,
    label="aud_care2"
)


# ---------------------------------------------------------------------
# Create 4 scenarios
# ---------------------------------------------------------------------
print(f"Initializing sim \"No interaction (HIV+AUD, no CASM effects)\" with {n_agents} agents")
ppl1 = make_people()
sim_noInteraction = ss.Sim(
    n_agents=n_agents,
    start=inityear,
    stop=endyear,
    people=ppl1,
    networks=networks,
    demographics=[death, pregnancy],
    diseases=[hiv, aud],
    interventions=[hiv_test1, art1],
    rand_seed=42,
    analyzers=[prevalence_analyzer, art_analyzer, art_analyzer_by_sex],
    label="No interaction (HIV+AUD, no CASM effects)"
)

print(f"Initializing sim \"With interaction (HIV+AUD, CASM reduces adherence)\" with {n_agents} agents")
ppl2 = make_people()
adherence_engine2 = mi.AdherenceEngine(
    casm_rel={"AlcoholUseDisorder": 0.709},
    sdoh_rel={}
)
sim_withInteraction = ss.Sim(
    n_agents=n_agents,
    start=inityear,
    stop=endyear,
    people=ppl2,
    networks=networks,
    demographics=[death, pregnancy],
    diseases=[hiv, aud],
    interventions=[hiv_test2, art2],
    modules=[adherence_engine2, intervention_disruptor],
    connectors=[art_dropout1],
    rand_seed=43,
    analyzers=[prevalence_analyzer, art_analyzer, art_analyzer_by_sex],
    label="With interaction (HIV+AUD, CASM reduces adherence)"
)

print(f"Initializing sim \"No interaction + AUD care\" with {n_agents} agents")
ppl3 = make_people()
sim_noInteraction_withAUDCare = ss.Sim(
    n_agents=n_agents,
    start=inityear,
    stop=endyear,
    people=ppl3,
    networks=networks,
    demographics=[death, pregnancy],
    diseases=[hiv, aud],
    interventions=[hiv_test3, art3, aud_care1],
    rand_seed=44,
    analyzers=[prevalence_analyzer, art_analyzer, art_analyzer_by_sex],
    label="No interaction + AUD care"
)

print(f"Initializing sim \"With interaction + AUD care\" with {n_agents} agents")
ppl4 = make_people()
adherence_engine4 = mi.AdherenceEngine(
    casm_rel={"AlcoholUseDisorder": 0.709},
    sdoh_rel={}
)
sim_withInteraction_withAUDCare = ss.Sim(
    n_agents=n_agents,
    start=inityear,
    stop=endyear,
    people=ppl4,
    networks=networks,
    demographics=[death, pregnancy],
    diseases=[hiv, aud],
    interventions=[hiv_test4, art4, aud_care2],
    modules=[adherence_engine4, intervention_disruptor],
    connectors=[art_dropout2],
    rand_seed=45,
    analyzers=[prevalence_analyzer, art_analyzer, art_analyzer_by_sex],
    label="With interaction + AUD care"
)

# ---------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------
msim = ss.MultiSim([
    sim_noInteraction,
    sim_withInteraction,
    sim_noInteraction_withAUDCare,
    sim_withInteraction_withAUDCare
])

msim.run()

# ---------------------------------------------------------------------
# Diagnostic function
# ---------------------------------------------------------------------
def diagnose_art_coverage_issue(msim):
    """Print diagnostic information about ART coverage by scenario."""
    print("\n" + "="*80)
    print("ART COVERAGE DIAGNOSTICS")
    print("="*80)
    
    for sim in msim.sims:
        print(f"\n--- {sim.label} ---")
        ppl = sim.people
        st = ppl.states
        
        # Get states
        hiv_infected = np.asarray(st.get("hiv.infected", []), bool)
        hiv_diagnosed = np.asarray(st.get("hiv.diagnosed", []), bool)
        hiv_on_art = np.asarray(st.get("hiv.on_art", []), bool)
        aud_affected = np.asarray(st.get("alcoholusedisorder.affected", []), bool)
        
        # Calculate counts
        aud_hiv = hiv_infected & aud_affected
        noaud_hiv = hiv_infected & ~aud_affected
        aud_hiv_count = aud_hiv.sum()
        noaud_hiv_count = noaud_hiv.sum()
        
        # Check diagnosed status
        aud_diag = aud_hiv & hiv_diagnosed
        noaud_diag = noaud_hiv & hiv_diagnosed
        aud_diag_count = aud_diag.sum()
        noaud_diag_count = noaud_diag.sum()
        aud_diag_rate = aud_diag_count/aud_hiv_count if aud_hiv_count > 0 else 0.0
        noaud_diag_rate = noaud_diag_count/noaud_hiv_count if noaud_hiv_count > 0 else 0.0
        print(f"  Diagnosed AUD HIV+: {aud_diag_count} ({aud_diag_rate:.3f})")
        print(f"  Diagnosed No-AUD HIV+: {noaud_diag_count} ({noaud_diag_rate:.3f})")
        print(f"  Diagnosis rate difference: {aud_diag_rate - noaud_diag_rate:.3f}")
        
        # Check age/sex distribution to see if that explains the difference
        ppl = sim.people
        aud_hiv_ages = ppl.age[aud_hiv]
        noaud_hiv_ages = ppl.age[noaud_hiv]
        aud_hiv_male = ppl.male[aud_hiv].sum() / aud_hiv_count if aud_hiv_count > 0 else 0.0
        noaud_hiv_male = ppl.male[noaud_hiv].sum() / noaud_hiv_count if noaud_hiv_count > 0 else 0.0
        print(f"  AUD HIV+ mean age: {aud_hiv_ages.mean():.1f}, % male: {aud_hiv_male:.3f}")
        print(f"  No-AUD HIV+ mean age: {noaud_hiv_ages.mean():.1f}, % male: {noaud_hiv_male:.3f}")
        
        # Check if diagnosis rate difference is consistent (should be ~0 if not modeled)
        if abs(aud_diag_rate - noaud_diag_rate) > 0.01:
            print(f"  ⚠️  WARNING: Diagnosis rate difference ({aud_diag_rate - noaud_diag_rate:.3f}) is NOT explicitly modeled!")
            print(f"     This could be due to:")
            print(f"     1. Age/sex distribution differences")
            print(f"     2. Initialization artifact")
            print(f"     3. Implicit mechanism in HIVTest or HIV module")
            print(f"     4. Random variation (unlikely given consistency across scenarios)")
        
        # Check ART coverage among diagnosed (more meaningful metric)
        aud_on_art = hiv_on_art & aud_diag
        noaud_on_art = hiv_on_art & noaud_diag
        aud_art_cov = aud_on_art.sum() / aud_diag_count if aud_diag_count > 0 else 0.0
        noaud_art_cov = noaud_on_art.sum() / noaud_diag_count if noaud_diag_count > 0 else 0.0
        print(f"  ART coverage (AUD): {aud_art_cov:.3f} ({aud_on_art.sum()}/{aud_diag_count})")
        print(f"  ART coverage (No-AUD): {noaud_art_cov:.3f} ({noaud_on_art.sum()}/{noaud_diag_count})")
        print(f"  ART coverage difference: {aud_art_cov - noaud_art_cov:.3f}")
        
        # Normalize by diagnosis rate to show true interaction effect
        # If diagnosis rates differ, we need to account for that
        if abs(aud_diag_rate - noaud_diag_rate) > 0.01:
            # Calculate what ART coverage would be if diagnosis rates were equal
            # This normalizes for the diagnosis rate difference
            # Formula: normalized_coverage = (on_art / diagnosed) * (diagnosed / infected) = on_art / infected
            aud_art_cov_normalized = aud_on_art.sum() / aud_hiv_count if aud_hiv_count > 0 else 0.0
            noaud_art_cov_normalized = noaud_on_art.sum() / noaud_hiv_count if noaud_hiv_count > 0 else 0.0
            print(f"  ART coverage (normalized by HIV+): AUD={aud_art_cov_normalized:.3f}, No-AUD={noaud_art_cov_normalized:.3f}, Difference={aud_art_cov_normalized - noaud_art_cov_normalized:.3f}")
            print(f"  → This shows ART coverage among ALL HIV+ (not just diagnosed), accounting for diagnosis rate differences")
        
        # Additional diagnostics: Check if there's something wrong with the calculation
        # Check how many people ever started ART vs currently on ART
        if "hiv.ti_art" in st:
            try:
                ti_art_raw = st.get("hiv.ti_art", [])
                if hasattr(ti_art_raw, 'values'):
                    ti_art = np.asarray(ti_art_raw.values, dtype=float)
                else:
                    ti_art = np.asarray(ti_art_raw, dtype=float)
                
                # Count people who ever started ART (ti_art >= 0 and finite)
                aud_ever_art = (ti_art[aud_diag] >= 0) & np.isfinite(ti_art[aud_diag])
                noaud_ever_art = (ti_art[noaud_diag] >= 0) & np.isfinite(ti_art[noaud_diag])
                aud_ever_art_count = aud_ever_art.sum()
                noaud_ever_art_count = noaud_ever_art.sum()
                
                # Calculate retention rate (currently on ART / ever started)
                aud_retention = aud_on_art.sum() / aud_ever_art_count if aud_ever_art_count > 0 else 0.0
                noaud_retention = noaud_on_art.sum() / noaud_ever_art_count if noaud_ever_art_count > 0 else 0.0
                
                print(f"  Ever started ART (AUD): {aud_ever_art_count}/{aud_diag_count} ({aud_ever_art_count/aud_diag_count:.3f})")
                print(f"  Ever started ART (No-AUD): {noaud_ever_art_count}/{noaud_diag_count} ({noaud_ever_art_count/noaud_diag_count:.3f})")
                print(f"  Retention rate (Currently on / Ever started) (AUD): {aud_retention:.3f} ({aud_on_art.sum()}/{aud_ever_art_count})")
                print(f"  Retention rate (Currently on / Ever started) (No-AUD): {noaud_retention:.3f} ({noaud_on_art.sum()}/{noaud_ever_art_count})")
                
                # Check if the higher coverage is due to higher initiation or higher retention
                if aud_ever_art_count/aud_diag_count > noaud_ever_art_count/noaud_diag_count:
                    print(f"  → AUD has HIGHER ART initiation rate")
                elif aud_retention > noaud_retention:
                    print(f"  → AUD has HIGHER retention rate (less dropout)")
                else:
                    print(f"  → Coverage difference likely due to diagnosis rate difference")
            except Exception as e:
                print(f"  [WARNING] Could not calculate ART initiation history: {e}")
        
        # Check if there's a difference in when people were diagnosed relative to when they could start ART
        # If AUD people are diagnosed earlier in the simulation, they have more opportunities to start ART
        if "hiv.ti_diagnosed" in st:
            try:
                ti_diag_raw = st.get("hiv.ti_diagnosed", [])
                if hasattr(ti_diag_raw, 'values'):
                    ti_diag = np.asarray(ti_diag_raw.values, dtype=float)
                else:
                    ti_diag = np.asarray(ti_diag_raw, dtype=float)
                
                # Check what year people were diagnosed (convert ti to year)
                dt = getattr(sim, 'dt', 1.0/12.0)
                aud_diag_years = (ti_diag[aud_diag] * dt) + inityear
                noaud_diag_years = (ti_diag[noaud_diag] * dt) + inityear
                aud_diag_years_valid = aud_diag_years[(aud_diag_years >= inityear) & (aud_diag_years <= endyear)]
                noaud_diag_years_valid = noaud_diag_years[(noaud_diag_years >= inityear) & (noaud_diag_years <= endyear)]
                
                if len(aud_diag_years_valid) > 0 and len(noaud_diag_years_valid) > 0:
                    print(f"  Mean diagnosis year (AUD): {aud_diag_years_valid.mean():.1f}")
                    print(f"  Mean diagnosis year (No-AUD): {noaud_diag_years_valid.mean():.1f}")
                    print(f"  Diagnosis year difference: {aud_diag_years_valid.mean() - noaud_diag_years_valid.mean():.1f} years")
            except Exception as e:
                print(f"  [WARNING] Could not calculate diagnosis year: {e}")
        
        # Check timing of diagnosis (when were people diagnosed?)
        # Note: ti_diagnosed might be stored in a special format, so we'll be careful
        if "hiv.ti_diagnosed" in st:
            try:
                ti_diag_raw = st.get("hiv.ti_diagnosed", [])
                # Convert to plain numpy array, handling any special types
                if hasattr(ti_diag_raw, 'values'):
                    ti_diag = np.asarray(ti_diag_raw.values, dtype=float)
                else:
                    ti_diag = np.asarray(ti_diag_raw, dtype=float)
                
                aud_ti_diag = ti_diag[aud_diag]
                noaud_ti_diag = ti_diag[noaud_diag]
                
                # Filter valid values (>= 0 and <= current timestep)
                dt = getattr(sim, 'dt', 1.0/12.0)
                current_ti = float(sim.ti)
                
                aud_mask = (aud_ti_diag >= 0) & (aud_ti_diag <= current_ti)
                noaud_mask = (noaud_ti_diag >= 0) & (noaud_ti_diag <= current_ti)
                
                aud_ti_diag_valid = np.asarray(aud_ti_diag[aud_mask], dtype=float)
                noaud_ti_diag_valid = np.asarray(noaud_ti_diag[noaud_mask], dtype=float)
                
                if len(aud_ti_diag_valid) > 0:
                    aud_years_since_diag = np.asarray((current_ti - aud_ti_diag_valid) * dt, dtype=float)
                    print(f"  Mean years since diagnosis (AUD): {aud_years_since_diag.mean():.1f} (n={len(aud_years_since_diag)})")
                if len(noaud_ti_diag_valid) > 0:
                    noaud_years_since_diag = np.asarray((current_ti - noaud_ti_diag_valid) * dt, dtype=float)
                    print(f"  Mean years since diagnosis (No-AUD): {noaud_years_since_diag.mean():.1f} (n={len(noaud_ti_diag_valid)})")
            except Exception as e:
                print(f"  [WARNING] Could not calculate diagnosis timing: {e}")
        
        # Check timing of ART initiation (when did people start ART?)
        if "hiv.ti_art" in st:
            try:
                ti_art_raw = st.get("hiv.ti_art", [])
                # Convert to plain numpy array
                if hasattr(ti_art_raw, 'values'):
                    ti_art = np.asarray(ti_art_raw.values, dtype=float)
                else:
                    ti_art = np.asarray(ti_art_raw, dtype=float)
                
                aud_ti_art = ti_art[aud_on_art]
                noaud_ti_art = ti_art[noaud_on_art]
                
                dt = getattr(sim, 'dt', 1.0/12.0)
                current_ti = float(sim.ti)
                
                aud_mask = (aud_ti_art >= 0) & (aud_ti_art <= current_ti)
                noaud_mask = (noaud_ti_art >= 0) & (noaud_ti_art <= current_ti)
                
                aud_ti_art_valid = np.asarray(aud_ti_art[aud_mask], dtype=float)
                noaud_ti_art_valid = np.asarray(noaud_ti_art[noaud_mask], dtype=float)
                
                if len(aud_ti_art_valid) > 0:
                    aud_years_on_art = np.asarray((current_ti - aud_ti_art_valid) * dt, dtype=float)
                    print(f"  Mean years on ART (AUD): {aud_years_on_art.mean():.1f} (n={len(aud_years_on_art)})")
                if len(noaud_ti_art_valid) > 0:
                    noaud_years_on_art = np.asarray((current_ti - noaud_ti_art_valid) * dt, dtype=float)
                    print(f"  Mean years on ART (No-AUD): {noaud_years_on_art.mean():.1f} (n={len(noaud_ti_art_valid)})")
            except Exception as e:
                print(f"  [WARNING] Could not calculate ART timing: {e}")
        
        # Check if there's a difference in who gets diagnosed early vs late
        # This could explain why AUD has higher coverage (if they're diagnosed earlier)
        if "hiv.ti_diagnosed" in st and aud_diag_count > 0 and noaud_diag_count > 0:
            try:
                ti_diag_raw = st.get("hiv.ti_diagnosed", [])
                if hasattr(ti_diag_raw, 'values'):
                    ti_diag = np.asarray(ti_diag_raw.values, dtype=float)
                else:
                    ti_diag = np.asarray(ti_diag_raw, dtype=float)
                
                aud_ti_diag_all = np.asarray(ti_diag[aud_diag], dtype=float)
                noaud_ti_diag_all = np.asarray(ti_diag[noaud_diag], dtype=float)
                
                # Check what proportion were diagnosed in first 3 years vs later
                dt = getattr(sim, 'dt', 1.0/12.0)
                early_cutoff_ti = float(3.0 / dt)  # 3 years in timesteps
                
                # Filter valid and early diagnoses
                aud_valid = (aud_ti_diag_all >= 0) & (aud_ti_diag_all <= early_cutoff_ti)
                noaud_valid = (noaud_ti_diag_all >= 0) & (noaud_ti_diag_all <= early_cutoff_ti)
                
                aud_early_pct = aud_valid.sum() / len(aud_ti_diag_all) if len(aud_ti_diag_all) > 0 else 0.0
                noaud_early_pct = noaud_valid.sum() / len(noaud_ti_diag_all) if len(noaud_ti_diag_all) > 0 else 0.0
                print(f"  Diagnosed in first 3 years (AUD): {aud_early_pct:.1%}")
                print(f"  Diagnosed in first 3 years (No-AUD): {noaud_early_pct:.1%}")
            except Exception as e:
                print(f"  [WARNING] Could not calculate early diagnosis proportion: {e}")
        
        # Check mean adherence if available
        if "adherence" in st:
            adher = np.asarray(st["adherence"], float)
            aud_adher = adher[aud_affected]
            print(f"  Mean adherence (AUD): {aud_adher.mean():.3f}")
            print(f"  Mean adherence (No-AUD): {adher[~aud_affected].mean():.3f}")

diagnose_art_coverage_issue(msim)

# ---------------------------------------------------------------------
# Additional analysis: Show downstream effects of adherence disruption
# ---------------------------------------------------------------------
def analyze_downstream_effects(msim):
    """Analyze downstream effects of adherence disruption: viral suppression, mortality, etc."""
    print("\n" + "="*80)
    print("DOWNSTREAM EFFECTS OF ADHERENCE DISRUPTION")
    print("="*80)
    
    for sim in msim.sims:
        print(f"\n--- {sim.label} ---")
        ppl = sim.people
        st = ppl.states
        
        # Get states
        hiv_infected = np.asarray(st.get("hiv.infected", []), bool)
        hiv_diagnosed = np.asarray(st.get("hiv.diagnosed", []), bool)
        hiv_on_art = np.asarray(st.get("hiv.on_art", []), bool)
        aud_affected = np.asarray(st.get("alcoholusedisorder.affected", []), bool)
        
        # Calculate ART coverage
        aud_diag = hiv_diagnosed & aud_affected
        noaud_diag = hiv_diagnosed & ~aud_affected
        aud_on_art = hiv_on_art & aud_diag
        noaud_on_art = hiv_on_art & noaud_diag
        
        aud_art_cov = aud_on_art.sum() / aud_diag.sum() if aud_diag.sum() > 0 else 0.0
        noaud_art_cov = noaud_on_art.sum() / noaud_diag.sum() if noaud_diag.sum() > 0 else 0.0
        
        print(f"  ART Coverage: AUD={aud_art_cov:.3f}, No-AUD={noaud_art_cov:.3f}, Difference={aud_art_cov - noaud_art_cov:.3f}")
        
        # Check mean adherence
        if "adherence" in st:
            adher = np.asarray(st["adherence"], float)
            aud_adher = adher[aud_affected].mean() if aud_affected.any() else 1.0
            noaud_adher = adher[~aud_affected].mean() if (~aud_affected).any() else 1.0
            print(f"  Mean Adherence: AUD={aud_adher:.3f}, No-AUD={noaud_adher:.3f}, Difference={aud_adher - noaud_adher:.3f}")
        
        # Calculate cumulative ART dropout (people who ever started but are not currently on ART)
        # Also check if ti_stop_art is set (STIsim's built-in dropout mechanism)
        if "hiv.ti_art" in st:
            try:
                ti_art_raw = st.get("hiv.ti_art", [])
                if hasattr(ti_art_raw, 'values'):
                    ti_art = np.asarray(ti_art_raw.values, dtype=float)
                else:
                    ti_art = np.asarray(ti_art_raw, dtype=float)
                
                # Check if ti_stop_art exists and is set (STIsim's built-in dropout)
                ti_stop_art = None
                if "hiv.ti_stop_art" in st:
                    ti_stop_art_raw = st.get("hiv.ti_stop_art", [])
                    if hasattr(ti_stop_art_raw, 'values'):
                        ti_stop_art = np.asarray(ti_stop_art_raw.values, dtype=float)
                    else:
                        ti_stop_art = np.asarray(ti_stop_art_raw, dtype=float)
                    
                    # Count how many have ti_stop_art set (scheduled to stop)
                    if ti_stop_art is not None:
                        aud_has_stop = (ti_stop_art[aud_diag] >= 0) & np.isfinite(ti_stop_art[aud_diag]) if aud_diag.any() else np.array([], dtype=bool)
                        noaud_has_stop = (ti_stop_art[noaud_diag] >= 0) & np.isfinite(ti_stop_art[noaud_diag]) if noaud_diag.any() else np.array([], dtype=bool)
                        print(f"  Scheduled to stop ART (ti_stop_art set): AUD={aud_has_stop.sum()}, No-AUD={noaud_has_stop.sum()}")
                
                # Count people who ever started ART
                aud_ever_art = (ti_art[aud_diag] >= 0) & np.isfinite(ti_art[aud_diag])
                noaud_ever_art = (ti_art[noaud_diag] >= 0) & np.isfinite(ti_art[noaud_diag])
                aud_ever_count = aud_ever_art.sum()
                noaud_ever_count = noaud_ever_art.sum()
                
                # Calculate dropout rate (ever started - currently on) / ever started
                # IMPORTANT: Exclude deaths from dropout count (they're not "dropouts", they died)
                # Only count alive people who started ART but are not currently on ART
                alive = np.asarray(ppl.alive, dtype=bool)
                aud_alive = alive[aud_diag] if aud_diag.any() else np.array([], dtype=bool)
                noaud_alive = alive[noaud_diag] if noaud_diag.any() else np.array([], dtype=bool)
                
                # Count alive people who ever started ART
                aud_ever_art_alive = aud_ever_art & aud_alive if len(aud_ever_art) == len(aud_alive) else aud_ever_art
                noaud_ever_art_alive = noaud_ever_art & noaud_alive if len(noaud_ever_art) == len(noaud_alive) else noaud_ever_art
                aud_ever_count_alive = aud_ever_art_alive.sum()
                noaud_ever_count_alive = noaud_ever_art_alive.sum()
                
                # Count alive people currently on ART
                aud_on_art_alive = aud_on_art & aud_alive if len(aud_on_art) == len(aud_alive) else aud_on_art
                noaud_on_art_alive = noaud_on_art & noaud_alive if len(noaud_on_art) == len(noaud_alive) else noaud_on_art
                
                # Dropout = alive people who started but are not currently on ART
                aud_dropout_count = aud_ever_count_alive - aud_on_art_alive.sum()
                noaud_dropout_count = noaud_ever_count_alive - noaud_on_art_alive.sum()
                aud_dropout_rate = aud_dropout_count / aud_ever_count_alive if aud_ever_count_alive > 0 else 0.0
                noaud_dropout_rate = noaud_dropout_count / noaud_ever_count_alive if noaud_ever_count_alive > 0 else 0.0
                print(f"  ART Dropout Rate (alive only): AUD={aud_dropout_rate:.3f} ({aud_dropout_count}/{aud_ever_count_alive}), No-AUD={noaud_dropout_rate:.3f} ({noaud_dropout_count}/{noaud_ever_count_alive}), Difference={aud_dropout_rate - noaud_dropout_rate:.3f}")
                
                # Check if STIsim's ti_stop_art is being used (scheduled stops)
                if ti_stop_art is not None:
                    # Count people who have ti_stop_art set (scheduled to stop or already stopped)
                    aud_has_stop = (ti_stop_art[aud_diag] >= 0) & np.isfinite(ti_stop_art[aud_diag]) if aud_diag.any() else np.array([], dtype=bool)
                    noaud_has_stop = (ti_stop_art[noaud_diag] >= 0) & np.isfinite(ti_stop_art[noaud_diag]) if noaud_diag.any() else np.array([], dtype=bool)
                    print(f"  Scheduled to stop ART (ti_stop_art set): AUD={aud_has_stop.sum()}, No-AUD={noaud_has_stop.sum()}")
                    
                    # Check if people who are NOT on ART have ti_stop_art set (they were stopped)
                    aud_stopped = aud_has_stop & ~aud_on_art if len(aud_has_stop) == len(aud_on_art) else np.array([], dtype=bool)
                    noaud_stopped = noaud_has_stop & ~noaud_on_art if len(noaud_has_stop) == len(noaud_on_art) else np.array([], dtype=bool)
                    print(f"  Stopped ART (ti_stop_art set AND not on ART): AUD={aud_stopped.sum()}, No-AUD={noaud_stopped.sum()}")
                
                # Additional check: Are people being dropped and re-initiated?
                if aud_dropout_count > 0:
                    print(f"  [INFO] AUD: {aud_dropout_count} ALIVE people started ART but are NOT currently on (dropped out)")
                if noaud_dropout_count > 0:
                    print(f"  [INFO] No-AUD: {noaud_dropout_count} ALIVE people started ART but are NOT currently on (dropped out)")
            except Exception as e:
                print(f"  [WARNING] Could not calculate dropout rate: {e}")
                import traceback
                traceback.print_exc()

analyze_downstream_effects(msim)

# ---------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------
def plot_art_coverage_over_time(msim, prefix="Fig7A_ART_coverage", burn_in_years=3):
    """
    Plot ART coverage over time, stratified by AUD status and sex.
    Shows differences between AUD and No AUD individuals, separately for males and females.
    """
    sns.set_context("talk", font_scale=1.6)
    
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "axes.labelweight": "bold",
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 15,
        "lines.linewidth": 3,
        "axes.linewidth": 1.3,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })
    
    # Create 2 rows (Male, Female) x 4 columns (scenarios)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Extract stratified ART coverage over time by AUD status and sex
    def _extract_stratified_coverage_by_sex(sim):
        """Extract ART coverage stratified by AUD and sex from analyzer results."""
        ana_key = "onart_alcoholusedisorder_affected_by_sex"
        if ana_key in sim.results:
            res = sim.results[ana_key]
            male_aud_cov = np.asarray(res["onart_cond_male"])
            male_noaud_cov = np.asarray(res["onart_nocond_male"])
            female_aud_cov = np.asarray(res["onart_cond_female"])
            female_noaud_cov = np.asarray(res["onart_nocond_female"])
            
            # Get time vector
            n_steps = len(male_aud_cov)
            if hasattr(sim.t, 'yearvec'):
                if len(sim.t.yearvec) >= n_steps:
                    timevec = sim.t.yearvec[:n_steps]
                else:
                    timevec = np.array(list(sim.t.yearvec) + [sim.t.year] * (n_steps - len(sim.t.yearvec)))
            elif "hiv" in sim.results and "timevec" in sim.results["hiv"]:
                hiv_timevec = np.asarray(sim.results["hiv"]["timevec"])
                if len(hiv_timevec) >= n_steps:
                    timevec = hiv_timevec[:n_steps]
                else:
                    timevec = np.concatenate([hiv_timevec, [hiv_timevec[-1]] * (n_steps - len(hiv_timevec))])
            else:
                dt = getattr(sim, 'dt', 1.0/12.0)
                timevec = np.array([sim.start + ti * dt for ti in range(n_steps)])
            
            return male_aud_cov, male_noaud_cov, female_aud_cov, female_noaud_cov, timevec
        else:
            # Fallback: use final state only
            st = sim.people.states
            ppl = sim.people
            aud_affected = np.asarray(st.get("alcoholusedisorder.affected", []), bool)
            diagnosed = np.asarray(st.get("hiv.diagnosed", []), bool) if "hiv.diagnosed" in st else None
            on_art = np.asarray(st.get("hiv.on_art", []), bool) if "hiv.on_art" in st else np.zeros(len(sim.people), dtype=bool)
            male = np.asarray(ppl.male, bool)
            female = np.asarray(ppl.female, bool)
            
            if diagnosed is not None:
                aud_diag = diagnosed & aud_affected
                noaud_diag = diagnosed & ~aud_affected
            else:
                hiv = np.asarray(st.get("hiv.infected", []), bool) if "hiv.infected" in st else np.zeros(len(sim.people), dtype=bool)
                aud_diag = hiv & aud_affected
                noaud_diag = hiv & ~aud_affected
            
            male_aud_diag = aud_diag & male
            male_noaud_diag = noaud_diag & male
            female_aud_diag = aud_diag & female
            female_noaud_diag = noaud_diag & female
            
            male_aud_cov = on_art[male_aud_diag].sum() / male_aud_diag.sum() if male_aud_diag.sum() > 0 else 0.0
            male_noaud_cov = on_art[male_noaud_diag].sum() / male_noaud_diag.sum() if male_noaud_diag.sum() > 0 else 0.0
            female_aud_cov = on_art[female_aud_diag].sum() / female_aud_diag.sum() if female_aud_diag.sum() > 0 else 0.0
            female_noaud_cov = on_art[female_noaud_diag].sum() / female_noaud_diag.sum() if female_noaud_diag.sum() > 0 else 0.0
            
            timevec = np.array([sim.t.year])
            return (
                np.array([male_aud_cov]),
                np.array([male_noaud_cov]),
                np.array([female_aud_cov]),
                np.array([female_noaud_cov]),
                timevec
            )
    
    scenario_labels = [
        "No interaction\n(no CASM effects)",
        "With interaction\n(CASM reduces adherence)",
        "No interaction\n+ AUD care",
        "With interaction\n+ AUD care"
    ]
    
    for i, sim in enumerate(msim.sims):
        male_aud_cov, male_noaud_cov, female_aud_cov, female_noaud_cov, timevec = _extract_stratified_coverage_by_sex(sim)
        
        # Convert time to years
        if len(timevec) > 0:
            if hasattr(timevec[0], 'year'):  # Date objects
                years = np.array([t.year + (t.month - 1) / 12.0 for t in timevec])
            else:
                years = np.asarray(timevec, dtype=float)
        else:
            years = np.array([sim.t.year])
        
        # Average by year if we have sub-yearly data
        n_years = endyear - inityear + 1
        if len(years) > n_years * 1.5:
            df = pd.DataFrame({
                "year": np.floor(years).astype(int),
                "male_aud": male_aud_cov,
                "male_noaud": male_noaud_cov,
                "female_aud": female_aud_cov,
                "female_noaud": female_noaud_cov
            })
            df_avg = df.groupby("year", as_index=False).mean()
            years_plot = df_avg["year"].values
            male_aud_plot = df_avg["male_aud"].values
            male_noaud_plot = df_avg["male_noaud"].values
            female_aud_plot = df_avg["female_aud"].values
            female_noaud_plot = df_avg["female_noaud"].values
        else:
            years_plot = np.asarray(years, dtype=float)
            male_aud_plot = male_aud_cov
            male_noaud_plot = male_noaud_cov
            female_aud_plot = female_aud_cov
            female_noaud_plot = female_noaud_cov
        
        # Apply burn-in period
        burn_in_year = inityear + burn_in_years
        mask = years_plot >= burn_in_year
        years_plot_filtered = years_plot[mask]
        male_aud_plot_filtered = male_aud_plot[mask]
        male_noaud_plot_filtered = male_noaud_plot[mask]
        female_aud_plot_filtered = female_aud_plot[mask]
        female_noaud_plot_filtered = female_noaud_plot[mask]
        
        # Plot for males (row 0)
        ax_male = axes[0, i]
        ax_male.plot(years_plot_filtered, male_aud_plot_filtered, "-", color="#3182bd", label="AUD", linewidth=3)
        ax_male.plot(years_plot_filtered, male_noaud_plot_filtered, "--", color="#bdbdbd", label="No AUD", linewidth=3)
        ax_male.set_ylim(0, 1)
        ax_male.set_xlim(burn_in_year - 0.5, endyear + 0.5)
        if i == 0:
            ax_male.set_ylabel("Proportion on ART\n(Male)", fontweight="bold")
        else:
            ax_male.set_ylabel("")
        ax_male.set_xlabel("Year", fontweight="bold")
        ax_male.set_title(scenario_labels[i], fontweight="bold", fontsize=18)
        if i == 0:
            ax_male.legend(frameon=False, loc='best')
        ax_male.grid(True, alpha=0.3)
        
        # Plot for females (row 1)
        ax_female = axes[1, i]
        ax_female.plot(years_plot_filtered, female_aud_plot_filtered, "-", color="#3182bd", label="AUD", linewidth=3)
        ax_female.plot(years_plot_filtered, female_noaud_plot_filtered, "--", color="#bdbdbd", label="No AUD", linewidth=3)
        ax_female.set_ylim(0, 1)
        ax_female.set_xlim(burn_in_year - 0.5, endyear + 0.5)
        if i == 0:
            ax_female.set_ylabel("Proportion on ART\n(Female)", fontweight="bold")
        else:
            ax_female.set_ylabel("")
        ax_female.set_xlabel("Year", fontweight="bold")
        if i == 0:
            ax_female.legend(frameon=False, loc='best')
        ax_female.grid(True, alpha=0.3)
    
    plt.suptitle("ART Coverage Over Time by AUD Status and Sex", fontweight="bold", fontsize=24, y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{prefix}.png", dpi=400)
    plt.show()


# ---------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------
plot_art_coverage_over_time(msim, burn_in_years=3)
