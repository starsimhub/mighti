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
art_dropout1 = mi.ARTAdherenceDisruptor(base_dropout=0.30, label="artadherencedisruptor")
art_dropout2 = mi.ARTAdherenceDisruptor(base_dropout=0.30, label="artadherencedisruptor")

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
    analyzers=[prevalence_analyzer, art_analyzer],
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
    analyzers=[prevalence_analyzer, art_analyzer],
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
    analyzers=[prevalence_analyzer, art_analyzer],
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
    analyzers=[prevalence_analyzer, art_analyzer],
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
        
        # Check mean adherence if available
        if "adherence" in st:
            adher = np.asarray(st["adherence"], float)
            aud_adher = adher[aud_affected]
            print(f"  Mean adherence (AUD): {aud_adher.mean():.3f}")
            print(f"  Mean adherence (No-AUD): {adher[~aud_affected].mean():.3f}")

diagnose_art_coverage_issue(msim)


# ---------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------
def plot_art_coverage_over_time(msim, prefix="Fig7A_ART_coverage", burn_in_years=3):
    """
    Plot ART coverage over time, stratified by AUD status.
    Shows differences between AUD and No AUD individuals, which is where the effects are visible.
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Extract stratified ART coverage over time using OnARTByConditionAnalyzer results
    def _extract_stratified_coverage(sim):
        """Extract ART coverage stratified by AUD from analyzer results."""
        ana_key = "onart_alcoholusedisorder_affected"
        if ana_key in sim.results:
            res = sim.results[ana_key]
            aud_coverage = np.asarray(res["onart_with_condition"])
            noaud_coverage = np.asarray(res["onart_without_condition"])
            
            # Analyzer results are indexed by timestep (ti), so use yearvec which matches timesteps
            # Get the actual length of analyzer results
            n_steps = len(aud_coverage)
            
            # Use yearvec which has one entry per timestep
            # The analyzer stores one value per timestep, so we need timevec with same length
            if hasattr(sim.t, 'yearvec'):
                # yearvec has one entry per timestep
                if len(sim.t.yearvec) >= n_steps:
                    timevec = sim.t.yearvec[:n_steps]
                else:
                    # If yearvec is shorter, pad or use what we have
                    timevec = np.array(list(sim.t.yearvec) + [sim.t.year] * (n_steps - len(sim.t.yearvec)))
            elif "hiv" in sim.results and "timevec" in sim.results["hiv"]:
                hiv_timevec = np.asarray(sim.results["hiv"]["timevec"])
                # Take first n_steps entries (HIV results also have one per timestep)
                if len(hiv_timevec) >= n_steps:
                    timevec = hiv_timevec[:n_steps]
                else:
                    # Pad if needed
                    timevec = np.concatenate([hiv_timevec, [hiv_timevec[-1]] * (n_steps - len(hiv_timevec))])
            else:
                # Fallback: construct from start year and timesteps
                # Assuming monthly timesteps (dt = 1/12)
                dt = getattr(sim, 'dt', 1.0/12.0)
                timevec = np.array([sim.start + ti * dt for ti in range(n_steps)])
            
            # Ensure arrays have same length
            min_len = min(len(aud_coverage), len(noaud_coverage), len(timevec))
            return aud_coverage[:min_len], noaud_coverage[:min_len], timevec[:min_len]
        
        # Fallback: return final values only
        st = sim.people.states
        aud_affected = np.asarray(st.get("alcoholusedisorder.affected", []), bool)
        diagnosed = np.asarray(st.get("hiv.diagnosed", []), bool) if "hiv.diagnosed" in st else None
        on_art = np.asarray(st.get("hiv.on_art", []), bool) if "hiv.on_art" in st else np.zeros(len(sim.people), dtype=bool)
        
        if diagnosed is not None:
            aud_diag = diagnosed & aud_affected
            noaud_diag = diagnosed & ~aud_affected
            aud_cov = on_art[aud_diag].sum() / aud_diag.sum() if aud_diag.sum() > 0 else 0.0
            noaud_cov = on_art[noaud_diag].sum() / noaud_diag.sum() if noaud_diag.sum() > 0 else 0.0
        else:
            hiv = np.asarray(st.get("hiv.infected", []), bool) if "hiv.infected" in st else np.zeros(len(sim.people), dtype=bool)
            aud_hiv = hiv & aud_affected
            noaud_hiv = hiv & ~aud_affected
            aud_cov = on_art[aud_hiv].sum() / aud_hiv.sum() if aud_hiv.sum() > 0 else 0.0
            noaud_cov = on_art[noaud_hiv].sum() / noaud_hiv.sum() if noaud_hiv.sum() > 0 else 0.0
        
        timevec = np.asarray(sim.results.timevec) if hasattr(sim.results, 'timevec') else np.array([sim.t.year])
        return (np.full(len(timevec), aud_cov), 
                np.full(len(timevec), noaud_cov), 
                timevec)
    
    scenario_labels = [
        "No interaction\n(no CASM effects)",
        "With interaction\n(CASM reduces adherence)",
        "No interaction\n+ AUD care",
        "With interaction\n+ AUD care"
    ]
    
    for i, sim in enumerate(msim.sims):
        ax = axes[i]
        aud_cov, noaud_cov, timevec = _extract_stratified_coverage(sim)
        
        # Debug: print what we extracted
        if i == 0:  # Only print for first sim to avoid clutter
            print(f"[DEBUG plot] Sim {i}: aud_cov len={len(aud_cov)}, timevec len={len(timevec)}")
            print(f"[DEBUG plot] timevec first 5: {timevec[:5] if len(timevec) > 0 else 'empty'}")
            print(f"[DEBUG plot] timevec last 5: {timevec[-5:] if len(timevec) > 0 else 'empty'}")
        
        # Convert time to years
        # timevec should already be in year format (2007, 2008, etc.) based on our extraction
        if len(timevec) > 0:
            if hasattr(timevec[0], 'year'):  # Date objects
                years = np.array([t.year + (t.month - 1) / 12.0 for t in timevec])
            else:
                # Convert to float array - should already be in years
                years = np.asarray(timevec, dtype=float)
        else:
            years = np.array([sim.t.year])
        
        # Debug: verify year conversion
        if i == 0:
            print(f"[DEBUG plot] After conversion: years range {years.min():.1f} to {years.max():.1f}, len={len(years)}")
        
        # Average by year if we have sub-yearly data (monthly timesteps)
        # Check if we have more data points than years
        n_years = endyear - inityear + 1
        if len(years) > n_years * 1.5:  # More than 1.5x yearly resolution
            df = pd.DataFrame({
                "year": np.floor(years).astype(int), 
                "aud": aud_cov, 
                "noaud": noaud_cov
            })
            df_avg = df.groupby("year", as_index=False).mean()
            years_plot = df_avg["year"].values
            aud_plot = df_avg["aud"].values
            noaud_plot = df_avg["noaud"].values
        else:
            # Use years as-is, but ensure they're numeric
            years_plot = np.asarray(years, dtype=float)
            aud_plot = aud_cov
            noaud_plot = noaud_cov
        
        # Apply burn-in period: filter out early years
        burn_in_year = inityear + burn_in_years
        mask = years_plot >= burn_in_year
        years_plot_filtered = years_plot[mask]
        aud_plot_filtered = aud_plot[mask]
        noaud_plot_filtered = noaud_plot[mask]
        
        # Debug: check what we're plotting
        if i == 0:
            print(f"[DEBUG plot] years_plot range: {years_plot.min():.1f} to {years_plot.max():.1f}, len={len(years_plot)}")
            print(f"[DEBUG plot] After burn-in (>= {burn_in_year}): {len(years_plot_filtered)} data points")
        
        ax.plot(years_plot_filtered, aud_plot_filtered, "-", color="#3182bd", label="AUD", linewidth=3)
        ax.plot(years_plot_filtered, noaud_plot_filtered, "--", color="#bdbdbd", label="No AUD", linewidth=3)
        ax.set_ylim(0, 1)
        ax.set_xlim(burn_in_year - 0.5, endyear + 0.5)  # Start x-axis from burn-in year
        ax.set_xlabel("Year", fontweight="bold")
        ax.set_ylabel("Proportion on ART", fontweight="bold")
        ax.set_title(scenario_labels[i], fontweight="bold", fontsize=18)
        ax.legend(frameon=False, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("ART Coverage Over Time by AUD Status", fontweight="bold", fontsize=24, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(f"{prefix}.png", dpi=400)
    plt.show()


# ---------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------
plot_art_coverage_over_time(msim, burn_in_years=3)
