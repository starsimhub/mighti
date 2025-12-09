"""
MIGHTI Simulation Script for a selected region: HIV and Health Conditions Interaction Modeling

This script initializes and runs an agent-based simulation using the MIGHTI framework
(built on StarSim and STI-Sim) to analyze the interplay between HIV and
other health conditions (HCs) in selected country. 
It loads demographic data, initializes diseases and networks, 
applies interventions, and analyzes prevalence and mortality outcomes for the selected period.

Key components:
- Loads parameters and prevalence data from CSV files.
- Initializes networks: maternal and structured sexual.
- Initializes HIV and HC modules.
- Sets up demographic modules (deaths, pregnancy).
- Applies HIV interventions (e.g., ART, VMMC).
- Computes and plots prevalence, mortality rates, and life expectancy.

To run: `python mighti_cea.py`
"""


import logging
import mighti as mi
import pandas as pd
import prepare_data_for_year
import starsim as ss
import stisim as sti
import numpy as np
import matplotlib.pyplot as plt

# Set up logging and random seeds for reproducibility
logger = logging.getLogger('MIGHTI')
logger.setLevel(logging.INFO) 



# ---------------------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------------------
logger = logging.getLogger("MIGHTI")
logger.setLevel(logging.INFO)

n_agents = 100_000
inityear = 2007
endyear = 2020
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
csv_path_sdoh = f'mighti/data/sdoh.csv'

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

# Keep it minimal for debugging: HIV + Lower Respiratory Infection to test severity framework
healthconditions = ["LowerRespiratoryInfections"]
diseases = ["HIV"] + healthconditions

#---------------------------------------------------------------------
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
prevalence_analyzer = mi.PrevalenceAnalyzer_HIV(prevalence_data=prevalence_data, diseases=diseases)
survivorship_analyzer = mi.SurvivorshipAnalyzer()
deaths_analyzer = mi.DeathsByAgeSexAnalyzer()

death_cause_analyzer = mi.ConditionAtDeathAnalyzer(
    conditions=healthconditions)

analyzers = [deaths_analyzer, survivorship_analyzer, prevalence_analyzer, death_cause_analyzer]


# Analyzers
# Note: disability_weights are used as fallback if severity system is not available.
# If diseases have severity initialized, severity-specific weights will be used automatically.
microcosting_analyzer_base = mi.MicrocostingAnalyzer(
    unit_costs={'art': 50}, 
    disability_weights={'hiv': 0.2, 'lowerrespiratoryinfections': 0.1},  # Fallback weights if severity not available
    discount_rate_costs=0.03,
    discount_rate_outcomes=0.03,
    name='microcostinganalyzer'
)
microcosting_analyzer_intv = mi.MicrocostingAnalyzer(
    unit_costs={'art': 50}, 
    disability_weights={'hiv': 0.2, 'lowerrespiratoryinfections': 0.1},  # Fallback weights if severity not available
    discount_rate=0.03,
    discount_rate_costs=0.03,
    discount_rate_outcomes=0.03,
    name='microcostinganalyzer' )

intervention_analyzer = mi.InterventionAnalyzer(interventions=['art'], name='intervention_analyzer')

analyzers_base = [deaths_analyzer, survivorship_analyzer, prevalence_analyzer, 
                  intervention_analyzer, death_cause_analyzer, microcosting_analyzer_base]

analyzers_intv = [deaths_analyzer, survivorship_analyzer, prevalence_analyzer, 
                  intervention_analyzer, death_cause_analyzer, microcosting_analyzer_intv]

# ---------------------------------------------------------------------
# Demographics & networks
# ---------------------------------------------------------------------
maternal = ss.MaternalNet()
structuredsexual = sti.StructuredSexual()
networks = [maternal, structuredsexual]

death_rates = {"death_rate": pd.read_csv(csv_path_death), "rate_units": 1}
death = ss.Deaths(death_rates)

fertility_rate = {"fertility_rate": pd.read_csv(csv_path_fertility)}
pregnancy = ss.Pregnancy(pars=fertility_rate)

ppl = ss.People(n_agents, age_data=pd.read_csv(csv_path_age))



# ---------------------------------------------------------------------
# Diseases 
# ---------------------------------------------------------------------
disease_objects = []

# --- HIV ---
hiv = sti.HIV()

# Debug: Check if HIV columns exist in CSV
print(f"[DEBUG] Checking HIV prevalence data...")
print(f"[DEBUG] CSV columns: {list(prevalence_data_df.columns)[:10]}...")  # Show first 10 columns
hiv_male_col = 'HIV_male' in prevalence_data_df.columns
hiv_female_col = 'HIV_female' in prevalence_data_df.columns
print(f"[DEBUG] HIV_male column exists: {hiv_male_col}")
print(f"[DEBUG] HIV_female column exists: {hiv_female_col}")

# Assign prevalence
prev_func = get_prevalence_function('HIV')

# Debug: Check if HIV is in prevalence_data
if 'HIV' in prevalence_data:
    print(f"[DEBUG] HIV found in prevalence_data. Sample values:")
    for sex in ['male', 'female']:
        if sex in prevalence_data['HIV']:
            sample_ages = list(prevalence_data['HIV'][sex].keys())[:5]
            sample_vals = [prevalence_data['HIV'][sex][age] for age in sample_ages]
            print(f"  {sex}: ages {sample_ages} -> {sample_vals}")
            print(f"  {sex}: total age keys: {len(prevalence_data['HIV'][sex])}")
else:
    print(f"[DEBUG] WARNING: HIV NOT found in prevalence_data!")
    print(f"[DEBUG] Available diseases in prevalence_data: {list(prevalence_data.keys())}")

# Test the prevalence function with a dummy sim to see what it returns
class DummySim:
    def __init__(self):
        self.people = type('obj', (object,), {
            'age': np.array([25, 30, 35, 40, 45]),
            'female': np.array([False, True, False, True, False])
        })()

try:
    dummy_sim = DummySim()
    test_prev = prev_func(dummy_sim, None, size=np.arange(5))
    print(f"[DEBUG] Test prevalence function returned: {test_prev}")
    print(f"[DEBUG] Mean prevalence: {test_prev.mean():.4f}, Max: {test_prev.max():.4f}, Min: {test_prev.min():.4f}")
    
    # If all zeros, use constant prevalence as fallback
    if test_prev.mean() == 0.0:
        print(f"[DEBUG] WARNING: HIV prevalence is all zeros! Using constant 15% prevalence as fallback.")
        hiv.pars.init_prev = ss.bernoulli(p=0.15)
    else:
        hiv.pars.init_prev = ss.bernoulli(
            p=lambda sim, uids, size=None: prev_func(sim, uids, size)
        )
except Exception as e:
    print(f"[DEBUG] Error testing prevalence function: {e}")
    print(f"[DEBUG] Using constant 15% prevalence as fallback.")
    hiv.pars.init_prev = ss.bernoulli(p=0.15)

# Transmission parameters
# Best pars: {'hiv_beta_m2f': 0.09553835265049065, 'hiv_beta_m2c': 0.003895160642773216}
# Best pars: {'hiv_beta_m2f': 0.041126225026336546, 'hiv_beta_m2c': 0.02313161100759324}
hiv.pars.beta = {
    'structuredsexual': [0.029594299274445842, 0.029594299274445842],
    'maternal': [0.0011249414706988527, 0.0011249414706988527],
}

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
# Interactions
# ---------------------------------------------------------------------
ncd_hiv_rel_sus = df.set_index('condition')['rel_sus'].to_dict()
ncd_hiv_connector = mi.NCDHIVConnector(ncd_hiv_rel_sus)
connectors = [ncd_hiv_connector]

ncd_interactions = mi.read_interactions(csv_path_interactions) 
connectors.extend(mi.create_connectors(ncd_interactions))


# -------------------------
# Adherence
# -------------------------

# adherence_connectors = [
#     mi.create_adherence_connector('T2D_Tx'),
#     mi.create_adherence_connector('ART'),
# ]
# interactions.extend(adherence_connectors)


# ---------------------------------------------------------------------
# Interventions 
# ---------------------------------------------------------------------
# ART coverage among PLHIV (from 95-95-95 cascade estimates and Lancet data)
art_coverage_data = pd.DataFrame({
    'p_art': [0.10, 0.34, 0.50, 0.65, 0.741, 0.85]
}, index=[2003, 2010, 2013, 2014, 2016, 2022])

# HIV testing probabilities over time (estimated testing uptake)
test_prob_data = [0.10, 0.25, 0.60, 0.70, 0.80, 0.95]
test_years = [2003, 2005, 2007, 2010, 2014, 2016]

intervention_df = pd.read_csv(csv_path_intervention)
unified_product = ss.Tx(df=intervention_df, label='UnifiedTx')


hiv_test = sti.HIVTest(test_prob_data=test_prob_data, years=test_years)
art = sti.ART(coverage_data=art_coverage_data)
vmmc = sti.VMMC(pars={'future_coverage': {'year': 2015, 'prop': 0.30}})
prep = sti.Prep(pars={'coverage': [0, 0.05, 0.25], 'years': [2007, 2015, 2020]})

interventions1 = [hiv_test, art, vmmc, prep]


# ---------------------------------------------------------------------
# Utility: Get Modules
# ---------------------------------------------------------------------
def get_deaths_module(sim):
    for module in sim.modules:
        if isinstance(module, mi.DeathsByAgeSexAnalyzer):
            return module
    raise ValueError("Deaths module not found in the simulation. Make sure you've added the DeathsByAgeSexAnalyzer to your simulation configuration")

def get_pregnancy_module(sim):
    for module in sim.modules:
        if isinstance(module, ss.Pregnancy):
            return module
    raise ValueError("Pregnancy module not found in the simulation.")


# ---------------------------------------------------------------------
# Visualization Function
# ---------------------------------------------------------------------
def plot_cea_results(sim_base, sim_intv, analyzer_base, analyzer_intv, cost_increment, daly_averted, icer):
    """
    Create comprehensive visualization of CEA results.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    # Note: ax3 (HIV Prevalence) and ax4 (ART Coverage) removed
    
    # 1. Cost-Effectiveness Plane (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    results = [{
        'label': 'ART vs Baseline',
        'delta_daly': daly_averted,
        'delta_cost': cost_increment
    }]
    
    # Plot WTP threshold lines
    x_max = max(abs(daly_averted) * 1.2, 1000)
    x_vals = np.linspace(0, x_max, 100)
    wtp_thresholds = [100, 500, 1000]
    colors_wtp = ['green', 'orange', 'red']
    for wtp, color in zip(wtp_thresholds, colors_wtp):
        ax1.plot(x_vals, wtp * x_vals, linestyle='--', color=color, alpha=0.6, 
                label=f'${wtp}/DALY WTP threshold', linewidth=1.5)
    
    # Plot intervention point
    ax1.scatter(daly_averted, cost_increment, s=300, color='blue', 
               marker='o', edgecolor='black', linewidth=2, zorder=5, label='ART Intervention')
    ax1.annotate(f'ICER = ${icer:.2f}/DALY', 
                xy=(daly_averted, cost_increment),
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('DALYs Averted', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Incremental Cost ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Cost-Effectiveness Plane', fontsize=14, fontweight='bold')
    ax1.axhline(0, color='gray', linewidth=0.8, linestyle='-')
    ax1.axvline(0, color='gray', linewidth=0.8, linestyle='-')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    
    # 2. Summary Bar Chart (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    categories = ['Total Cost', 'Total DALY', 'YLD', 'YLL']
    base_vals = [
        analyzer_base.results.total_cost / 1e6,  # Convert to millions
        analyzer_base.results.total_daly / 1e3,   # Convert to thousands
        analyzer_base.results.total_yld / 1e3,
        analyzer_base.results.total_yll / 1e3
    ]
    intv_vals = [
        analyzer_intv.results.total_cost / 1e6,
        analyzer_intv.results.total_daly / 1e3,
        analyzer_intv.results.total_yld / 1e3,
        analyzer_intv.results.total_yll / 1e3
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width/2, base_vals, width, label='Baseline', color='lightcoral', alpha=0.8)
    ax2.bar(x + width/2, intv_vals, width, label='With ART', color='lightblue', alpha=0.8)
    ax2.set_ylabel('Value (Millions $ or Thousands)', fontsize=10)
    ax2.set_title('Summary Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Cost Breakdown (middle left, spans 2 columns)
    ax5 = fig.add_subplot(gs[1, :2])
    try:
        art_cost_val = 0
        if hasattr(analyzer_intv, 'detailed_outputs') and analyzer_intv.detailed_outputs is not None:
            art_cost_val = analyzer_intv.detailed_outputs.get('art_cost', pd.Series([0])).sum()
        elif hasattr(analyzer_intv.results, 'get'):
            art_cost_val = analyzer_intv.results.get('art_cost', 0)
        elif hasattr(analyzer_intv.results, 'art_cost'):
            art_cost_val = analyzer_intv.results.art_cost
        
        total_cost_val = analyzer_intv.results.total_cost
        other_cost = total_cost_val - art_cost_val
        
        if total_cost_val > 0:
            colors_pie = ['#ff9999', '#66b3ff']
            sizes = [art_cost_val, max(other_cost, 0)]
            labels = ['ART Cost', 'Other Costs']
            if other_cost <= 0:
                sizes = [art_cost_val]
                labels = ['ART Cost']
            ax5.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_pie[:len(sizes)])
            ax5.set_title(f'Cost Breakdown\n(Total: ${total_cost_val/1e6:.2f}M)', fontsize=12, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No cost data', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Cost Breakdown', fontsize=12, fontweight='bold')
    except Exception as e:
        ax5.text(0.5, 0.5, f'Error: {str(e)[:30]}', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=10)
        ax5.set_title('Cost Breakdown', fontsize=12, fontweight='bold')
    
    # 6. HIV+ with Lower Respiratory Infections (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    
    # Try to get HIV+ vs HIV- with LRI data
    lri_name_variants = ['lowerrespiratoryinfections', 'LowerRespiratoryInfections', 'lowerrespiratoryinfection']
    lri_disease = None
    lri_name = None
    
    for variant in lri_name_variants:
        if hasattr(sim_intv.diseases, variant):
            lri_disease = getattr(sim_intv.diseases, variant)
            lri_name = variant
            break
        elif variant in sim_intv.diseases:
            lri_disease = sim_intv.diseases[variant]
            lri_name = variant
            break
    
    if lri_disease is not None and hasattr(lri_disease, 'infected') and hasattr(sim_intv.diseases, 'hiv'):
        hiv = sim_intv.diseases.hiv
        lri_infected = lri_disease.infected
        hiv_infected = hiv.infected
        hiv_susceptible = hiv.susceptible
        
        # Find HIV+ and HIV- individuals with LRI
        lri_uids = lri_infected.uids
        hiv_pos_with_lri = lri_uids[lri_infected[lri_uids] & hiv_infected[lri_uids]]
        hiv_neg_with_lri = lri_uids[lri_infected[lri_uids] & hiv_susceptible[lri_uids]]
        
        # Calculate total with LRI
        total_with_lri = len(hiv_pos_with_lri) + len(hiv_neg_with_lri)
        
        if total_with_lri > 0:
            # Calculate proportions
            hiv_pos_prop = len(hiv_pos_with_lri) / total_with_lri * 100
            hiv_neg_prop = len(hiv_neg_with_lri) / total_with_lri * 100
            
            # Get severity distribution
            if hasattr(lri_disease, 'severity_level') and len(hiv_pos_with_lri) > 0:
                hiv_pos_severity = lri_disease.severity_level[hiv_pos_with_lri]
                unique_hiv_pos, counts_hiv_pos = np.unique(hiv_pos_severity, return_counts=True)
                
                # Create stacked bar chart with proportions
                severity_levels = sorted(set(unique_hiv_pos))
                hiv_pos_counts = [counts_hiv_pos[unique_hiv_pos == sev][0] if sev in unique_hiv_pos else 0 for sev in severity_levels]
                
                if len(hiv_neg_with_lri) > 0:
                    hiv_neg_severity = lri_disease.severity_level[hiv_neg_with_lri]
                    unique_hiv_neg, counts_hiv_neg = np.unique(hiv_neg_severity, return_counts=True)
                    hiv_neg_counts = [counts_hiv_neg[unique_hiv_neg == sev][0] if sev in unique_hiv_neg else 0 for sev in severity_levels]
                else:
                    hiv_neg_counts = [0] * len(severity_levels)
                
                # Calculate proportions for each severity level
                total_by_severity = [hiv_pos_counts[i] + hiv_neg_counts[i] for i in range(len(severity_levels))]
                hiv_pos_props = [hiv_pos_counts[i] / total_by_severity[i] * 100 if total_by_severity[i] > 0 else 0 
                                for i in range(len(severity_levels))]
                hiv_neg_props = [hiv_neg_counts[i] / total_by_severity[i] * 100 if total_by_severity[i] > 0 else 0 
                                for i in range(len(severity_levels))]
                
                x = np.arange(len(severity_levels))
                width = 0.35
                
                colors = ['#90EE90', '#FFD700', '#FF6347', '#8B0000']  # Light green, gold, tomato, dark red
                
                # Plot proportions as stacked bars
                for i, sev in enumerate(severity_levels):
                    if hiv_pos_props[i] > 0:
                        ax6.bar(x[i] - width/2, hiv_pos_props[i], width, 
                               label='HIV+' if i == 0 else '', color=colors[min(sev, len(colors)-1)], 
                               alpha=0.8)
                    if hiv_neg_props[i] > 0:
                        ax6.bar(x[i] + width/2, hiv_neg_props[i], width,
                               label='HIV-' if i == 0 else '', color=colors[min(sev, len(colors)-1)],
                               alpha=0.5)
                
                ax6.set_xlabel('Severity Level', fontsize=10)
                ax6.set_ylabel('Proportion (%)', fontsize=10)
                ax6.set_title('Lower Respiratory Infections:\nProportion of HIV+ vs HIV- by Severity', fontsize=12, fontweight='bold')
                ax6.set_xticks(x)
                ax6.set_xticklabels([f'Level {sev}' for sev in severity_levels])
                ax6.set_ylim(0, 105)
                ax6.legend()
                ax6.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for i, sev in enumerate(severity_levels):
                    if hiv_pos_props[i] > 0:
                        ax6.text(x[i] - width/2, hiv_pos_props[i], f'{hiv_pos_props[i]:.1f}%',
                                ha='center', va='bottom', fontsize=8)
                    if hiv_neg_props[i] > 0:
                        ax6.text(x[i] + width/2, hiv_neg_props[i], f'{hiv_neg_props[i]:.1f}%',
                                ha='center', va='bottom', fontsize=8)
            else:
                # Fallback: just show overall proportions
                categories = ['HIV+ with LRI', 'HIV- with LRI']
                proportions = [hiv_pos_prop, hiv_neg_prop]
                ax6.bar(categories, proportions, color=['#FF6347', '#90EE90'], alpha=0.8)
                ax6.set_ylabel('Proportion (%)', fontsize=10)
                ax6.set_title('Lower Respiratory Infections:\nProportion of HIV+ vs HIV-', fontsize=12, fontweight='bold')
                ax6.set_ylim(0, 105)
                ax6.grid(True, alpha=0.3, axis='y')
                for i, prop in enumerate(proportions):
                    ax6.text(i, prop, f'{prop:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            # No LRI cases
            ax6.text(0.5, 0.5, 'No LRI cases\navailable', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Lower Respiratory Infections:\nProportion of HIV+ vs HIV-', fontsize=12, fontweight='bold')
    else:
        # Fallback if LRI data not available
        ax6.text(0.5, 0.5, 'LRI data\nnot available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('HIV+ with Lower Respiratory Infections', fontsize=12, fontweight='bold')
    
    # 7. Key Metrics Summary (bottom center + right)
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')
    
    # Get ART cost for summary
    art_cost_summary = 0
    try:
        if hasattr(analyzer_intv, 'detailed_outputs') and analyzer_intv.detailed_outputs is not None:
            art_cost_summary = analyzer_intv.detailed_outputs.get('art_cost', pd.Series([0])).sum()
        elif hasattr(analyzer_intv.results, 'get'):
            art_cost_summary = analyzer_intv.results.get('art_cost', 0)
        elif hasattr(analyzer_intv.results, 'art_cost'):
            art_cost_summary = analyzer_intv.results.art_cost
    except:
        art_cost_summary = 0
    
    # Create summary text
    summary_text = f"""
    COST-EFFECTIVENESS ANALYSIS SUMMARY
    
    Baseline Scenario:
      • Total Cost: ${analyzer_base.results.total_cost:,.2f}
      • Total DALY: {analyzer_base.results.total_daly:,.2f}
      • YLD: {analyzer_base.results.total_yld:,.2f}
      • YLL: {analyzer_base.results.total_yll:,.2f}
    
    With ART Scenario:
      • Total Cost: ${analyzer_intv.results.total_cost:,.2f}
      • Total DALY: {analyzer_intv.results.total_daly:,.2f}
      • YLD: {analyzer_intv.results.total_yld:,.2f}
      • YLL: {analyzer_intv.results.total_yll:,.2f}
      • ART Cost: ${art_cost_summary:,.2f}
    
    Incremental Results:
      • DALYs Averted: {daly_averted:,.2f}
      • Incremental Cost: ${cost_increment:,.2f}
      • ICER: ${icer:,.2f} per DALY averted
    """
    
    # Determine cost-effectiveness
    if icer < 100:
        ce_status = "HIGHLY COST-EFFECTIVE"
        ce_color = 'green'
    elif icer < 500:
        ce_status = "COST-EFFECTIVE"
        ce_color = 'blue'
    elif icer < 1000:
        ce_status = "MODERATELY COST-EFFECTIVE"
        ce_color = 'orange'
    else:
        ce_status = "NOT COST-EFFECTIVE"
        ce_color = 'red'
    
    summary_text += f"\n    Cost-Effectiveness Status: {ce_status}"
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add status indicator (removed "HIGHLY COST-EFFECTIVE" text as requested)
    # Only show if not "HIGHLY COST-EFFECTIVE"
    if ce_status != "HIGHLY COST-EFFECTIVE":
        ax7.text(0.95, 0.95, ce_status, transform=ax7.transAxes,
                fontsize=14, fontweight='bold', color=ce_color,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=ce_color, linewidth=3))
    
    plt.suptitle('Cost-Effectiveness Analysis: ART Intervention', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save as PNG instead of showing
    plt.savefig('cea_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("CEA results saved to: cea_results.png")
    
    return fig
    
    
# ---------------------------------------------------------------------
# Main Simulation
# ---------------------------------------------------------------------
if __name__ == '__main__':
    sim_base = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        analyzers=analyzers_base,
        diseases=disease_objects,
        connectors=connectors,
        label='Baseline'
    )

    sim_intv = ss.Sim(
        n_agents=n_agents,
        networks=networks,
        start=inityear,
        stop=endyear,
        people=ppl,
        demographics=[pregnancy, death],
        analyzers=analyzers_intv,
        diseases=disease_objects,
        connectors=connectors,
        interventions=[hiv_test, art],
        label='With ART'
    )

    msim = ss.MultiSim([sim_base, sim_intv])
    msim.run()
    
    analyzer_base = sim_base.analyzers.microcostinganalyzer
    analyzer_intv = sim_intv.analyzers.microcostinganalyzer
    
    # # Compute ICER
    icer = analyzer_intv.compute_icer(analyzer_base)
    
    # Print results
    df_art = sim_intv.analyzers.intervention_analyzer.to_df()
    n_art = df_art[df_art['received_art'] == True]['uid'].nunique()

    cost_base = analyzer_base.results.total_cost
    cost_art = analyzer_intv.results.total_cost
    daly_base = analyzer_base.results.total_daly
    daly_art = analyzer_intv.results.total_daly

    daly_averted = daly_base - daly_art
    cost_increment = cost_art - cost_base

    icer = cost_increment / daly_averted if daly_averted > 0 else np.inf

    print("\n ICER Calculation:")
    print(f"  Cost (baseline): ${cost_base:,.2f}")
    print(f"  Cost (ART):      ${cost_art:,.2f}")
    print(f"  DALY (baseline): {daly_base:,.2f}")
    print(f"  DALY (ART):      {daly_art:,.2f}")
    print(f"  DALYs averted:   {daly_averted:,.2f}")
    print(f"  Incremental Cost: ${cost_increment:,.2f}")
    print(f"  ICER: ${icer:,.2f} per DALY averted")

    # Debug: Check duration for the first non-HIV disease in the simulation
    import inspect
    non_hiv_diseases = [d for d in sim_intv.diseases.keys() if d != 'hiv']
    if non_hiv_diseases:
        disease_name = non_hiv_diseases[0]
        d = sim_intv.diseases.get(disease_name, None)
        if d is not None:
            dur = d.duration
            print(f"\n--- {disease_name.upper()} DEBUG ---")
            print("NaNs:", np.isnan(dur).sum(), "mean duration:", np.mean(dur))
            print('Class:', d.__class__)
            print('MRO:', inspect.getmro(d.__class__))
            print('Has duration attr:', hasattr(d, 'duration'))
            if hasattr(d, 'duration'):
                print('duration type:', type(d.duration))
                print('first few values:', d.duration[:10])
            
            # Also check baseline
            d_base = sim_base.diseases.get(disease_name, None)
            if d_base is not None:
                print(f'--- {disease_name.upper()} DEBUG (Base) ---')
                print('Class:', d_base.__class__)
                print('MRO:', inspect.getmro(d_base.__class__))
                print('Has duration attr:', hasattr(d_base, 'duration'))
                if hasattr(d_base, 'duration'):
                    print('duration type:', type(d_base.duration))
                    print('first few values:', d_base.duration[:10])
    else:
        print("\nNo non-HIV diseases found in simulation for duration debugging")

    summary_base = mi.summarize_microcosting_results(analyzer_base)
    summary_intv = mi.summarize_microcosting_results(analyzer_intv)

    print("\nSummary: Baseline")
    for k, v in summary_base.items():
        print(f"{k}: {v:,.2f}")

    print("\nSummary: With ART")
    for k, v in summary_intv.items():
        print(f"{k}: {v:,.2f}")

    # Debug: Print all available diseases and YLD keys
    print("\n" + "="*60)
    print("DEBUG: Available Diseases and YLD Data")
    print("="*60)
    print(f"Available diseases in simulation: {list(sim_intv.diseases.keys())}")
    
    if hasattr(analyzer_base, 'detailed_outputs') and analyzer_base.detailed_outputs is not None:
        print(f"\nAvailable YLD keys in analyzer_base.detailed_outputs:")
        for key in analyzer_base.detailed_outputs.keys():
            if 'yld' in key.lower():
                val = analyzer_base.detailed_outputs[key]
                if hasattr(val, 'sum'):
                    print(f"  {key}: {val.sum():,.2f}")
                else:
                    print(f"  {key}: {val}")
    else:
        print("\nanalyzer_base.detailed_outputs is None or not available")
    
    # Print Lower Respiratory Infections statistics
    print("\n" + "="*60)
    print("LOWER RESPIRATORY INFECTIONS STATISTICS")
    print("="*60)
    
    # Check if disease exists in simulation
    lri_name_variants = ['lowerrespiratoryinfections', 'LowerRespiratoryInfections', 'lowerrespiratoryinfection']
    lri_disease = None
    lri_name = None
    
    for variant in lri_name_variants:
        if hasattr(sim_intv.diseases, variant):
            lri_disease = getattr(sim_intv.diseases, variant)
            lri_name = variant
            break
        elif variant in sim_intv.diseases:
            lri_disease = sim_intv.diseases[variant]
            lri_name = variant
            break
    
    # Also try to find by iterating through all diseases
    if lri_disease is None:
        for name, disease_obj in sim_intv.diseases.items():
            if hasattr(disease_obj, 'disease_name'):
                if 'lower' in disease_obj.disease_name.lower() and 'respiratory' in disease_obj.disease_name.lower():
                    lri_disease = disease_obj
                    lri_name = name
                    print(f"Found LRI disease with name '{name}' (disease_name: {disease_obj.disease_name})")
                    break
    
    if lri_disease is not None:
        # Get prevalence
        if hasattr(lri_disease, 'infected'):
            n_infected_base = sim_base.diseases[lri_name].infected.sum() if lri_name in sim_base.diseases else 0
            n_infected_intv = lri_disease.infected.sum()
            n_total_base = len(sim_base.people)
            n_total_intv = len(sim_intv.people)
            prev_base = n_infected_base / n_total_base * 100 if n_total_base > 0 else 0
            prev_intv = n_infected_intv / n_total_intv * 100 if n_total_intv > 0 else 0
            
            print(f"\nPrevalence:")
            print(f"  Baseline: {n_infected_base:,} / {n_total_base:,} ({prev_base:.2f}%)")
            print(f"  With ART: {n_infected_intv:,} / {n_total_intv:,} ({prev_intv:.2f}%)")
        
        # Get YLD from analyzer
        if hasattr(analyzer_base, 'detailed_outputs') and analyzer_base.detailed_outputs is not None:
            lri_yld_base = analyzer_base.detailed_outputs.get('lowerrespiratoryinfections_yld', pd.Series([0])).sum()
        else:
            lri_yld_base = 0
        
        if hasattr(analyzer_intv, 'detailed_outputs') and analyzer_intv.detailed_outputs is not None:
            lri_yld_intv = analyzer_intv.detailed_outputs.get('lowerrespiratoryinfections_yld', pd.Series([0])).sum()
        else:
            lri_yld_intv = 0
        
        print(f"\nYLD (Years Lived with Disability):")
        print(f"  Baseline: {lri_yld_base:,.2f}")
        print(f"  With ART: {lri_yld_intv:,.2f}")
        print(f"  Difference: {lri_yld_base - lri_yld_intv:,.2f}")
        
        # Get severity distribution if available
        if hasattr(lri_disease, 'severity_level'):
            severity_base = sim_base.diseases[lri_name].severity_level if lri_name in sim_base.diseases else None
            severity_intv = lri_disease.severity_level
            
            if severity_base is not None:
                unique_base, counts_base = np.unique(severity_base[lri_disease.infected.uids] if hasattr(lri_disease, 'infected') else severity_base, return_counts=True)
                print(f"\nSeverity Distribution (Baseline):")
                for sev, count in zip(unique_base, counts_base):
                    print(f"  Level {sev}: {count:,} ({count/len(severity_base)*100:.1f}%)")
            
            if hasattr(lri_disease, 'infected'):
                infected_uids = lri_disease.infected.uids
                if len(infected_uids) > 0:
                    unique_intv, counts_intv = np.unique(severity_intv[infected_uids], return_counts=True)
                    print(f"\nSeverity Distribution (With ART):")
                    for sev, count in zip(unique_intv, counts_intv):
                        print(f"  Level {sev}: {count:,} ({count/len(infected_uids)*100:.1f}%)")
        
        # =====================================================================
        # HIV+ INDIVIDUALS WITH LOWER RESPIRATORY INFECTIONS ANALYSIS
        # =====================================================================
        print("\n" + "="*60)
        print("HIV+ INDIVIDUALS WITH LOWER RESPIRATORY INFECTIONS")
        print("="*60)
        
        # Check HIV status
        if hasattr(sim_intv.diseases, 'hiv') and hasattr(sim_intv.diseases.hiv, 'infected'):
            hiv_infected = sim_intv.diseases.hiv.infected
            hiv_susceptible = sim_intv.diseases.hiv.susceptible
            
            if hasattr(lri_disease, 'infected'):
                lri_infected = lri_disease.infected
                
                # Find HIV+ individuals with LRI
                hiv_pos_with_lri = lri_infected.uids[lri_infected[lri_infected.uids] & hiv_infected[lri_infected.uids]]
                hiv_neg_with_lri = lri_infected.uids[lri_infected[lri_infected.uids] & hiv_susceptible[lri_infected.uids]]
                
                print(f"\nCo-infection Status (With ART, Final Year):")
                print(f"  HIV+ with LRI: {len(hiv_pos_with_lri):,}")
                print(f"  HIV- with LRI: {len(hiv_neg_with_lri):,}")
                print(f"  Total with LRI: {len(lri_infected.uids):,}")
                
                if len(hiv_pos_with_lri) > 0:
                    # Severity distribution for HIV+ individuals
                    if hasattr(lri_disease, 'severity_level'):
                        hiv_pos_severity = lri_disease.severity_level[hiv_pos_with_lri]
                        unique_hiv_pos, counts_hiv_pos = np.unique(hiv_pos_severity, return_counts=True)
                        print(f"\nSeverity Distribution - HIV+ with LRI:")
                        for sev, count in zip(unique_hiv_pos, counts_hiv_pos):
                            pct = count / len(hiv_pos_with_lri) * 100
                            print(f"  Level {sev}: {count:,} ({pct:.1f}%)")
                    
                    # Compare with HIV- individuals
                    if len(hiv_neg_with_lri) > 0 and hasattr(lri_disease, 'severity_level'):
                        hiv_neg_severity = lri_disease.severity_level[hiv_neg_with_lri]
                        unique_hiv_neg, counts_hiv_neg = np.unique(hiv_neg_severity, return_counts=True)
                        print(f"\nSeverity Distribution - HIV- with LRI:")
                        for sev, count in zip(unique_hiv_neg, counts_hiv_neg):
                            pct = count / len(hiv_neg_with_lri) * 100
                            print(f"  Level {sev}: {count:,} ({pct:.1f}%)")
                    
                    # Calculate mean severity
                    if hasattr(lri_disease, 'severity_level'):
                        mean_sev_hiv_pos = lri_disease.severity_level[hiv_pos_with_lri].mean()
                        if len(hiv_neg_with_lri) > 0:
                            mean_sev_hiv_neg = lri_disease.severity_level[hiv_neg_with_lri].mean()
                            print(f"\nMean Severity:")
                            print(f"  HIV+ with LRI: {mean_sev_hiv_pos:.2f}")
                            print(f"  HIV- with LRI: {mean_sev_hiv_neg:.2f}")
                            print(f"  Difference: {mean_sev_hiv_pos - mean_sev_hiv_neg:.2f}")
                
                # Calculate YLD for HIV+ vs HIV- individuals with LRI
                if hasattr(analyzer_intv, 'detailed_outputs') and analyzer_intv.detailed_outputs is not None:
                    # Get per-individual YLD if available
                    lri_yld_array = None
                    for name_var in ['lowerrespiratoryinfections_yld', 'LowerRespiratoryInfections_yld', 'lowerrespiratoryinfection_yld']:
                        if name_var in analyzer_intv.detailed_outputs:
                            lri_yld_array = analyzer_intv.detailed_outputs[name_var]
                            break
                    
                    if lri_yld_array is not None and hasattr(lri_yld_array, '__getitem__'):
                        # Calculate YLD for HIV+ vs HIV- with LRI
                        hiv_pos_yld = lri_yld_array[hiv_pos_with_lri].sum() if len(hiv_pos_with_lri) > 0 else 0
                        hiv_neg_yld = lri_yld_array[hiv_neg_with_lri].sum() if len(hiv_neg_with_lri) > 0 else 0
                        
                        print(f"\nYLD Contribution:")
                        print(f"  HIV+ with LRI: {hiv_pos_yld:,.2f} YLD")
                        print(f"  HIV- with LRI: {hiv_neg_yld:,.2f} YLD")
                        if len(hiv_pos_with_lri) > 0:
                            print(f"  Mean YLD per HIV+ person with LRI: {hiv_pos_yld / len(hiv_pos_with_lri):,.2f}")
                        if len(hiv_neg_with_lri) > 0:
                            print(f"  Mean YLD per HIV- person with LRI: {hiv_neg_yld / len(hiv_neg_with_lri):,.2f}")
                
                # Check deaths from LRI in HIV+ individuals
                if hasattr(sim_intv.analyzers, 'condition_at_death_analyzer'):
                    death_analyzer = sim_intv.analyzers.condition_at_death_analyzer
                    if hasattr(death_analyzer, 'to_df'):
                        df_deaths = death_analyzer.to_df()
                        if len(df_deaths) > 0:
                            # Count deaths from LRI
                            died_from_lri = df_deaths[df_deaths.get('cause_lowerrespiratoryinfections', False) == True]
                            hiv_pos_died_from_lri = died_from_lri[died_from_lri.get('hiv_positive', False) == True]
                            
                            print(f"\nDeaths from Lower Respiratory Infections:")
                            print(f"  Total deaths from LRI: {len(died_from_lri):,}")
                            print(f"  HIV+ deaths from LRI: {len(hiv_pos_died_from_lri):,}")
                            if len(died_from_lri) > 0:
                                print(f"  Proportion HIV+: {len(hiv_pos_died_from_lri) / len(died_from_lri) * 100:.1f}%")
                            
                            # Calculate YLL for HIV+ deaths from LRI
                            if 'yll' in df_deaths.columns:
                                total_yll_lri = died_from_lri['yll'].sum()
                                hiv_pos_yll_lri = hiv_pos_died_from_lri['yll'].sum() if len(hiv_pos_died_from_lri) > 0 else 0
                                
                                print(f"\nYLL Contribution (Years of Life Lost):")
                                print(f"  Total YLL from LRI deaths: {total_yll_lri:,.2f}")
                                print(f"  HIV+ YLL from LRI deaths: {hiv_pos_yll_lri:,.2f}")
                                if total_yll_lri > 0:
                                    print(f"  Proportion from HIV+: {hiv_pos_yll_lri / total_yll_lri * 100:.1f}%")
                
                # Baseline comparison
                print(f"\n--- Baseline Scenario Comparison ---")
                if lri_name in sim_base.diseases:
                    lri_base = sim_base.diseases[lri_name]
                    if hasattr(lri_base, 'infected') and hasattr(sim_base.diseases, 'hiv'):
                        hiv_base = sim_base.diseases.hiv
                        lri_infected_base = lri_base.infected
                        hiv_infected_base = hiv_base.infected
                        hiv_susceptible_base = hiv_base.susceptible
                        
                        hiv_pos_with_lri_base = lri_infected_base.uids[
                            lri_infected_base[lri_infected_base.uids] & hiv_infected_base[lri_infected_base.uids]
                        ]
                        
                        print(f"  HIV+ with LRI (Baseline): {len(hiv_pos_with_lri_base):,}")
                        print(f"  HIV+ with LRI (With ART): {len(hiv_pos_with_lri):,}")
                        if len(hiv_pos_with_lri_base) > 0:
                            change = len(hiv_pos_with_lri) - len(hiv_pos_with_lri_base)
                            pct_change = change / len(hiv_pos_with_lri_base) * 100
                            print(f"  Change: {change:+,} ({pct_change:+.1f}%)")
        else:
            print("\nWARNING: HIV disease not found in simulation!")
        
    else:
        print("\nWARNING: Lower Respiratory Infections disease not found in simulation!")
        print(f"Available diseases: {list(sim_intv.diseases.keys())}")
    
    print("="*60)

    # ---------------------------------------------------------------------
    # Create comprehensive visualization
    # ---------------------------------------------------------------------
    plot_cea_results(sim_base, sim_intv, analyzer_base, analyzer_intv, cost_increment, daly_averted, icer)