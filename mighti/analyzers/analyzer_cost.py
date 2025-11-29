import starsim as ss
import numpy as np
import pandas as pd

# Import severity function - handle import error gracefully
try:
    from mighti.diseases.base_disease import get_disability_weight_by_severity
    HAS_SEVERITY_SUPPORT = True
except ImportError:
    HAS_SEVERITY_SUPPORT = False
    def get_disability_weight_by_severity(disease, uids=None):
        """Fallback if severity module not available."""
        return np.array([])

__all__ = ['MicrocostingAnalyzer', 'HRHAnalyzer', 'summarize_microcosting_results']

class MicrocostingAnalyzer(ss.Analyzer):
    def __init__(self, unit_costs=None, disability_weights=None,
                 discount_rate=None,
                 discount_rate_costs=0.03, discount_rate_outcomes=0.03,
                 **kwargs):

        if discount_rate is not None:
            discount_rate_costs = discount_rate_outcomes = discount_rate

        self.unit_costs = unit_costs or {}
        self.disability_weights = disability_weights or {}
        self.discount_rate_costs = discount_rate_costs
        self.discount_rate_outcomes = discount_rate_outcomes

        # Remove these to prevent warnings from Starsim
        for k in ['discount_rate', 'discount_rate_costs', 'discount_rate_outcomes']:
            kwargs.pop(k, None)

        super().__init__(**kwargs)
        self.name = 'microcostinganalyzer'
        self.detailed_outputs = None

    def init_results(self):
        super().init_results() 
        self.results = ss.Results(self)  
        return

    def step(self):
        """Dummy step to silence Starsim warnings."""
        pass

    def finalize(self):
        super().finalize()

        ppl = self.sim.people
        n_total = ppl.n_uids      
        n_alive = len(ppl)
        uids_all = np.arange(n_total)
        years = self.sim.t.yearvec
        n_years = len(years)

        print(f"\n Finalizing MicrocostingAnalyzer for {n_alive:,} alive / {n_total:,} total agents across {n_years} years")

        # Initialize arrays at full population size
        total_cost = np.zeros(n_total)
        total_yld  = np.zeros(n_total)
        total_yll  = np.zeros(n_total)
        cost_details, yld_details, yll_details = {}, {}, {}

        # ---------------------------------------------------------------------
        # Event-based costs
        # ---------------------------------------------------------------------
        print("\n Event-based costs:")
        for event, unit_cost in self.unit_costs.items():
            if event == 'art':
                continue
            if hasattr(ppl, f'{event}_count'):
                counts = getattr(ppl, f'{event}_count')
                if isinstance(counts, dict):
                    arr = np.array([counts.get(uid, 0) for uid in uids_all])
                else:
                    arr = np.asarray(counts)
                    if len(arr) < n_total:
                        arr = np.pad(arr, (0, n_total - len(arr)))
                cost = arr * unit_cost / ((1 + self.discount_rate_costs) ** (n_years - 1))
                total_cost += cost
                cost_details[f'{event}_cost'] = cost
                print(f"  • {event}: {cost.sum():,.2f}")

        # ---------------------------------------------------------------------
        # YLDs
        # ---------------------------------------------------------------------
        print("\n YLDs by condition:")

        for cond, fallback_weight in self.disability_weights.items():

            # Special logic for HIV (ti_infected-based)
            if cond == 'hiv' and hasattr(ppl, 'hiv') and hasattr(ppl.hiv, 'ti_infected'):
                print(f"  Calculating HIV YLDs dynamically from ti_infected")

                ti_infected_arr = np.full(n_total, np.nan)
                ti_infected_arr[:len(ppl.hiv.ti_infected)] = ppl.hiv.ti_infected[:]  # Fill full-length array

                infected_mask = ~np.isnan(ti_infected_arr)
                ti_infected_clipped = np.clip(ti_infected_arr[infected_mask], 0, len(self.sim.t.yearvec) - 1).astype(int)
                start_years = self.sim.t.yearvec[ti_infected_clipped]
                end_year = self.sim.t.yearvec[-1]
                dur_years = end_year - start_years

                # Check if HIV has severity system
                hiv_disease = getattr(self.sim.diseases, 'hiv', None)
                if HAS_SEVERITY_SUPPORT and hiv_disease and hasattr(hiv_disease, 'severity_weights'):
                    # Use severity-specific weights
                    infected_uids = np.where(infected_mask)[0]
                    severity_weights = get_disability_weight_by_severity(hiv_disease, infected_uids)
                    if len(severity_weights) > 0:
                        print(f"    Using severity-specific weights (mean: {severity_weights.mean():.4f}, range: [{severity_weights.min():.4f}, {severity_weights.max():.4f}])")
                    else:
                        severity_weights = np.full(len(infected_uids), fallback_weight)
                        print(f"    Using fixed weight: {fallback_weight} (severity weights not available)")
                else:
                    # Fallback to fixed weight
                    infected_uids = np.where(infected_mask)[0]
                    severity_weights = np.full(len(infected_uids), fallback_weight)
                    print(f"    Using fixed weight: {fallback_weight}")

                yld = np.zeros(n_total)
                yld[infected_mask] = dur_years * severity_weights / ((1 + self.discount_rate_outcomes) ** (n_years - 1))
                yld_details[f'{cond}_yld'] = yld
                total_yld += yld
                print(f"  {cond}: {yld.sum():.2f}")
                continue

            # New: dynamic condition duration lookup with severity support
            elif hasattr(self.sim.diseases, cond) and hasattr(self.sim.diseases[cond], 'duration'):
                disease = self.sim.diseases[cond]
                durations = disease.duration
                
                # Check if disease has severity system
                if HAS_SEVERITY_SUPPORT and hasattr(disease, 'severity_weights'):
                    # Use severity-specific weights
                    # Get affected/infected individuals
                    if hasattr(disease, 'affected'):
                        affected_uids = disease.affected.uids
                    elif hasattr(disease, 'infected'):
                        affected_uids = disease.infected.uids
                    else:
                        affected_uids = np.arange(len(durations))
                    
                    # Get severity-specific weights for affected individuals
                    severity_weights = get_disability_weight_by_severity(disease, affected_uids)
                    
                    # Map weights to full array
                    weights_array = np.full(n_total, fallback_weight)
                    if len(affected_uids) > 0 and len(severity_weights) > 0:
                        valid_uids = affected_uids[affected_uids < n_total]
                        valid_weights = severity_weights[:len(valid_uids)]
                        weights_array[valid_uids] = valid_weights
                    
                    print(f"  Calculating {cond} YLDs from disease.duration with severity-specific weights")
                    if len(severity_weights) > 0:
                        print(f"    Mean severity weight: {severity_weights.mean():.4f}, range: [{severity_weights.min():.4f}, {severity_weights.max():.4f}]")
                    print(f"    Affected individuals: {len(affected_uids)}")
                else:
                    # Fallback to fixed weight
                    weights_array = np.full(n_total, fallback_weight)
                    print(f"  Calculating {cond} YLDs from disease.duration (using fixed weight: {fallback_weight})")
                
                # Calculate YLD: duration * weight (per individual)
                yld = np.zeros(n_total)
                if len(durations) > 0:
                    # For affected individuals, use their duration and severity-specific weight
                    if hasattr(disease, 'affected'):
                        affected_uids = disease.affected.uids
                    elif hasattr(disease, 'infected'):
                        affected_uids = disease.infected.uids
                    else:
                        affected_uids = np.arange(min(len(durations), n_total))
                    
                    # Ensure arrays are aligned
                    n_affected = min(len(affected_uids), len(durations))
                    affected_uids = affected_uids[:n_affected]
                    durations_subset = durations[:n_affected]
                    weights_subset = weights_array[affected_uids]
                    
                    # Calculate YLD for affected individuals
                    yld[affected_uids] = durations_subset * weights_subset / ((1 + self.discount_rate_outcomes) ** (n_years - 1))
                
                if len(yld) < n_total:
                    yld = np.pad(yld, (0, n_total - len(yld)))
                total_yld += yld
                yld_details[f'{cond}_yld'] = yld
                print(f"  {cond}: {yld.sum():.2f}")

            else:
                print(f"  Missing: {cond} or duration attribute not found")
        
        
        # ---------------------------------------------------------------------
        # YLLs
        # ---------------------------------------------------------------------
        print("\n YLLs from ConditionAtDeathAnalyzer:")
        condition_death = self.sim.analyzers.get('condition_at_death_analyzer', None)
        if condition_death and hasattr(condition_death, 'to_df'):
            df_yll = condition_death.to_df()
            n_deaths = len(df_yll)
            print(f"  Number of deaths recorded: {n_deaths:,}")
            if n_deaths:
                yll_array_discounted = df_yll['yll'].to_numpy() / ((1 + self.discount_rate_outcomes) ** (n_years - 1))
                discounted_series = pd.Series(yll_array_discounted, index=df_yll['uid'])
                mapped = discounted_series.reindex(uids_all, fill_value=0.0).to_numpy()
                total_yll += mapped
                yll_details['yll'] = mapped
                print(f"  • Total YLLs: {mapped.sum():,.2f}")
        else:
            print(" ConditionAtDeathAnalyzer not found")

        # ---------------------------------------------------------------------
        # ART costs
        # ---------------------------------------------------------------------
        print("\n ART costing:")
        intervention_analyzer = self.sim.analyzers.get('intervention_analyzer', None)
        if intervention_analyzer is None:
            raise ValueError("MicrocostingAnalyzer requires 'intervention_analyzer' in sim.analyzers.")
        if 'art' in self.unit_costs:
            art_df = intervention_analyzer.to_df()
            
            # Debug: Check dataframe structure
            print(f"  Debug: InterventionAnalyzer dataframe shape: {art_df.shape}")
            print(f"  Debug: Columns: {art_df.columns.tolist()}")
            if 'received_art' in art_df.columns:
                print(f"  Debug: received_art column exists, True count: {art_df['received_art'].sum()}")
                print(f"  Debug: received_art unique values: {art_df['received_art'].unique()}")
            else:
                print(f"  Debug: WARNING - 'received_art' column not found!")
                print(f"  Debug: Available columns: {art_df.columns.tolist()}")
            
            # Check if anyone is on ART in the simulation
            if hasattr(self.sim.diseases, 'hiv'):
                hiv = self.sim.diseases.hiv
                if hasattr(hiv, 'on_art'):
                    n_on_art = hiv.on_art.sum()
                    print(f"  Debug: Current number on ART in sim: {n_on_art}")
                if hasattr(hiv, 'infected'):
                    n_infected = hiv.infected.sum()
                    print(f"  Debug: Current number HIV infected: {n_infected}")
                if hasattr(hiv, 'diagnosed'):
                    n_diagnosed = hiv.diagnosed.sum()
                    print(f"  Debug: Current number HIV diagnosed: {n_diagnosed}")
                # Check ART intervention
                art_intervention = None
                for intv in getattr(self.sim, 'interventions', []):
                    if hasattr(intv, '__class__') and 'ART' in intv.__class__.__name__:
                        art_intervention = intv
                        break
                if art_intervention:
                    print(f"  Debug: ART intervention found: {art_intervention.__class__.__name__}")
                else:
                    print(f"  Debug: WARNING - No ART intervention found in sim.interventions!")
                    print(f"  Debug: Available interventions: {[i.__class__.__name__ for i in getattr(self.sim, 'interventions', [])]}")
            
            if 'received_art' in art_df.columns and len(art_df) > 0:
                art_counts = (
                    art_df[art_df['received_art']]
                    .groupby('uid').size()
                    .reindex(uids_all, fill_value=0)
                    .to_numpy()
                )
                art_cost = art_counts * self.unit_costs['art'] / ((1 + self.discount_rate_costs) ** (n_years - 1))
                total_cost += art_cost
                cost_details['art_cost'] = art_cost
                print(f"  • ART total cost: {art_cost.sum():,.2f} for {art_counts.sum():,} doses")
            else:
                print(f"  • WARNING: No ART data found in InterventionAnalyzer. ART cost set to 0.")
                art_cost = np.zeros(n_total)
                cost_details['art_cost'] = art_cost

        # ---------------------------------------------------------------------
        # Final DataFrame
        # ---------------------------------------------------------------------
        df = pd.DataFrame({
            'uid': uids_all,
            'total_cost': total_cost,
            'total_yld': total_yld,
            'total_yll': total_yll,
            'total_daly': total_yld + total_yll,
        })

        for key, val in {**cost_details, **yld_details, **yll_details}.items():
            df[key] = val

        self.detailed_outputs = df

        print(f"\n Finalized MicrocostingAnalyzer")
        print(f"   → Total cost: ${total_cost.sum():,.2f}")
        print(f"   → Total YLD: {total_yld.sum():,.2f}")
        print(f"   → Total YLL: {total_yll.sum():,.2f}")
        print(f"   → Total DALY: {df['total_daly'].sum():,.2f}")

        print(f"\n Finalized MicrocostingAnalyzer")
        print(f"   → Total cost: ${total_cost.sum():,.2f}")
        print(f"   → Total YLD: {total_yld.sum():,.2f}")
        print(f"   → Total YLL: {total_yll.sum():,.2f}")
        print(f"   → Total DALY: {df['total_daly'].sum():,.2f}")

        # ---------------------------------------------------------------------
        # Store summary results for programmatic access
        # ---------------------------------------------------------------------
        self.results['total_cost'] = total_cost.sum()
        self.results['total_yld'] = total_yld.sum()
        self.results['total_yll'] = total_yll.sum()
        self.results['total_daly'] = (total_yld + total_yll).sum()

        # Optionally store the detailed dataframe too
        self.results['detailed_outputs'] = self.detailed_outputs

        return

    def compute_icer(self, other_analyzer):
        df_self = self.to_df().set_index('uid')
        df_other = other_analyzer.to_df().set_index('uid')
        df_common = df_self.join(df_other, lsuffix='_intv', rsuffix='_base', how='inner')
        delta_cost = df_common['total_cost_intv'].sum() - df_common['total_cost_base'].sum()
        delta_daly = df_common['total_daly_base'].sum() - df_common['total_daly_intv'].sum()
        icer = delta_cost / delta_daly if delta_daly != 0 else np.inf
        return {'delta_cost': delta_cost, 'delta_daly': delta_daly, 'icer': icer}

    def to_df(self):
        return self.detailed_outputs


class HRHAnalyzer(ss.Analyzer):
    """Summarizes human resource utilization by cadre per timestep."""

    def __init__(self, label="hrh_analyzer"):
        super().__init__(label=label)
        self.records = []

    def apply(self, sim):
        econ = sim.get_module("budget_constraint", optional=True)
        if econ and econ.resources:
            used = {**econ.resources.summarize()}
            used["t"] = sim.t
            self.records.append(used)

    def finalize(self, sim):
        self.df = pd.DataFrame(self.records)
        sim.results["hrh"] = self.df


def summarize_microcosting_results(analyzer):
    """
    Summarize total cost, YLL, YLD, and DALYs from a MicrocostingAnalyzer,
    including per-condition and per-event breakdowns if available.
    """
    if not hasattr(analyzer, 'detailed_outputs') or analyzer.detailed_outputs is None:
        raise ValueError("Analyzer does not contain detailed_outputs. Run the simulation first.")

    df = analyzer.detailed_outputs

    summary = {
        'total_cost': df['total_cost'].sum(),
        'total_yll': df['total_yll'].sum(),
        'total_yld': df['total_yld'].sum(),
        'total_daly': df['total_daly'].sum(),
    }

    # Include all columns that end in _yld or _cost (per-condition/per-event)
    for col in df.columns:
        if col.endswith('_yld') or col.endswith('_cost'):
            summary[col] = df[col].sum()

    return summary   
