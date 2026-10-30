import starsim as ss
import numpy as np
import pandas as pd
import logging

from mighti.analyzers.disability_weights import (
    DEFAULT_HIV_YLD_MODE,
    DEFAULT_MULTIMORBIDITY_RULE,
    HIV_YLD_MODES,
    MULTIMORBIDITY_RULES,
    adjust_total_yld_for_multimorbidity,
    canonical_dw_key,
    classify_hiv_stage,
    hiv_stage_disability_weight,
    resolve_disability_weights,
    resolve_disease_module,
)

logger = logging.getLogger(__name__)

__all__ = ['MicrocostingAnalyzer', 'HRHAnalyzer', 'summarize_microcosting_results']

class MicrocostingAnalyzer(ss.Analyzer):
    def __init__(self, unit_costs=None, disability_weights=None,
                 discount_rate=None,
                 discount_rate_costs=0.03, discount_rate_outcomes=0.03,
                 use_default_disability_weights=True,
                 multimorbidity_rule=DEFAULT_MULTIMORBIDITY_RULE,
                 hiv_yld_mode=DEFAULT_HIV_YLD_MODE,
                 treated_disability_weights=None,
                 treatment_cost_keys=None,
                 **kwargs):

        if discount_rate is not None:
            discount_rate_costs = discount_rate_outcomes = discount_rate

        self.unit_costs = unit_costs or {}
        # None → core GBD defaults (HIV + cardiometabolic). Explicit dict replaces defaults.
        self.disability_weights = resolve_disability_weights(
            disability_weights,
            use_defaults=use_default_disability_weights,
        )
        rule = (multimorbidity_rule or DEFAULT_MULTIMORBIDITY_RULE).lower()
        if rule not in MULTIMORBIDITY_RULES:
            raise ValueError(
                f"Unknown multimorbidity_rule={multimorbidity_rule!r}; "
                f"choose from {MULTIMORBIDITY_RULES}"
            )
        self.multimorbidity_rule = rule
        mode = (hiv_yld_mode or DEFAULT_HIV_YLD_MODE).lower()
        if mode not in HIV_YLD_MODES:
            raise ValueError(
                f"Unknown hiv_yld_mode={hiv_yld_mode!r}; choose from {HIV_YLD_MODES}"
            )
        self.hiv_yld_mode = mode
        self.discount_rate_costs = discount_rate_costs
        self.discount_rate_outcomes = discount_rate_outcomes
        treated = treated_disability_weights or {}
        self.treated_disability_weights = {
            canonical_dw_key(k): float(v) for k, v in treated.items()
        }
        self.treatment_cost_keys = {
            canonical_dw_key(condition): str(cost_key)
            for condition, cost_key in (treatment_cost_keys or {}).items()
        }

        # Remove these to prevent warnings from Starsim
        for k in [
            'discount_rate',
            'discount_rate_costs',
            'discount_rate_outcomes',
            'multimorbidity_rule',
            'use_default_disability_weights',
            'hiv_yld_mode',
            'treated_disability_weights',
            'treatment_cost_keys',
        ]:
            kwargs.pop(k, None)

        super().__init__(**kwargs)
        self.name = 'microcostinganalyzer'
        self.detailed_outputs = None
        self._hiv_yld_stage_acc = None  # undiscounted person-years × stage DW
        self._hiv_stage_py = None  # optional person-years by stage (for Methods tables)
        self._treated_yld_acc = None  # cond → per-agent undiscounted YLD
        self._treatment_py_acc = None  # cost_key → per-agent person-years on treatment

    def init_pre(self, sim):
        super().init_pre(sim)
        n = max(int(sim.people.uid.len_used), 1)
        if self.hiv_yld_mode == "stage":
            self._hiv_yld_stage_acc = np.zeros(n, dtype=float)
            self._hiv_stage_py = {
                "art": 0.0,
                "aids": 0.0,
                "symptomatic": 0.0,
                "early": 0.0,
            }
        if self.treated_disability_weights or any(
            k in self.unit_costs for k in self.treatment_cost_keys.values()
        ):
            self._treated_yld_acc = {}
            self._treatment_py_acc = {}
            for cond in set(self.treated_disability_weights) | set(self.treatment_cost_keys):
                self._treated_yld_acc[cond] = np.zeros(n, dtype=float)
                cost_key = self.treatment_cost_keys.get(cond)
                if cost_key and cost_key in self.unit_costs:
                    self._treatment_py_acc[cost_key] = np.zeros(n, dtype=float)

    def init_results(self):
        super().init_results()
        # starsim ≥3.2 locks ``results``; keep the parent Results container.
        return

    def _ensure_hiv_acc(self, n_used: int):
        if self._hiv_yld_stage_acc is None:
            self._hiv_yld_stage_acc = np.zeros(n_used, dtype=float)
        elif self._hiv_yld_stage_acc.size < n_used:
            new = np.zeros(n_used, dtype=float)
            new[: self._hiv_yld_stage_acc.size] = self._hiv_yld_stage_acc
            self._hiv_yld_stage_acc = new

    @staticmethod
    def _sim_dt_year(sim) -> float:
        t = sim.t
        if hasattr(t, "dt_year") and t.dt_year is not None:
            try:
                return float(t.dt_year)
            except Exception:
                pass
        dt = getattr(t, "dt", 1.0)
        try:
            return float(getattr(dt, "years", dt))
        except Exception:
            return 1.0

    def _ensure_treated_acc(self, n_used: int):
        if not self._treated_yld_acc:
            return
        for key, acc in list(self._treated_yld_acc.items()):
            if acc.size < n_used:
                new = np.zeros(n_used, dtype=float)
                new[: acc.size] = acc
                self._treated_yld_acc[key] = new
        if self._treatment_py_acc:
            for key, acc in list(self._treatment_py_acc.items()):
                if acc.size < n_used:
                    new = np.zeros(n_used, dtype=float)
                    new[: acc.size] = acc
                    self._treatment_py_acc[key] = new

    def _accrue_treated_yld_and_tx_py(self):
        if self._treated_yld_acc is None and self._treatment_py_acc is None:
            return
        sim = self.sim
        n_used = int(sim.people.uid.len_used)
        self._ensure_treated_acc(n_used)
        dt = self._sim_dt_year(sim)
        conds = set(self._treated_yld_acc or {}) | {
            k for k, ck in self.treatment_cost_keys.items()
            if ck in (self._treatment_py_acc or {})
        }
        for cond in conds:
            disease, _ = resolve_disease_module(sim.diseases, cond)
            if disease is None or not hasattr(disease, "affected"):
                continue
            affected = np.zeros(n_used, dtype=bool)
            raw_aff = getattr(disease.affected, "raw", None)
            src = raw_aff if raw_aff is not None else disease.affected
            n = min(len(np.asarray(src)), n_used)
            affected[:n] = np.asarray(src[:n], dtype=bool)
            on_tx = np.zeros(n_used, dtype=bool)
            if hasattr(disease, "on_treatment"):
                raw_tx = getattr(disease.on_treatment, "raw", None)
                src_tx = raw_tx if raw_tx is not None else disease.on_treatment
                n_tx = min(len(np.asarray(src_tx)), n_used)
                on_tx[:n_tx] = np.asarray(src_tx[:n_tx], dtype=bool)

            untreated_w = float(self.disability_weights.get(cond, 0.0))
            treated_w = float(self.treated_disability_weights.get(cond, untreated_w))
            if self._treated_yld_acc is not None and cond in self._treated_yld_acc:
                acc = self._treated_yld_acc[cond]
                acc[:n_used] += np.where(affected & on_tx, treated_w, 0.0) * dt
                acc[:n_used] += np.where(affected & ~on_tx, untreated_w, 0.0) * dt

            cost_key = self.treatment_cost_keys.get(cond)
            if cost_key and self._treatment_py_acc is not None and cost_key in self._treatment_py_acc:
                self._treatment_py_acc[cost_key][:n_used] += on_tx * dt

    def step(self):
        """Accrue HIV stage YLD and (if configured) treated NCD YLD / treatment person-years."""
        self._accrue_treated_yld_and_tx_py()
        if self.hiv_yld_mode != "stage":
            return
        sim = self.sim
        hiv = getattr(sim.diseases, "hiv", None)
        if hiv is None:
            return
        n_used = int(sim.people.uid.len_used)
        self._ensure_hiv_acc(n_used)
        dt = self._sim_dt_year(sim)

        def _bool(name):
            arr = getattr(hiv, name, None)
            if arr is None:
                return np.zeros(n_used, dtype=bool)
            raw = getattr(arr, "raw", None)
            if raw is not None:
                return np.asarray(raw[:n_used], dtype=bool)
            return np.asarray(arr[:n_used], dtype=bool)

        def _float(name):
            arr = getattr(hiv, name, None)
            if arr is None:
                return np.full(n_used, np.nan)
            raw = getattr(arr, "raw", None)
            if raw is not None:
                return np.asarray(raw[:n_used], dtype=float)
            return np.asarray(arr[:n_used], dtype=float)

        infected = _bool("infected")
        if not infected.any():
            return
        stages = classify_hiv_stage(
            infected=infected,
            on_art=_bool("on_art"),
            cd4=_float("cd4"),
            acute=_bool("acute"),
            latent=_bool("latent"),
            falling=_bool("falling"),
        )
        dws = hiv_stage_disability_weight(stages)
        self._hiv_yld_stage_acc[:n_used] += dws * dt
        if self._hiv_stage_py is not None:
            for key in self._hiv_stage_py:
                self._hiv_stage_py[key] += float((stages == key).sum()) * dt

    def finalize(self):
        super().finalize()

        ppl = self.sim.people
        n_total = ppl.n_uids      
        n_alive = len(ppl)
        uids_all = np.arange(n_total)
        years = self.sim.t.yearvec
        n_years = len(years)

        logger.info(
            "Finalizing MicrocostingAnalyzer for %s alive / %s total agents across %s years",
            f"{n_alive:,}",
            f"{n_total:,}",
            n_years,
        )

        # Initialize arrays at full population size
        total_cost = np.zeros(n_total)
        total_yld  = np.zeros(n_total)
        total_yll  = np.zeros(n_total)
        cost_details, yld_details, yll_details = {}, {}, {}

        # ---------------------------------------------------------------------
        # Event-based costs
        # ---------------------------------------------------------------------
        logger.info("Event-based costs:")
        ncd_tx_keys = set(self.treatment_cost_keys.values())
        for event, unit_cost in self.unit_costs.items():
            if event == 'art' or event in ncd_tx_keys:
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
                logger.info("  - %s: %s", event, f"{cost.sum():,.2f}")

        # ---------------------------------------------------------------------
        # YLDs
        # ---------------------------------------------------------------------
        logger.info("YLDs by condition:")

        for cond, weight in self.disability_weights.items():

            # HIV YLD: stage time-in-state (Option A) or average duration × DW
            if cond == 'hiv' and hasattr(self.sim.diseases, 'hiv'):
                disc = (1 + self.discount_rate_outcomes) ** (n_years - 1)
                yld = np.zeros(n_total)
                yld_avg = np.zeros(n_total)

                # Average (single-DW) path — always computed for sensitivity columns
                hiv_mod = self.sim.diseases.hiv
                if hasattr(hiv_mod, 'ti_infected'):
                    ti_raw = getattr(hiv_mod.ti_infected, 'raw', hiv_mod.ti_infected)
                    ti_infected_arr = np.full(n_total, np.nan)
                    n_ti = min(len(ti_raw), n_total)
                    ti_infected_arr[:n_ti] = np.asarray(ti_raw[:n_ti], dtype=float)
                    infected_mask = np.isfinite(ti_infected_arr)
                    if infected_mask.any():
                        ti_clipped = np.clip(
                            ti_infected_arr[infected_mask], 0, len(self.sim.t.yearvec) - 1
                        ).astype(int)
                        start_years = self.sim.t.yearvec[ti_clipped]
                        end_year = self.sim.t.yearvec[-1]
                        dur_years = end_year - start_years
                        yld_avg[infected_mask] = dur_years * weight / disc

                if self.hiv_yld_mode == 'stage' and self._hiv_yld_stage_acc is not None:
                    acc = self._hiv_yld_stage_acc
                    n_acc = min(len(acc), n_total)
                    yld[:n_acc] = acc[:n_acc] / disc
                    logger.info(
                        "  - hiv (stage): %0.2f  [average sensitivity: %0.2f]",
                        float(yld.sum()),
                        float(yld_avg.sum()),
                    )
                    yld_details['hiv_yld'] = yld
                    yld_details['hiv_yld_average'] = yld_avg
                    total_yld += yld
                else:
                    yld[:] = yld_avg
                    yld_details['hiv_yld'] = yld
                    total_yld += yld
                    logger.info("  - hiv (average): %0.2f", float(yld.sum()))
                continue

            # Treated/untreated time-in-state YLD (Paper 2) when accrued
            if self._treated_yld_acc is not None and cond in self._treated_yld_acc:
                disc = (1 + self.discount_rate_outcomes) ** (n_years - 1)
                acc = self._treated_yld_acc[cond]
                yld = np.zeros(n_total)
                n_acc = min(len(acc), n_total)
                yld[:n_acc] = acc[:n_acc] / disc
                total_yld += yld
                yld_details[f'{cond}_yld'] = yld
                logger.info("  - %s (treated/untreated time): %0.2f", cond, float(yld.sum()))
                continue

            # Duration-based YLD for named conditions (alias-tolerant lookup)
            disease, resolved = resolve_disease_module(self.sim.diseases, cond)
            if disease is None:
                logger.debug("  Missing: %s module not found", cond)
                continue
            try:
                durations = disease.duration
            except AttributeError:
                logger.debug("  Missing: %s duration attribute not found", cond)
                continue
            logger.debug(
                "  Calculating %s YLDs from disease.duration (resolved=%s)",
                cond,
                resolved,
            )
            yld = durations * weight / ((1 + self.discount_rate_outcomes) ** (n_years - 1))
            if len(yld) < n_total:
                yld = np.pad(yld, (0, n_total - len(yld)))
            total_yld += yld
            yld_details[f'{cond}_yld'] = yld
            logger.info("  - %s: %0.2f", cond, float(yld.sum()))

        # Multimorbidity: keep per-condition YLD columns additive for attribution;
        # recombine totals under the chosen rule (default: multiplicative).
        # Exclude sensitivity-only columns (e.g. hiv_yld_average).
        if yld_details:
            yld_for_mm = {
                k[: -len("_yld")]: v
                for k, v in yld_details.items()
                if k.endswith("_yld") and "average" not in k
            }
            total_yld = adjust_total_yld_for_multimorbidity(
                yld_for_mm,
                self.disability_weights,
                rule=self.multimorbidity_rule,
            )
            logger.info(
                "Multimorbidity rule=%s → total YLD %s",
                self.multimorbidity_rule,
                f"{total_yld.sum():,.2f}",
            )

        # ---------------------------------------------------------------------
        # YLLs
        # ---------------------------------------------------------------------
        logger.info("YLLs from ConditionAtDeathAnalyzer:")
        condition_death = self.sim.analyzers.get('condition_at_death_analyzer', None)
        if condition_death and hasattr(condition_death, 'to_df'):
            df_yll = condition_death.to_df()
            n_deaths = len(df_yll)
            logger.info("  - Number of deaths recorded: %s", f"{n_deaths:,}")
            if n_deaths:
                yll_array_discounted = df_yll['yll'].to_numpy() / ((1 + self.discount_rate_outcomes) ** (n_years - 1))
                discounted_series = pd.Series(yll_array_discounted, index=df_yll['uid'])
                mapped = discounted_series.reindex(uids_all, fill_value=0.0).to_numpy()
                total_yll += mapped
                yll_details['yll'] = mapped
                logger.info("  - Total YLLs: %s", f"{mapped.sum():,.2f}")
        else:
            logger.debug("ConditionAtDeathAnalyzer not found")

        # ---------------------------------------------------------------------
        # ART costs
        # ---------------------------------------------------------------------
        logger.info("ART costing:")
        intervention_analyzer = self.sim.analyzers.get('intervention_analyzer', None)
        if intervention_analyzer is None:
            raise ValueError("MicrocostingAnalyzer requires 'intervention_analyzer' in sim.analyzers.")
        if 'art' in self.unit_costs:
            art_df = intervention_analyzer.to_df()
            if art_df is None or len(art_df) == 0 or 'uid' not in art_df.columns:
                art_counts = np.zeros(n_total, dtype=float)
            else:
                received = art_df['received_art'] if 'received_art' in art_df.columns else True
                art_counts = (
                    art_df.loc[received]
                    .groupby('uid').size()
                    .reindex(uids_all, fill_value=0)
                    .to_numpy()
                )
            art_cost = art_counts * self.unit_costs['art'] / ((1 + self.discount_rate_costs) ** (n_years - 1))
            total_cost += art_cost
            cost_details['art_cost'] = art_cost
            logger.info(
                "  - ART total cost: %s for %s doses",
                f"{art_cost.sum():,.2f}",
                f"{int(art_counts.sum()):,}",
            )

        # ---------------------------------------------------------------------
        # NCD treatment person-year costs (Paper 2)
        # ---------------------------------------------------------------------
        if self._treatment_py_acc:
            disc_c = (1 + self.discount_rate_costs) ** (n_years - 1)
            for cost_key, py in self._treatment_py_acc.items():
                if cost_key not in self.unit_costs:
                    continue
                arr = np.zeros(n_total)
                n_py = min(len(py), n_total)
                arr[:n_py] = py[:n_py]
                cost = arr * float(self.unit_costs[cost_key]) / disc_c
                total_cost += cost
                cost_details[f"{cost_key}_cost"] = cost
                logger.info(
                    "  - %s: %s for %s person-years",
                    cost_key,
                    f"{cost.sum():,.2f}",
                    f"{float(arr.sum()):,.2f}",
                )

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

        logger.info("Finalized MicrocostingAnalyzer")
        logger.info("  -> Total cost: $%s", f"{total_cost.sum():,.2f}")
        logger.info("  -> Total YLD: %s", f"{total_yld.sum():,.2f}")
        logger.info("  -> Total YLL: %s", f"{total_yll.sum():,.2f}")
        logger.info("  -> Total DALY: %s", f"{df['total_daly'].sum():,.2f}")

        # Store summary results for programmatic access (mutate container; do not replace it)
        try:
            self.results["total_cost"] = total_cost.sum()
            self.results["total_yld"] = total_yld.sum()
            self.results["total_yll"] = total_yll.sum()
            self.results["total_daly"] = (total_yld + total_yll).sum()
            self.results["detailed_outputs"] = self.detailed_outputs
        except Exception:
            self._summary = {
                "total_cost": float(total_cost.sum()),
                "total_yld": float(total_yld.sum()),
                "total_yll": float(total_yll.sum()),
                "total_daly": float((total_yld + total_yll).sum()),
            }

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
        # NOTE: BudgetConstraint-based accounting removed from non-economic package.
        return

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
