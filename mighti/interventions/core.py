"""
Defines interventions.
"""


import inspect
import pandas as pd
import starsim as ss
import stisim as sti
import numpy as np
import logging

from mighti.util.rng import get_rng

logger = logging.getLogger(__name__)


def _stisim_art_accepts_coverage_kw():
    """True when installed STIsim ART uses ``coverage=`` (STIsim >= 1.5)."""
    return "coverage" in inspect.signature(sti.ART.__init__).parameters


def _coverage_to_legacy_dataframe(coverage, value_col="p_art", default_year=2000):
    """Convert scalar/dict coverage targets to the DataFrame STIsim <1.5 expects."""
    if coverage is None:
        return None
    if isinstance(coverage, pd.DataFrame):
        return coverage
    if isinstance(coverage, (int, float)):
        return pd.DataFrame({value_col: [float(coverage)]}, index=[default_year])
    if isinstance(coverage, dict) and "year" in coverage and "value" in coverage:
        return pd.DataFrame({value_col: coverage["value"]}, index=coverage["year"])
    return coverage


def _prepare_art_init(pars=None, coverage=None, coverage_data=None, **kwargs):
    """Map legacy ART constructor names to the installed STIsim API."""
    pars = dict(pars) if pars else {}
    kwargs = dict(kwargs)

    if coverage_data is not None and coverage is None:
        coverage = coverage_data

    if _stisim_art_accepts_coverage_kw():
        if "init_prob" in pars:
            pars["art_initiation"] = pars.pop("init_prob")
        if "future_coverage" in pars and coverage is None:
            fc = pars.pop("future_coverage")
            coverage = {"year": [fc["year"]], "value": [fc["prop"]]}
        return {"pars": pars, "coverage": coverage, "kwargs": kwargs}

    if coverage is not None:
        coverage_data = _coverage_to_legacy_dataframe(coverage)
    elif coverage_data is not None and not isinstance(coverage_data, pd.DataFrame):
        coverage_data = _coverage_to_legacy_dataframe(coverage_data)
    if "art_initiation" in pars:
        pars["init_prob"] = pars.pop("art_initiation")
    return {"pars": pars, "coverage_data": coverage_data, "kwargs": kwargs}


def _call_stisim_art_init(self, pars=None, coverage=None, coverage_data=None, **kwargs):
    prepared = _prepare_art_init(pars, coverage, coverage_data, **kwargs)
    if _stisim_art_accepts_coverage_kw():
        sti.ART.__init__(self, pars=prepared["pars"], coverage=prepared.get("coverage"), **prepared["kwargs"])
    else:
        sti.ART.__init__(
            self,
            pars=prepared["pars"],
            coverage_data=prepared.get("coverage_data"),
            **prepared["kwargs"],
        )
    raw = getattr(self, "_raw_coverage", None)
    if raw is None:
        raw = getattr(self, "coverage_data", None)
    if raw is not None:
        self.coverage_data = raw

__all__ = [
    "ART",
    "ARTwithCASM",
    "ARTNoAutoAdjust",
    "ImproveHospitalDischarge",
    "GiveHousingToDepressed",
    "GiveHousingSupport",
    "HousingSupportForAUD",
]
# Back-compat (older code may have referenced this private name)
_all_ = __all__


class ART(sti.ART):
    """
    ART intervention.
    """

    def __init__(self, pars=None, coverage=None, coverage_data=None, **kwargs):
        _call_stisim_art_init(self, pars=pars, coverage=coverage, coverage_data=coverage_data, **kwargs)

    def init_pre(self, sim):
        super().init_pre(sim)

    # NOTE: economic/budget hooks intentionally removed


class ARTwithCASM(sti.ART):
    def __init__(self, pars=None, coverage=None, coverage_data=None, **kwargs):
        _call_stisim_art_init(self, pars=pars, coverage=coverage, coverage_data=coverage_data, **kwargs)
        self.casm_sensitivity = "pharma"


class ARTNoAutoAdjust(sti.ART):
    """
    ART intervention with auto-adjustment disabled.
    
    By default, sti.ART may auto-adjust coverage based on current on_art status
    to maintain target coverage levels. This class disables that behavior by
    returning the target coverage without correction.
    
    Also fixes the issue where sti.ART calculates coverage based on all infected
    individuals instead of diagnosed individuals.
    """
    
    def __init__(self, pars=None, coverage=None, coverage_data=None, **kwargs):
        _call_stisim_art_init(self, pars=pars, coverage=coverage, coverage_data=coverage_data, **kwargs)
        self._debug_coverage_calls = []
        self._debug_apply_calls = []
        # Track if art_coverage_correction was called
        self._correction_called = False
        self._last_correction_value = None
    
    def art_coverage_correction(self, sim, target_coverage=None):
        """
        Override to disable auto-adjustment and ensure coverage is calculated correctly.
        
        CRITICAL: This method also excludes people who have dropped out (_ever_dropped)
        from being re-added to ART, which is essential for making dropout effects visible.
        """
        # Get the raw coverage proportion from coverage_data by interpolating
        if hasattr(self, 'coverage_data') and self.coverage_data is not None:
            # Interpolate coverage proportion for current year
            year = sim.t.year
            years = self.coverage_data.index.values
            props = self.coverage_data['p_art'].values
            coverage_prop = np.interp(year, years, props)
            # Cap at the maximum value in the data to prevent extrapolation beyond observed data
            max_coverage = props.max()
            coverage_prop = min(coverage_prop, max_coverage)
        else:
            # Fallback: if no coverage_data, use target_coverage as-is
            coverage_prop = None
        
        # Get _ever_dropped set to exclude from re-initiation
        ever_dropped = set()
        art_dropout_connector = None
        if hasattr(sim, 'connectors'):
            if isinstance(sim.connectors, dict):
                # Try multiple possible keys
                art_dropout_connector = sim.connectors.get("artadherencedisruptor", None)
                if art_dropout_connector is None:
                    art_dropout_connector = sim.connectors.get("adherence_art_dropout", None)
            else:
                # It's a list, search by label
                for conn in sim.connectors:
                    if hasattr(conn, 'label'):
                        label_lower = conn.label.lower()
                        if ('adherence' in label_lower and 'art' in label_lower) or 'artadherencedisruptor' in label_lower:
                            art_dropout_connector = conn
                            break
            
            if art_dropout_connector is not None and hasattr(art_dropout_connector, "_ever_dropped"):
                ever_dropped = art_dropout_connector._ever_dropped
                # Debug: verify we found it
                if sim.ti % 12 == 0 and len(ever_dropped) > 0:  # Print once per year if there are dropped people
                    logger.debug(f"[ARTNoAutoAdjust.art_coverage_correction] Found connector '{art_dropout_connector.label}', _ever_dropped={len(ever_dropped)}")
            elif sim.ti % 12 == 0:  # Debug: print if connector not found
                connector_labels = []
                if hasattr(sim, 'connectors'):
                    if isinstance(sim.connectors, dict):
                        connector_labels = list(sim.connectors.keys())
                    else:
                        connector_labels = [getattr(conn, 'label', 'no_label') for conn in sim.connectors]
                logger.warning(f"[ARTNoAutoAdjust.art_coverage_correction] Could not find ART dropout connector! Available connectors: {connector_labels}")
        
        # If target_coverage is provided and is an absolute number (likely > 1.0),
        # recalculate based on the actual eligible population (diagnosed HIV+)
        hiv = getattr(sim.diseases, "hiv", None)
        if hiv is not None and "hiv.diagnosed" in sim.people.states:
            diagnosed = np.asarray(sim.people.states.get("hiv.diagnosed", []), bool)
            n_diagnosed = diagnosed.sum()
            
            if n_diagnosed > 0 and coverage_prop is not None:
                # Recalculate target as proportion of diagnosed
                # This ensures we're applying coverage to the right population
                corrected_target = int(coverage_prop * n_diagnosed)
                
                # CRITICAL: Reduce target by the number of _ever_dropped people who are diagnosed but not on ART
                # This prevents the parent method from trying to add them back
                # We need to be aggressive here - permanently exclude _ever_dropped people from coverage calculations
                if len(ever_dropped) > 0:
                    on_art = np.asarray(sim.people.states.get("hiv.on_art", []), bool) if "hiv.on_art" in sim.people.states else np.zeros(len(sim.people), dtype=bool)
                    # Count how many _ever_dropped people are diagnosed
                    # CRITICAL: Count ALL _ever_dropped people who are diagnosed, regardless of whether they're on ART or not.
                    # If they're in _ever_dropped, they will drop out (or have dropped out), so they should be excluded
                    # from the eligible pool for coverage calculations.
                    ever_dropped_diagnosed = 0
                    for uid in ever_dropped:
                        if uid < len(diagnosed):
                            if diagnosed[uid]:
                                ever_dropped_diagnosed += 1
                    
                    # AGGRESSIVE: Exclude ALL _ever_dropped diagnosed people from eligible pool
                    # This ensures that _ever_dropped people are permanently excluded from coverage calculations
                    # The target should be based on eligible people (diagnosed - _ever_dropped), not all diagnosed
                    current_on_art = on_art.sum()
                    target_before_reduction = corrected_target
                    
                    # Calculate eligible diagnosed (diagnosed minus ALL _ever_dropped who are diagnosed)
                    # This is the true pool of people who can be on ART
                    eligible_diagnosed = n_diagnosed - ever_dropped_diagnosed
                    
                    # Recalculate target based on eligible diagnosed pool
                    # CRITICAL: The target should be based on coverage_prop * eligible_diagnosed.
                    # We should NOT use max(current_on_art, ...) because that prevents the target from decreasing
                    # when people drop out. If people drop out, current_on_art decreases, and the target should
                    # reflect that by being based on the smaller eligible pool.
                    # We only add people if current_on_art < corrected_target, never force removal.
                    if eligible_diagnosed > 0:
                        corrected_target = int(coverage_prop * eligible_diagnosed)
                        # Only add people if current is below target, never force removal
                        # So if corrected_target < current_on_art, that's fine - we just won't add anyone
                        # But we should still use corrected_target as the target (not max it)
                    else:
                        corrected_target = current_on_art
                    
                    if sim.ti % 12 == 0 and ever_dropped_diagnosed > 0:
                        logger.debug(
                            "[ARTNoAutoAdjust.art_coverage_correction] Year %s: Excluding %s _ever_dropped diagnosed people from coverage calculation "
                            "(diagnosed=%s, eligible=%s, target before=%s, after=%s, current_on_art=%s, _ever_dropped=%s)",
                            sim.t.year,
                            ever_dropped_diagnosed,
                            n_diagnosed,
                            eligible_diagnosed,
                            target_before_reduction,
                            corrected_target,
                            current_on_art,
                            len(ever_dropped),
                        )
                
                # Debug output
                if hasattr(sim, 't') and len(self._debug_coverage_calls) < 10:
                    self._debug_coverage_calls.append((sim.t.year, target_coverage, coverage_prop, n_diagnosed, corrected_target))
                    if len(self._debug_coverage_calls) <= 3 or sim.t.year in [2010, 2013, 2016, 2020]:
                        logger.debug(
                            "[ARTNoAutoAdjust.art_coverage_correction] Year %s: coverage_prop=%0.3f, diagnosed=%s, "
                            "target_count=%s (original=%0.0f), excluding %s dropped people",
                            sim.t.year,
                            coverage_prop,
                            n_diagnosed,
                            corrected_target,
                            float(target_coverage) if target_coverage is not None else float("nan"),
                            len(ever_dropped),
                        )
                
                # Track that correction was called and what value was used
                self._correction_called = True
                self._last_correction_value = corrected_target
                
                # Track how many people are on ART before correction
                hiv = getattr(sim.diseases, "hiv", None)
                n_on_art_before = hiv.on_art.sum() if hiv is not None else 0
                
                # CRITICAL: Instead of calling parent method, implement our own logic that explicitly excludes _ever_dropped
                # The parent method might bypass our prioritize_art override, so we need to do it ourselves
                if hiv is not None and corrected_target > n_on_art_before:
                    # Find all diagnosed people who are not on ART
                    diagnosed = np.asarray(sim.people.states.get("hiv.diagnosed", []), bool) if "hiv.diagnosed" in sim.people.states else np.zeros(len(sim.people), dtype=bool)
                    on_art = np.asarray(sim.people.states.get("hiv.on_art", []), bool) if "hiv.on_art" in sim.people.states else np.zeros(len(sim.people), dtype=bool)
                    
                    # Eligible: diagnosed, not on ART, and NOT in _ever_dropped
                    eligible_mask = diagnosed & ~on_art
                    eligible_uids = np.where(eligible_mask)[0]
                    
                    # Filter out _ever_dropped people
                    if len(ever_dropped) > 0:
                        eligible_uids = [uid for uid in eligible_uids if uid not in ever_dropped]
                    
                    # How many do we need to add?
                    n_needed = corrected_target - n_on_art_before
                    n_to_add = min(n_needed, len(eligible_uids))
                    
                    if n_to_add > 0:
                        # Use prioritize_art to add people (it will handle prioritization)
                        eligible_uids_ss = ss.uids(eligible_uids)
                        self.prioritize_art(sim, n=n_to_add, awaiting_art_uids=eligible_uids_ss)
                    
                    if sim.ti % 12 == 0 and len(ever_dropped) > 0:
                        n_excluded_from_eligible = len([uid for uid in np.where(eligible_mask)[0] if uid in ever_dropped])
                        logger.debug(
                            "[ARTNoAutoAdjust.art_coverage_correction] Year %s: Implemented own correction logic: eligible=%s "
                            "(excluded %s _ever_dropped), target=%s, current=%s, needed=%s, adding=%s",
                            sim.t.year,
                            len(eligible_uids),
                            n_excluded_from_eligible,
                            corrected_target,
                            n_on_art_before,
                            n_needed,
                            n_to_add,
                        )
                else:
                    # No correction needed or can't do it - call parent as fallback
                    super().art_coverage_correction(sim, target_coverage=corrected_target)
                
                # Track how many people were added by correction
                n_on_art_after = hiv.on_art.sum() if hiv is not None else 0
                n_added_by_correction = n_on_art_after - n_on_art_before
                
                # Debug: show what correction did
                if n_added_by_correction > 0 and sim.ti % 12 == 0:  # Print once per year
                    aud_affected = np.asarray(sim.people.states.get("alcoholusedisorder.affected", []), dtype=bool) if "alcoholusedisorder.affected" in sim.people.states else np.zeros(len(sim.people), dtype=bool)
                    on_art_after = np.asarray(sim.people.states.get("hiv.on_art", []), bool) if "hiv.on_art" in sim.people.states else np.zeros(len(sim.people), dtype=bool)
                    # Find who was just added (approximate - compare before/after)
                    # This is approximate but gives us an idea
                    if hiv is not None and len(aud_affected) == len(sim.people):
                        # Count AUD vs No-AUD among those on ART
                        aud_on_art = (on_art_after & aud_affected).sum()
                        noaud_on_art = (on_art_after & ~aud_affected).sum()
                        logger.debug(
                            "[ARTNoAutoAdjust.art_coverage_correction] Year %s: Added %s people via correction "
                            "(target=%s, on_art before=%s, after=%s, total on ART: AUD=%s, No-AUD=%s, _ever_dropped=%s)",
                            sim.t.year,
                            n_added_by_correction,
                            corrected_target,
                            n_on_art_before,
                            n_on_art_after,
                            aud_on_art,
                            noaud_on_art,
                            len(ever_dropped),
                        )
                
                return
            elif target_coverage is not None:
                # Use the target_coverage as-is if we can't recalculate
                super().art_coverage_correction(sim, target_coverage=target_coverage)
                return
        
        # Fallback: call parent with original target_coverage
        super().art_coverage_correction(sim, target_coverage=target_coverage)
    
    def step(self):
        """
        Override step() to fix the issue where sti.ART calculates coverage
        based on all infected individuals instead of diagnosed individuals.
        """
        sim = self.sim
        hiv = sim.diseases.hiv
        inf_uids = hiv.infected.uids
        diag_uids = hiv.diagnosed.uids  # Use diagnosed instead of all infected

        # Figure out how many people should be treated
        # FIX: Calculate based on diagnosed, not all infected
        if self.t.now('year') < self.pars.future_coverage['year']:
            if self.coverage is None:
                n_to_treat = 0
            else:
                if self.coverage_format == 'n_art':
                    n_to_treat = int(self.coverage[self.ti]/sim.pars.pop_scale)
                elif self.coverage_format == 'p_art':
                    # FIX: Use diagnosed count instead of all infected
                    n_to_treat = int(self.coverage[self.ti]*len(diag_uids))
        else:
            p_cov = self.pars.future_coverage['prop']
            # FIX: Use diagnosed count instead of all infected
            n_to_treat = int(p_cov*len(diag_uids))

        # Firstly, check who is stopping ART
        if hiv.on_art.any():
            stopping = hiv.on_art & (hiv.ti_stop_art <= self.ti)
            if stopping.any():
                stopping_uids = stopping.uids
                try:
                    hiv.stop_art(stopping_uids)
                    
                    # CRITICAL: Immediately add stopping UIDs to _ever_dropped to prevent re-initiation
                    # This must happen BEFORE art_coverage_correction() runs, otherwise people
                    # who just dropped out will be immediately re-added to reach target coverage
                    art_dropout_connector = None
                    if hasattr(sim, 'connectors'):
                        if isinstance(sim.connectors, dict):
                            art_dropout_connector = sim.connectors.get("artadherencedisruptor", None)
                            if art_dropout_connector is None:
                                art_dropout_connector = sim.connectors.get("adherence_art_dropout", None)
                        else:
                            for conn in sim.connectors:
                                if hasattr(conn, 'label'):
                                    label_lower = conn.label.lower()
                                    if ('adherence' in label_lower and 'art' in label_lower) or 'artadherencedisruptor' in label_lower:
                                        art_dropout_connector = conn
                                        break
                    
                    if art_dropout_connector is not None and hasattr(art_dropout_connector, "_ever_dropped"):
                        # Convert stopping_uids to a set and add to _ever_dropped
                        stopping_uids_set = set(stopping_uids) if hasattr(stopping_uids, '__iter__') else {stopping_uids}
                        art_dropout_connector._ever_dropped.update(stopping_uids_set)
                        if sim.ti % 12 == 0:  # Print once per year
                            logger.debug(
                                "[ARTNoAutoAdjust.step] Year %s: Stopped ART for %s people, added to _ever_dropped (total _ever_dropped=%s)",
                                sim.t.year,
                                len(stopping_uids_set),
                                len(art_dropout_connector._ever_dropped),
                            )
                except:
                    errormsg = f'Error stopping ART for {stopping_uids}'
                    raise ValueError(errormsg)

        # Next, see how many people we need to treat vs how many are already being treated
        on_art = hiv.on_art

        # A proportion of newly diagnosed agents onto ART will be willing to initiate ART
        diagnosed = hiv.ti_diagnosed == self.ti
        if len(diagnosed.uids):
            # First, filter by base init_prob
            dx_to_treat_base = self.pars.init_prob.filter(diagnosed.uids)
            
            # Debug: track initial counts
            aud_affected = np.asarray(sim.people.states.get("alcoholusedisorder.affected", []), dtype=bool)
            dx_uids = ss.uids(diagnosed)
            dx_has_aud = aud_affected[dx_uids] if aud_affected.any() else np.zeros(len(dx_uids), dtype=bool)
            n_diagnosed = len(dx_uids)
            n_diagnosed_aud = dx_has_aud.sum()
            n_base_willing = len(dx_to_treat_base)
            n_base_willing_aud = sum(1 for uid in dx_to_treat_base if dx_has_aud[list(dx_uids).index(uid)])
            
            # Reduce ART initiation probability for people with AUD
            # This is part of the INTERACTION: AUD affects ART initiation
            # Only apply when interaction is enabled (when adherence engine is present)
            # In "No interaction" scenarios, AUD should NOT affect ART initiation
            has_adherence_data = "adherence" in sim.people.states
            rng = get_rng(sim, salt=f"{self.__class__.__name__}:step")
            
            # Only apply AUD reduction if interaction is enabled (adherence engine present)
            if has_adherence_data and aud_affected.any():
                # Reduce initiation probability for AUD individuals
                # Use adherence-based reduction if available, otherwise use fixed reduction
                if has_adherence_data:
                    adher = np.asarray(sim.people.states["adherence"], float)
                    # For AUD individuals, reduce initiation prob based on adherence
                    # If adherence is 0.7, they should have much lower initiation prob
                    # Apply stronger reduction: use adherence^3 to make effect much more pronounced
                    # This means: adherence 0.7 -> 0.35 (65% reduction), adherence 0.5 -> 0.125 (87.5% reduction)
                    # This should create a much more visible difference in ART coverage
                    aud_init_prob = np.power(adher[dx_uids], 3.0)  # Cube makes reduction very strong
                    # Apply reduction only to AUD individuals
                    rand = rng.random(len(dx_uids))
                    # Keep people who pass both base init_prob AND (if AUD) the adherence-based prob
                    dx_to_treat = []
                    aud_filtered_out = 0
                    for i, uid in enumerate(dx_uids):
                        if uid in dx_to_treat_base:
                            if dx_has_aud[i]:
                                # AUD individual: apply adherence-based probability reduction
                                # This reduces ART initiation for people with AUD
                                if rand[i] < aud_init_prob[i]:
                                    dx_to_treat.append(uid)
                                else:
                                    aud_filtered_out += 1
                                    # Drop them (don't initiate ART)
                            else:
                                # Non-AUD: keep original probability
                                dx_to_treat.append(uid)
                    dx_to_treat = ss.uids(dx_to_treat)
                    
                    # Debug output
                    n_final_willing = len(dx_to_treat)
                    n_final_willing_aud = sum(1 for uid in dx_to_treat if dx_has_aud[list(dx_uids).index(uid)])
                    if n_diagnosed_aud > 0 and sim.ti % 2 == 0:  # Print every 2 timesteps
                        mean_adher_aud = adher[dx_uids][dx_has_aud].mean() if dx_has_aud.any() else 0.0
                        mean_init_prob_aud = aud_init_prob[dx_has_aud].mean() if dx_has_aud.any() else 0.0
                        logger.debug(
                            "[ARTNoAutoAdjust] Year %s: Diagnosed=%s (AUD=%s), Base willing=%s (AUD=%s), "
                            "After AUD reduction=%s (AUD=%s, filtered=%s), Mean adherence (AUD)=%0.3f, Mean init prob (AUD)=%0.3f",
                            sim.t.year,
                            n_diagnosed,
                            n_diagnosed_aud,
                            n_base_willing,
                            n_base_willing_aud,
                            n_final_willing,
                            n_final_willing_aud,
                            aud_filtered_out,
                            mean_adher_aud,
                            mean_init_prob_aud,
                        )
                else:
                    # No adherence data: simple 50% reduction for AUD
                    aud_reduction_factor = 0.5
                    rand = rng.random(len(dx_uids))
                    dx_to_treat = []
                    aud_filtered_out = 0
                    for i, uid in enumerate(dx_uids):
                        if uid in dx_to_treat_base:
                            if dx_has_aud[i]:
                                # AUD individual: 50% chance to keep
                                if rand[i] < aud_reduction_factor:
                                    dx_to_treat.append(uid)
                                else:
                                    aud_filtered_out += 1
                            else:
                                # Non-AUD: keep
                                dx_to_treat.append(uid)
                    dx_to_treat = ss.uids(dx_to_treat)
                    
                    # Debug output
                    n_final_willing = len(dx_to_treat)
                    if n_diagnosed_aud > 0 and sim.ti % 2 == 0:
                        logger.debug(
                            "[ARTNoAutoAdjust] Year %s: Diagnosed=%s (AUD=%s), Base willing=%s (AUD=%s), "
                            "After AUD reduction=%s (AUD filtered=%s)",
                            sim.t.year,
                            n_diagnosed,
                            n_diagnosed_aud,
                            n_base_willing,
                            n_base_willing_aud,
                            n_final_willing,
                            aud_filtered_out,
                        )
            elif not has_adherence_data:
                # No interaction enabled: AUD does NOT affect ART initiation
                # This is the "No interaction" scenario - diseases exist but don't interact
                dx_to_treat = dx_to_treat_base
                if n_diagnosed_aud > 0 and sim.ti % 2 == 0:
                    logger.debug(
                        "[ARTNoAutoAdjust] Year %s: NO INTERACTION - Diagnosed=%s (AUD=%s), Base willing=%s (AUD=%s) - "
                        "AUD does NOT reduce ART initiation",
                        sim.t.year,
                        n_diagnosed,
                        n_diagnosed_aud,
                        n_base_willing,
                        n_base_willing_aud,
                    )
            else:
                # No AUD in population, use base (no reduction needed)
                dx_to_treat = dx_to_treat_base

            # Figure out if there are treatment spots available and if so, prioritize newly diagnosed agents
            n_available_spots = n_to_treat - len(on_art.uids)
            if n_available_spots > 0:
                # Check if ARTAdherenceDisruptor dropped anyone this step or ever - exclude them from re-initiation
                dropped_this_step = set()
                ever_dropped = set()
                # Try to get the connector from sim.connectors (dict or list)
                art_dropout_connector = None
                if hasattr(sim, 'connectors'):
                    if isinstance(sim.connectors, dict):
                        # Try multiple possible keys
                        art_dropout_connector = sim.connectors.get("artadherencedisruptor", None)
                        if art_dropout_connector is None:
                            art_dropout_connector = sim.connectors.get("adherence_art_dropout", None)
                    else:
                        # List or other iterable
                        for conn in sim.connectors:
                            if hasattr(conn, 'label'):
                                label_lower = conn.label.lower()
                                if ('adherence' in label_lower and 'art' in label_lower) or 'artadherencedisruptor' in label_lower:
                                    art_dropout_connector = conn
                                    break
                    
                    if art_dropout_connector is not None:
                        if hasattr(art_dropout_connector, "_dropped_this_step"):
                            dropped_this_step = art_dropout_connector._dropped_this_step
                        if hasattr(art_dropout_connector, "_ever_dropped"):
                            ever_dropped = art_dropout_connector._ever_dropped
                        
                        total_excluded = len(dropped_this_step | ever_dropped)
                        if total_excluded > 0 and sim.ti % 12 == 0:  # Print once per year
                            # Get more detailed info about who's excluded
                            aud_affected = np.asarray(sim.people.states.get("alcoholusedisorder.affected", []), dtype=bool) if "alcoholusedisorder.affected" in sim.people.states else np.zeros(len(sim.people), dtype=bool)
                            excluded_aud = len([uid for uid in (dropped_this_step | ever_dropped) if uid < len(aud_affected) and aud_affected[uid]])
                            excluded_noaud = total_excluded - excluded_aud
                            logger.debug(
                                "[ARTNoAutoAdjust] Year %s: Excluding %s dropped agents from re-initiation "
                                "(this_step=%s, ever=%s, AUD=%s, No-AUD=%s)",
                                sim.t.year,
                                total_excluded,
                                len(dropped_this_step),
                                len(ever_dropped),
                                excluded_aud,
                                excluded_noaud,
                            )
                
                # Remove dropped agents from dx_to_treat to prevent immediate re-initiation
                # CRITICAL: This prevents people who dropped out from being immediately re-added,
                # which is key to making the dropout effect visible in coverage differences
                if dropped_this_step or ever_dropped:
                    dx_to_treat = ss.uids([uid for uid in dx_to_treat if uid not in dropped_this_step and uid not in ever_dropped])
                
                self.prioritize_art(sim, n=n_available_spots, awaiting_art_uids=dx_to_treat)

        # Apply correction to match ART coverage data:
        # The correction method will recalculate based on diagnosed, but we've already
        # calculated n_to_treat correctly above, so we can pass it through
        # CRITICAL: art_coverage_correction will exclude _ever_dropped people
        self.art_coverage_correction(sim, target_coverage=n_to_treat)
    
    def prioritize_art(self, sim, n, awaiting_art_uids):
        """
        Override to exclude _ever_dropped people from being re-added to ART.
        This ensures that people who dropped out due to low adherence (e.g., from AUD)
        are not immediately re-added, making the dropout effect visible.
        """
        # Get _ever_dropped set to exclude
        ever_dropped = set()
        art_dropout_connector = None
        if hasattr(sim, 'connectors'):
            if isinstance(sim.connectors, dict):
                # Try multiple possible keys
                art_dropout_connector = sim.connectors.get("artadherencedisruptor", None)
                if art_dropout_connector is None:
                    art_dropout_connector = sim.connectors.get("adherence_art_dropout", None)
            else:
                # It's a list, search by label
                for conn in sim.connectors:
                    if hasattr(conn, 'label'):
                        label_lower = conn.label.lower()
                        if ('adherence' in label_lower and 'art' in label_lower) or 'artadherencedisruptor' in label_lower:
                            art_dropout_connector = conn
                            break
            
            if art_dropout_connector is not None and hasattr(art_dropout_connector, "_ever_dropped"):
                ever_dropped = art_dropout_connector._ever_dropped
                # Debug: verify we found it (only print if there are dropped people to avoid spam)
                if sim.ti % 12 == 0 and len(ever_dropped) > 0:  # Print once per year if there are dropped people
                    logger.debug(
                        "[ARTNoAutoAdjust.prioritize_art] Found connector '%s', _ever_dropped=%s",
                        art_dropout_connector.label,
                        len(ever_dropped),
                    )
            elif sim.ti % 12 == 0 and len(awaiting_art_uids) > 0:  # Debug: print if connector not found and there are people awaiting
                connector_labels = []
                if hasattr(sim, 'connectors'):
                    if isinstance(sim.connectors, dict):
                        connector_labels = list(sim.connectors.keys())
                    else:
                        connector_labels = [getattr(conn, 'label', 'no_label') for conn in sim.connectors]
                logger.warning(
                    "[ARTNoAutoAdjust.prioritize_art] Could not find ART dropout connector! Available connectors: %s",
                    connector_labels,
                )
        
        # Exclude _ever_dropped people from awaiting_art_uids
        n_before = len(awaiting_art_uids)
        n_after = n_before  # Initialize in case ever_dropped is empty
        n_excluded = 0  # Initialize
        excluded_aud = 0  # Initialize
        excluded_noaud = 0  # Initialize
        
        # Debug: Always print when prioritize_art is called with _ever_dropped people (yearly)
        if sim.ti % 12 == 0 and len(ever_dropped) > 0:
            logger.debug(
                "[ARTNoAutoAdjust.prioritize_art] Year %s: prioritize_art called with n=%s, awaiting=%s, _ever_dropped=%s",
                sim.t.year,
                n,
                n_before,
                len(ever_dropped),
            )
        
        if ever_dropped:
            # Get detailed info about who's being excluded
            aud_affected = np.asarray(sim.people.states.get("alcoholusedisorder.affected", []), dtype=bool) if "alcoholusedisorder.affected" in sim.people.states else np.zeros(len(sim.people), dtype=bool)
            awaiting_list = list(awaiting_art_uids)
            excluded_list = [uid for uid in awaiting_list if uid in ever_dropped]
            excluded_aud = len([uid for uid in excluded_list if uid < len(aud_affected) and aud_affected[uid]])
            excluded_noaud = len(excluded_list) - excluded_aud
            
            awaiting_art_uids = ss.uids([uid for uid in awaiting_list if uid not in ever_dropped])
            n_after = len(awaiting_art_uids)
            n_excluded = n_before - n_after
            
            # Always print when exclusion happens (yearly)
            if sim.ti % 12 == 0:
                if n_excluded > 0:
                    logger.debug(
                        "[ARTNoAutoAdjust.prioritize_art] Year %s: Excluded %s dropped people from prioritize_art "
                        "(before=%s, after=%s, _ever_dropped=%s, AUD=%s, No-AUD=%s)",
                        sim.t.year,
                        n_excluded,
                        n_before,
                        n_after,
                        len(ever_dropped),
                        excluded_aud,
                        excluded_noaud,
                    )
                elif len(excluded_list) == 0 and n_before > 0:
                    # Debug: No overlap between awaiting and _ever_dropped
                    logger.debug(
                        "[ARTNoAutoAdjust.prioritize_art] Year %s: No overlap between awaiting_art_uids (%s) and _ever_dropped (%s)",
                        sim.t.year,
                        n_before,
                        len(ever_dropped),
                    )
        
        # Call parent method with filtered UIDs
        # Track how many people are actually added by parent method
        hiv = getattr(sim.diseases, "hiv", None)
        n_on_art_before = hiv.on_art.sum() if hiv is not None else 0
        
        super().prioritize_art(sim, n=n, awaiting_art_uids=awaiting_art_uids)
        
        # Track how many were actually added
        n_on_art_after = hiv.on_art.sum() if hiv is not None else 0
        n_added = n_on_art_after - n_on_art_before
        
        if n_added > 0 and sim.ti % 12 == 0:  # Print once per year
            # Check how many of those added are AUD
            aud_affected = np.asarray(sim.people.states.get("alcoholusedisorder.affected", []), dtype=bool) if "alcoholusedisorder.affected" in sim.people.states else np.zeros(len(sim.people), dtype=bool)
            # Find who was just added (wasn't on ART before, is on ART now)
            if hiv is not None and len(aud_affected) == len(sim.people):
                on_art_after = np.asarray(sim.people.states.get("hiv.on_art", []), bool)
                # This is approximate - we can't perfectly track who was just added
                # But we can see if any AUD people are in the newly added group
                logger.debug(
                    "[ARTNoAutoAdjust.prioritize_art] Year %s: Added %s people to ART via prioritize_art "
                    "(n=%s, awaiting=%s, filtered to %s, _ever_dropped excluded=%s)",
                    sim.t.year,
                    n_added,
                    n,
                    n_before,
                    n_after,
                    n_excluded,
                )
    
    def apply(self, sim):
        """Override to add debug output."""
        # Call parent step (we've overridden step, not apply)
        self.step()
        
        # Get state after applying for debug output
        ppl = sim.people
        st = ppl.states
        hiv = getattr(sim.diseases, "hiv", None)
        
        if hiv is not None and hasattr(sim, 't'):
            diagnosed = np.asarray(st.get("hiv.diagnosed", []), bool) if "hiv.diagnosed" in st else None
            on_art_after = np.asarray(st.get("hiv.on_art", []), bool) if "hiv.on_art" in st else np.zeros(len(ppl), dtype=bool)
            n_diagnosed = diagnosed.sum() if diagnosed is not None else 0
            n_on_art_after = on_art_after.sum()
            
            # Debug output for key years
            if len(self._debug_apply_calls) < 10 and (len(self._debug_apply_calls) <= 3 or sim.t.year in [2010, 2013, 2016, 2020]):
                coverage_prop_expected = np.interp(sim.t.year, self.coverage_data.index.values, self.coverage_data['p_art'].values) if hasattr(self, 'coverage_data') and self.coverage_data is not None else 0.0
                expected_on_art = int(coverage_prop_expected * n_diagnosed) if n_diagnosed > 0 else 0
                actual_coverage = n_on_art_after / n_diagnosed if n_diagnosed > 0 else 0.0
                gap = expected_on_art - n_on_art_after
                self._debug_apply_calls.append((sim.t.year, n_diagnosed, n_on_art_after, expected_on_art, actual_coverage))
                if len(self._debug_apply_calls) <= 3 or sim.t.year in [2010, 2013, 2016, 2020]:
                    correction_info = f", correction_returned={self._last_correction_value}" if self._correction_called and self._last_correction_value is not None else ", correction_not_called"
                    logger.debug(
                        "[ARTNoAutoAdjust.apply] Year %s: diagnosed=%s, on_art=%s, expected=%s (coverage=%0.3f), "
                        "actual_coverage=%0.3f, gap=%s%s",
                        sim.t.year,
                        n_diagnosed,
                        n_on_art_after,
                        expected_on_art,
                        coverage_prop_expected,
                        actual_coverage,
                        gap,
                        correction_info,
                    )
                # Reset tracking for next timestep
                self._correction_called = False
                self._last_correction_value = None


class ImproveHospitalDischarge(ss.Intervention):
    def __init__(self, disease_name, multiplier=2.0, start_day=0, end_day=None, label=None):
        super().__init__(label=label)
        self.disease_name = disease_name
        self.multiplier = multiplier
        self.start_day = start_day
        self.end_day = end_day

    def initialize(self, sim):
        self.sim = sim
        self.disease = sim.diseases[self.disease_name]

    def apply(self):
        ti = self.sim.ti
    
        # Always refresh the disease in case multiprocessing lost it
        if not hasattr(self, 'disease') or self.disease is None:
            try:
                self.disease = self.sim.diseases[self.disease_name]
            except KeyError:
                raise ValueError(f"Disease '{self.disease_name}' not found. Available: {self.sim.diseases.keys()}")
    
        active = self.start_day <= ti < (self.end_day if self.end_day is not None else float('inf'))
    
        if active:
            self.disease.pars.p_daily_discharge_multiplier = self.multiplier
        else:
            self.disease.pars.p_daily_discharge_multiplier = 1.0
    
    def step(self):
        self.apply()
    

class GiveHousingToDepressed(ss.Intervention):
    """
    Intervention that provides stable housing to individuals with Major Depressive Disorder
    who currently have unstable housing.
    """
    def __init__(self, coverage=0.5, start_day=0, label=None):
        super().__init__(label=label or "GiveHousingToDepressed")
        self.coverage = coverage
        self.start_day = start_day

    def initialize(self, sim):
        self.sim = sim

    def apply(self):
        sim = self.sim
        if sim.ti < self.start_day:
            return
        rng = get_rng(sim, salt=f"{self.__class__.__name__}:step")

        depression = sim.diseases.get('majordepressivedisorder', None)
        if depression is None or not hasattr(depression, 'affected'):
            logger.warning(
                "[GiveHousingToDepressed] MajorDepressiveDisorder not found or missing 'affected' at year %s",
                getattr(sim.t, "year", "?"),
            )
            return

        # Target depressed + unstably housed
        ppl = sim.people
        depressed = depression.affected
        housing_unstable = ~ppl.neighbourhood_situation
        target = depressed & housing_unstable

        # Apply intervention with given coverage
        target_uids = target.uids
        n = len(target_uids)
        mask = rng.random(n) < self.coverage
        to_house = target_uids[mask]        
        ppl.neighbourhood_situation[to_house] = True

        
    def step(self):
        self.apply()


class GiveHousingSupport(ss.Intervention):
    def __init__(self, coverage=0.5, start_year=None, start_day=None, label=None):
        super().__init__(label=label or "GiveHousingSupport")
        self.coverage   = coverage
        self.start_year = start_year
        self.start_day  = start_day  # optional: still support direct ti

    def initialize(self, sim):
        self.sim = sim
        if self.start_day is None:
            if self.start_year is None:
                self.start_day = 0
            else:
                # Convert calendar year to ti
                self.start_day = max(0, int(round(self.start_year - sim.pars['start'])))
        logger.debug("[GiveHousingSupport] %s: start_day=%s", self.label, self.start_day)

    def apply(self):
        sim = self.sim
        if sim.ti < self.start_day:
            return
        rng = get_rng(sim, salt=f"{self.__class__.__name__}:step")
        ppl = sim.people
        unstable = ~ppl.neighbourhood_situation
        adult = ppl.age >= 15
        target = unstable & adult
        uids = target.uids
        if len(uids):
            to_house = uids[rng.random(len(uids)) < self.coverage]
            ppl.neighbourhood_situation[to_house] = True
            logger.info(
                "[GiveHousingSupport] Year %s: %s housed %s / %s",
                float(sim.t.yearvec[sim.ti]),
                self.label,
                len(to_house),
                len(uids),
            )
    def step(self):
        self.apply()

class HousingSupportForAUD(ss.Intervention):
    """
    Provides supportive housing to adults with AUD who are unstably housed.
    Optionally reduces relapse risk after housing.
    """

    def __init__(self, coverage=0.5, start_year=2010, relapse_reduction=0.5, label=None):
        super().__init__(label=label or "HousingSupportForAUD")
        self.coverage = coverage
        self.start_year = start_year
        self.relapse_reduction = relapse_reduction

    def step(self):
        sim = self.sim
        current_year = sim.t.year
        if current_year < self.start_year:
            return
        rng = get_rng(sim, salt=f"{self.__class__.__name__}:step")

        ppl = sim.people
        aud = sim.diseases.alcoholusedisorder

        # Target: adults (≥15) who are unhoused and have AUD
        target = (~ppl.neighbourhood_situation) & aud.affected & (ppl.age >= 15)
        uids = target.uids
        if len(uids) == 0:
            return

        mask = rng.random(len(uids)) < self.coverage
        housed_uids = uids[mask]
        ppl.neighbourhood_situation[housed_uids] = True

        # Optional relapse protection
        if hasattr(aud, "relapse_rate"):
            aud.relapse_rate[housed_uids] *= self.relapse_reduction

        logger.info(
            "[HousingSupportForAUD] Year %0.1f: %s housed %s of %s eligible adults with AUD",
            float(current_year),
            self.label,
            len(housed_uids),
            len(uids),
        )
        
        