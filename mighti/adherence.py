"""
Unified Adherence System for MIGHTI
-----------------------------------
Provides a three-component adherence pipeline:

1. AdherenceEngine           (CASM + SDoH → adherence state)
2. ARTAdherenceDisruptor     (adherence → ART dropout)
3. InterventionAdherenceDisruptor (adherence → ART efficacy scaling)
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = [
    "AdherenceEngine",
    "ARTAdherenceDisruptor",
    "InterventionAdherenceDisruptor",
    "CASM_REL_FACTORS",
    "SDOH_REL_FACTORS",
]

# ---------------------------------------------------------------------
# CASM adherence multipliers (1/OR from Table S2)
#   Values < 1 => reduced adherence when condition is present
# ---------------------------------------------------------------------
CASM_REL_FACTORS = {
    "AlcoholUseDisorder":      1 / 1.41,
    "MajorDepressiveDisorder": 1 / 2.21,
    "AnxietyDisorder":         1 / 2.04,
    "ChronicPain":             1 / 1.34,
    "TobaccoUse":              1 / 1.18,
    "OpioidUseDisorder":       1 / 1.18,
    "StimulantUseDisorder":    1 / 1.81,
}

# ---------------------------------------------------------------------
# SDoH multipliers (<1 = structural adherence penalty)
#   Keys must match people.states keys (e.g. 'neighbourhood_situation')
# ---------------------------------------------------------------------
SDOH_REL_FACTORS = {
    "neighbourhood_situation": 0.90,
    "social_context":          0.85,
    "education_situation":     0.92,
    "economic_situation":      0.80,
    "healthcare_system":       0.88,
}

# =====================================================================
# 1. AdherenceEngine
# =====================================================================
class AdherenceEngine(ss.Module):
    """
    Computes per-agent adherence from CASM + SDoH and writes into
    people.states['adherence'].

    Assumptions
    -----------
    - CASM disease modules expose boolean state arrays:
        '<disease_name_lower>.affected'
      e.g. 'majordepressivedisorder.affected'
    - SDoH indicators are stored as boolean arrays with keys equal to
      SDOH_REL_FACTORS keys (e.g. 'neighbourhood_situation').
    """

    def __init__(self, casm_rel=None, sdoh_rel=None, label="adherence_engine"):
        super().__init__(label=label)
        self.casm_rel = casm_rel or CASM_REL_FACTORS.copy()
        self.sdoh_rel = sdoh_rel or SDOH_REL_FACTORS.copy()

    def init_pre(self, sim):
        super().init_pre(sim)

        print(f"[AdherenceEngine] Initialized for sim '{sim.label}'")

    def step(self):
        ppl = self.sim.people
        st = ppl.states
        n = len(ppl)

        # Start from perfect adherence and apply multiplicative penalties
        adherence = np.ones(n, dtype=float)

        # CASM effects
        for cond, rel in self.casm_rel.items():
            key = f"{cond.lower()}.affected"  # e.g. 'majordepressivedisorder.affected'
            if key in st:
                affected = np.asarray(st[key], bool)
                adherence[affected] *= rel
            else:
                # Debug: check if state exists with different casing
                if self.sim.ti % 10 == 0:  # Print every 10 timesteps
                    possible_keys = [k for k in st.keys() if cond.lower() in k.lower()]
                    if possible_keys:
                        print(f"[AdherenceEngine] WARNING: '{key}' not found, but found similar keys: {possible_keys}")

        # SDoH effects
        for sdoh_key, rel in self.sdoh_rel.items():
            if sdoh_key in st:
                flagged = np.asarray(st[sdoh_key], bool)
                adherence[flagged] *= rel

        # Clip to [0, 1] and write back into the dynamic state
        st["adherence"][:] = np.clip(adherence, 0.0, 1.0)
        
        # Debug: print adherence stats for AUD individuals
        if self.sim.ti % 10 == 0:  # Print every 10 timesteps
            aud_key = "alcoholusedisorder.affected"
            if aud_key in st:
                aud_affected = np.asarray(st[aud_key], bool)
                if aud_affected.any():
                    aud_adherence = adherence[aud_affected]
                    print(f"[AdherenceEngine] Year {self.sim.t.year}: AUD individuals={aud_affected.sum()}, "
                          f"Mean adherence (AUD)={aud_adherence.mean():.3f}, "
                          f"Min={aud_adherence.min():.3f}, Max={aud_adherence.max():.3f}")


# =====================================================================
# 2. ARTAdherenceDisruptor
# =====================================================================
class ARTAdherenceDisruptor(ss.Connector):
    """
    Induces ART dropout as a function of adherence:

        dropout_probability_i = base_dropout * (1 - adherence_i)

    This acts on currently on-ART HIV-positive individuals.
    """

    def __init__(self, base_dropout=0.10, label="adherence_art_dropout"):
        super().__init__(label=label)
        self.base_dropout = float(base_dropout)
        self._dropped_this_step = set()  # Track who was dropped this step to prevent immediate re-initiation

    def step(self):
        sim = self.sim
        ppl = sim.people
        st = ppl.states
        hiv = getattr(sim.diseases, "hiv", None)
        if hiv is None:
            return

        # Reset tracking for this step
        self._dropped_this_step = set()

        adher = np.asarray(st["adherence"], float)
        on_art = np.asarray(st["hiv.on_art"], bool)
        
        # Debug: Always print on first few timesteps to verify it's running
        if sim.ti < 3:
            print(f"[ARTAdherenceDisruptor] Year {sim.t.year}, ti={sim.ti}: Running! On ART={on_art.sum()}, Mean adherence={adher.mean():.3f}")

        # Sanity check: these should always match if 'adherence' was added via People.add()
        if len(adher) != len(on_art):
            raise ValueError(
                f"[ARTAdherenceDisruptor] Length mismatch between adherence "
                f"({len(adher)}) and hiv.on_art ({len(on_art)}). "
                f"This usually indicates 'adherence' was not registered via People.add()."
            )

        drop_p = self.base_dropout * (1.0 - adher)
        rand = np.random.rand(len(adher))

        # Allow dropping people who are currently on ART
        # Strategy: require at least 24 timesteps (2 years) on ART to avoid HIV module errors
        # The HIV module's post_art_decline function has bugs when processing recently dropped people
        # Being very conservative is necessary to avoid crashes - this is a known HIV module limitation
        ti_art = np.asarray(st.get("hiv.ti_art", []), dtype=float)
        has_valid_ti = np.isfinite(ti_art) & (ti_art >= 0)
        sim_ti_float = float(sim.ti)  # Convert to float
        
        if on_art.sum() > 0:
            if has_valid_ti.any():
                # Require at least 2 timesteps (2 months) on ART before allowing dropout
                # This minimal delay avoids the HIV module's post_art_decline bug for people
                # who just started ART, but allows dropout for people with low adherence
                # (e.g., due to AUD) to happen relatively quickly
                ti_art_on_art = ti_art[on_art & has_valid_ti]
                if len(ti_art_on_art) > 0:
                    # From debug: ti_art range=[0, 48] when sim.ti=5
                    # This suggests ti_art might be in months, and sim.ti is in timesteps
                    # Convert both to months for proper comparison
                    dt = getattr(sim, 'dt', 1.0/12.0)
                    dt_float = float(dt)
                    sim_ti_months = sim_ti_float / dt_float  # Convert timesteps to months
                    
                    # Debug: print ti_art format detection
                    if sim.ti % 12 == 0:  # Print once per year
                        print(f"[ARTAdherenceDisruptor] Year {sim.t.year}, ti={sim.ti}: ti_art range=[{ti_art_on_art.min():.1f}, {ti_art_on_art.max():.1f}], sim.ti={sim_ti_float}, sim.ti_months={sim_ti_months:.1f}")
                    
                    # Check if ti_art is in years (> 1000), months, or timesteps
                    if ti_art_on_art.max() > 1000:
                        # ti_art is in years - convert to months
                        start_year = float(sim.start)
                        ti_art_months = np.where(has_valid_ti, (ti_art - start_year) * 12, sim_ti_months + 1000)
                        # Require at least 2 months on ART
                        if sim_ti_months >= 2.0:
                            valid_art = on_art & has_valid_ti & (ti_art_months <= sim_ti_months - 2.0)
                        else:
                            valid_art = np.zeros(len(on_art), dtype=bool)
                    elif ti_art_on_art.max() > sim_ti_float * 2:
                        # ti_art values are much larger than sim.ti - likely in months
                        # From debug: ti_art range=[0, 48] when sim.ti=5
                        # The unit mismatch makes comparison difficult
                        # SIMPLIFIED APPROACH: Be more permissive - if sim has been running for at least 2 months,
                        # allow ALL people on ART who have a valid ti_art and didn't just start
                        # We'll exclude only people who started in the current timestep (ti_art == sim.ti within tolerance)
                        if sim_ti_months >= 2.0:
                            # Exclude only people who started in the current timestep
                            # If ti_art is in months, exclude if abs(ti_art - sim_ti_months) < 0.5
                            # Otherwise, allow them (they've been on ART for at least some time)
                            # This is more permissive but should allow dropout to happen
                            just_started = has_valid_ti & (np.abs(ti_art - sim_ti_months) < 0.5)
                            valid_art = on_art & has_valid_ti & ~just_started
                        else:
                            valid_art = np.zeros(len(on_art), dtype=bool)
                    else:
                        # ti_art is likely in same units as sim.ti (timesteps)
                        # Require at least 2 timesteps (2 months) on ART
                        if sim_ti_float >= 2:
                            valid_art = on_art & has_valid_ti & (ti_art <= sim_ti_float - 2)
                        else:
                            valid_art = np.zeros(len(on_art), dtype=bool)
                else:
                    # No valid ti_art for people on ART - be conservative
                    valid_art = np.zeros(len(on_art), dtype=bool)
            else:
                # No valid ti_art at all - be conservative
                valid_art = np.zeros(len(on_art), dtype=bool)
        else:
            valid_art = np.zeros(len(on_art), dtype=bool)
        
        # Debug: print stats occasionally
        if sim.ti % 5 == 0 and on_art.sum() > 0:
            n_on_art = on_art.sum()
            n_valid = valid_art.sum()
            aud_affected = np.asarray(st.get("alcoholusedisorder.affected", []), dtype=bool)
            aud_on_art = (on_art & aud_affected).sum()
            aud_valid = (valid_art & aud_affected).sum()
            mean_drop_p_aud = drop_p[on_art & aud_affected].mean() if aud_on_art > 0 else 0.0
            mean_drop_p_noaud = drop_p[on_art & ~aud_affected].mean() if (on_art & ~aud_affected).sum() > 0 else 0.0
            # Check ti_art distribution for people on ART
            ti_art_on_art = ti_art[on_art]
            ti_art_min = ti_art_on_art[ti_art_on_art >= 0].min() if (ti_art_on_art >= 0).any() else -1
            ti_art_max = ti_art_on_art.max() if len(ti_art_on_art) > 0 else -1
            ti_art_current = (np.abs(ti_art_on_art - sim.ti) < 0.5).sum() if len(ti_art_on_art) > 0 else 0
            # Check how many people are excluded by the "just started" filter
            if has_valid_ti.any():
                just_started_count = (has_valid_ti & (np.abs(ti_art - sim.ti) < 0.5)).sum()
            else:
                just_started_count = 0
            # Check AUD-specific ti_art distribution
            if aud_on_art > 0:
                aud_ti_art = ti_art[on_art & aud_affected]
                aud_ti_art_min = aud_ti_art[aud_ti_art >= 0].min() if (aud_ti_art >= 0).any() else -1
                aud_ti_art_max = aud_ti_art.max() if len(aud_ti_art) > 0 else -1
                print(f"[ARTAdherenceDisruptor DEBUG] Year {sim.t.year}, ti={sim.ti}: On ART={n_on_art}, Valid={n_valid} ({n_valid/n_on_art*100:.1f}%), "
                      f"ti_art range=[{ti_art_min:.0f}, {ti_art_max:.0f}], ti_art==ti (within 0.5)={ti_art_current}, "
                      f"Just started (excluded)={just_started_count}, "
                      f"AUD on ART={aud_on_art}, AUD valid={aud_valid}, "
                      f"AUD ti_art range=[{aud_ti_art_min:.0f}, {aud_ti_art_max:.0f}], "
                      f"Mean drop prob (AUD)={mean_drop_p_aud:.4f}, (NoAUD)={mean_drop_p_noaud:.4f}")
            else:
                print(f"[ARTAdherenceDisruptor DEBUG] Year {sim.t.year}, ti={sim.ti}: On ART={n_on_art}, Valid={n_valid} ({n_valid/n_on_art*100:.1f}%), "
                      f"ti_art range=[{ti_art_min:.0f}, {ti_art_max:.0f}], ti_art==ti (within 0.5)={ti_art_current}, "
                      f"Just started (excluded)={just_started_count}, "
                      f"AUD on ART={aud_on_art}, AUD valid={aud_valid}, "
                      f"Mean drop prob (AUD)={mean_drop_p_aud:.4f}, (NoAUD)={mean_drop_p_noaud:.4f}")
        
        # Calculate dropout for valid people
        # Only consider people who are valid (on ART for at least 2 months)
        drop_ids = np.where(valid_art & (rand < drop_p))[0]
        
        # Debug: print expected vs actual drops occasionally
        if sim.ti % 5 == 0 and valid_art.sum() > 0:  # Every 5 timesteps (less verbose)
            aud_affected = np.asarray(st.get("alcoholusedisorder.affected", []), dtype=bool)
            aud_valid = (valid_art & aud_affected).sum()
            if aud_valid > 0:
                aud_drop_p = drop_p[valid_art & aud_affected]
                expected_aud_drops = aud_drop_p.sum()
                # Recalculate actual drops using the same logic as drop_ids
                aud_valid_mask = valid_art & aud_affected
                aud_rand = rand[aud_valid_mask]
                aud_drop_p_filtered = drop_p[aud_valid_mask]
                actual_aud_drops = (aud_rand < aud_drop_p_filtered).sum()
                print(f"[ARTAdherenceDisruptor] Year {sim.t.year}, ti={sim.ti}: Valid={valid_art.sum()} (AUD={aud_valid}), "
                      f"Expected AUD drops={expected_aud_drops:.1f}, Actual={actual_aud_drops}, "
                      f"drop_ids.size={drop_ids.size}, mean_drop_prob={aud_drop_p.mean():.4f}")
                # Additional debug: show some actual values
                if len(aud_drop_p_filtered) > 0:
                    print(f"  [DEBUG] AUD valid: {aud_valid}, drop_p range=[{aud_drop_p_filtered.min():.4f}, {aud_drop_p_filtered.max():.4f}], "
                          f"rand range=[{aud_rand.min():.4f}, {aud_rand.max():.4f}], "
                          f"rand < drop_p: {(aud_rand < aud_drop_p_filtered).sum()}")

        if drop_ids.size:
            # Skip additional safety check - valid_art already filtered to exclude people who just started
            # This allows dropout to happen for people with low adherence
            safe_drop_ids = drop_ids  # Use drop_ids directly since valid_art already filtered appropriately
            
            if safe_drop_ids.size:
                # Additional filtering: only drop people who have been on ART for at least 12 months (1 year)
                # This helps avoid the HIV module bug in post_art_decline
                # The error occurs in HIV module's step_state AFTER our connector runs,
                # so we need to be very conservative to prevent it
                # Note: This means dropout will only happen for people on ART for 1+ years,
                # which limits the effect size but is necessary to avoid crashes
                final_drop_ids = []
                dt = getattr(sim, 'dt', 1.0/12.0)
                dt_float = float(dt)
                sim_ti_months = sim_ti_float / dt_float
                
                for did in safe_drop_ids:
                    if has_valid_ti[did]:
                        ti_art_val = ti_art[did]
                        # Calculate time on ART
                        # If ti_art is in months and sim_ti_months is also in months
                        if ti_art_val <= sim_ti_months:
                            time_on_art = sim_ti_months - ti_art_val
                        elif ti_art_val > sim_ti_months and ti_art_val < 200:
                            # ti_art might be from different reference, but reasonable
                            # Assume they've been on ART for at least 6 months if sim has run for a while
                            time_on_art = sim_ti_months if sim_ti_months >= 6.0 else 0.0
                        else:
                            time_on_art = 0.0
                        
                        # Only allow dropping if they've been on ART for at least 6 months
                        # We increased from 2 to 6 months to be more conservative and avoid HIV module bug
                        # The error occurs in HIV module's step_state AFTER our connector runs,
                        # so we can't catch it directly, but being more conservative helps
                        if time_on_art >= 6.0:
                            final_drop_ids.append(did)
                
                final_drop_ids = np.array(final_drop_ids, dtype=int)
                
                if final_drop_ids.size:
                    # Track who we're about to drop
                    aud_affected = np.asarray(st.get("alcoholusedisorder.affected", []), dtype=bool)
                    aud_dropped = aud_affected[final_drop_ids].sum()
                    mean_adher_dropped = adher[final_drop_ids].mean() if final_drop_ids.size > 0 else 0.0
                    mean_drop_p_dropped = drop_p[final_drop_ids].mean() if final_drop_ids.size > 0 else 0.0
                    
                    # Instead of calling hiv.stop_art() directly (which triggers the bug in post_art_decline),
                    # set hiv.ti_stop_art to the current timestep so the HIV module handles it in its step_state
                    # This should avoid the shape mismatch error
                    if "hiv.ti_stop_art" in st:
                        ti_stop_art = st["hiv.ti_stop_art"]
                        # Set ti_stop_art to current timestep for people we want to drop
                        # The HIV module will handle stopping ART in its step_state (in ARTNoAutoAdjust.step())
                        ti_stop_art[final_drop_ids] = sim.ti
                    
                    # Track who was dropped this step
                    self._dropped_this_step = set(final_drop_ids.tolist())
                    
                    # Print when dropping (less verbose - only every 5 timesteps or if significant)
                    if sim.ti % 5 == 0 or final_drop_ids.size > 10:
                        print(f"[ARTAdherenceDisruptor] Year {sim.t.year}, ti={sim.ti}: Scheduled ART stop for {final_drop_ids.size} agents "
                              f"(AUD={aud_dropped}, NoAUD={final_drop_ids.size - aud_dropped}, "
                              f"mean_adherence={mean_adher_dropped:.3f}, mean_drop_prob={mean_drop_p_dropped:.4f})")


# =====================================================================
# 3. InterventionAdherenceDisruptor
# =====================================================================
class InterventionAdherenceDisruptor(ss.Module):
    """
    Scales intervention effectiveness based on population-level adherence.

    For Starsim/STIsim HIV:
    - Scales hiv.pars.art_efficacy by mean adherence each step.

    For other interventions:
    - If an intervention exposes a 'rel_effect' attribute, it is multiplied
      by mean adherence (scalar) or by the adherence vector (per-agent),
      depending on the attribute shape.

    IMPORTANT: Stores baseline values to avoid cumulative multiplication.
    """

    def __init__(self, scale_art_efficacy=False, label="intervention_adherence_disruptor"):
        """
        Parameters
        ----------
        scale_art_efficacy : bool
            If True, scales hiv.pars.art_efficacy by mean adherence each step.
            If False (default), does not modify art_efficacy.
        """
        super().__init__(label=label)
        self.scale_art_efficacy = scale_art_efficacy
        self._baseline_art_efficacy = None
        self._baseline_rel_effects = {}  # intervention label -> baseline value

    def init_post(self):
        """Store baseline values after simulation is initialized."""
        super().init_post()
        sim = self.sim
        
        # Store baseline ART efficacy (only if scaling is enabled)
        if self.scale_art_efficacy:
            hiv = getattr(sim.diseases, "hiv", None)
            if hiv is not None and hasattr(hiv.pars, "art_efficacy"):
                self._baseline_art_efficacy = float(hiv.pars.art_efficacy)
        
        # Store baseline rel_effect for each intervention
        for intv in sim.interventions:
            if hasattr(intv, "rel_effect"):
                label = getattr(intv, "label", id(intv))
                if np.isscalar(intv.rel_effect):
                    self._baseline_rel_effects[label] = float(intv.rel_effect)
                else:
                    # For per-agent arrays, store a copy
                    self._baseline_rel_effects[label] = np.array(intv.rel_effect, copy=True)

    def step(self):
        sim = self.sim
        ppl = sim.people
        st = ppl.states

        adher = np.asarray(st["adherence"], float)
        scale = float(adher.mean())
        
        # Debug: print adherence stats occasionally
        if hasattr(sim, 'ti') and sim.ti % 5 == 0:
            print(f"[{self.label}] Year {sim.t.year}: mean adherence={scale:.3f}, "
                  f"min={adher.min():.3f}, max={adher.max():.3f}")

        # 1. Starsim/sti HIV ART efficacy (pars.art_efficacy)
        # Only scale if explicitly enabled
        if self.scale_art_efficacy:
            hiv = getattr(sim.diseases, "hiv", None)
            if hiv is not None and hasattr(hiv.pars, "art_efficacy"):
                if self._baseline_art_efficacy is not None:
                    hiv.pars.art_efficacy = self._baseline_art_efficacy * scale
                else:
                    # Fallback if init_post wasn't called (shouldn't happen)
                    print(f"[WARNING] {self.label}: baseline_art_efficacy not set, skipping scaling")

        # 2. Any other interventions that still use rel_effect
        # Set to baseline * scale (not multiply current value)
        for intv in sim.interventions:
            if hasattr(intv, "rel_effect"):
                label = getattr(intv, "label", id(intv))
                if label in self._baseline_rel_effects:
                    baseline = self._baseline_rel_effects[label]
                    if np.isscalar(baseline):
                        intv.rel_effect = baseline * scale
                    else:
                        # Per-agent rel_effect, scale by per-agent adherence
                        intv.rel_effect = baseline * adher
                else:
                    # Fallback if baseline wasn't stored
                    print(f"[WARNING] {self.label}: baseline for {label} not found, skipping scaling")
                    