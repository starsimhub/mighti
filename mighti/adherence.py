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

        # Only consider people who are on ART and have been on ART for at least one timestep
        # Check if they have valid ART start time (ti_art)
        ti_art = np.asarray(st.get("hiv.ti_art", []), dtype=float)
        has_valid_ti = np.isfinite(ti_art) & (ti_art >= 0)
        
        # Allow dropping anyone who is currently on ART
        # But exclude people who just started ART this timestep to avoid HIV module errors
        # Check ti_art to filter out people who started ART in the current timestep
        ti_art = np.asarray(st.get("hiv.ti_art", []), dtype=float)
        has_valid_ti = np.isfinite(ti_art) & (ti_art >= 0)
        # Only drop people who have been on ART for at least 1 timestep
        # Use a simple check: if ti_art is much larger than sim.ti, it might be in years
        # Otherwise, assume it's in timesteps
        if on_art.sum() > 0 and has_valid_ti.any():
            ti_art_sample = ti_art[on_art & has_valid_ti]
            if len(ti_art_sample) > 0:
                # If max ti_art is > 1000, assume years; convert to timestep
                if ti_art_sample.max() > 1000:
                    start_year = float(sim.start)
                    ti_art_timestep = np.where(has_valid_ti, ti_art - start_year, sim.ti + 1)
                else:
                    ti_art_timestep = ti_art
                # Only drop people who started ART before current timestep
                valid_art = on_art & has_valid_ti & (ti_art_timestep < sim.ti)
            else:
                valid_art = on_art & has_valid_ti & (ti_art < sim.ti)
        else:
            # Fallback: only drop if we have valid ti_art
            valid_art = on_art & has_valid_ti & (ti_art < sim.ti) if has_valid_ti.any() else np.zeros(len(on_art), dtype=bool)
        
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
            ti_art_current = (ti_art_on_art == sim.ti).sum() if len(ti_art_on_art) > 0 else 0
            # Check AUD-specific ti_art distribution
            if aud_on_art > 0:
                aud_ti_art = ti_art[on_art & aud_affected]
                aud_ti_art_valid = aud_ti_art[(aud_ti_art >= 0) & (aud_ti_art <= sim.ti - 1)]
                aud_ti_art_min = aud_ti_art[aud_ti_art >= 0].min() if (aud_ti_art >= 0).any() else -1
                aud_ti_art_max = aud_ti_art.max() if len(aud_ti_art) > 0 else -1
                aud_ti_art_valid_count = len(aud_ti_art_valid)
                print(f"[ARTAdherenceDisruptor DEBUG] Year {sim.t.year}, ti={sim.ti}: On ART={n_on_art}, Valid={n_valid}, "
                      f"ti_art range=[{ti_art_min:.0f}, {ti_art_max:.0f}], ti_art==ti={ti_art_current}, "
                      f"AUD on ART={aud_on_art}, AUD valid={aud_valid}, "
                      f"AUD ti_art range=[{aud_ti_art_min:.0f}, {aud_ti_art_max:.0f}], AUD ti_art valid={aud_ti_art_valid_count}, "
                      f"Mean drop prob (AUD)={mean_drop_p_aud:.4f}, (NoAUD)={mean_drop_p_noaud:.4f}")
            else:
                print(f"[ARTAdherenceDisruptor DEBUG] Year {sim.t.year}, ti={sim.ti}: On ART={n_on_art}, Valid={n_valid}, "
                      f"ti_art range=[{ti_art_min:.0f}, {ti_art_max:.0f}], ti_art==ti={ti_art_current}, "
                      f"AUD on ART={aud_on_art}, AUD valid={aud_valid}, "
                      f"Mean drop prob (AUD)={mean_drop_p_aud:.4f}, (NoAUD)={mean_drop_p_noaud:.4f}")
        
        drop_ids = np.where(valid_art & (rand < drop_p))[0]
        
        # Debug: print expected vs actual drops occasionally
        if sim.ti % 5 == 0 and valid_art.sum() > 0:  # Every 5 timesteps (less verbose)
            aud_affected = np.asarray(st.get("alcoholusedisorder.affected", []), dtype=bool)
            aud_valid = (valid_art & aud_affected).sum()
            if aud_valid > 0:
                aud_drop_p = drop_p[valid_art & aud_affected]
                expected_aud_drops = aud_drop_p.sum()
                actual_aud_drops = (valid_art & aud_affected & (rand < drop_p)).sum()
                print(f"[ARTAdherenceDisruptor] Year {sim.t.year}, ti={sim.ti}: Valid={valid_art.sum()} (AUD={aud_valid}), "
                      f"Expected AUD drops={expected_aud_drops:.1f}, Actual={actual_aud_drops}, "
                      f"drop_ids.size={drop_ids.size}, mean_drop_prob={aud_drop_p.mean():.4f}")

        if drop_ids.size:
            try:
                # Track who we're about to drop
                aud_affected = np.asarray(st.get("alcoholusedisorder.affected", []), dtype=bool)
                aud_dropped = aud_affected[drop_ids].sum()
                mean_adher_dropped = adher[drop_ids].mean() if drop_ids.size > 0 else 0.0
                mean_drop_p_dropped = drop_p[drop_ids].mean() if drop_ids.size > 0 else 0.0
                
                hiv.stop_art(drop_ids)
                # Track who was dropped this step
                self._dropped_this_step = set(drop_ids.tolist())
                
                # Always print when dropping (for debugging)
                print(f"[ARTAdherenceDisruptor] Year {sim.t.year}, ti={sim.ti}: Dropped ART for {drop_ids.size} agents "
                      f"(AUD={aud_dropped}, NoAUD={drop_ids.size - aud_dropped}, "
                      f"mean_adherence={mean_adher_dropped:.3f}, mean_drop_prob={mean_drop_p_dropped:.4f})")
            except (ValueError, IndexError) as e:
                # Handle errors from HIV module when processing post-ART decline
                # This can happen when dropping people - it's a known issue with the HIV module
                error_msg = str(e)
                if "post-ART duration" in error_msg or "Invalid entry" in error_msg or "shape mismatch" in error_msg or "broadcast" in error_msg:
                    # Try dropping only people who started ART at least 2 timesteps ago
                    if has_valid_ti.any():
                        very_safe_drop = []
                        for did in drop_ids:
                            if has_valid_ti[did] and ti_art[did] < sim.ti - 1:
                                very_safe_drop.append(did)
                        very_safe_drop = np.array(very_safe_drop, dtype=int)
                        if very_safe_drop.size:
                            try:
                                hiv.stop_art(very_safe_drop)
                                self._dropped_this_step = set(very_safe_drop.tolist())
                                aud_affected = np.asarray(st.get("alcoholusedisorder.affected", []), dtype=bool)
                                aud_dropped = aud_affected[very_safe_drop].sum()
                                print(f"[ARTAdherenceDisruptor] Year {sim.t.year}, ti={sim.ti}: Dropped ART for {very_safe_drop.size} agents "
                                      f"(AUD={aud_dropped}, conservative filter due to HIV module error)")
                            except (ValueError, IndexError):
                                # If still failing, skip this timestep
                                if sim.ti % 5 == 0:
                                    print(f"[ARTAdherenceDisruptor] Year {sim.t.year}: Could not drop ART (HIV module state issue)")
                    # Don't raise - this is expected when HIV module has edge cases
                else:
                    # Re-raise if it's a different error
                    if sim.ti % 5 == 0:
                        print(f"[ARTAdherenceDisruptor] Year {sim.t.year}: Unexpected error: {error_msg}")
                    raise


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
                    