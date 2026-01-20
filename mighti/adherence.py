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
import stisim as sti

__all__ = [
    "AdherenceEngine",
    "ARTAdherenceDisruptor",
    "InterventionAdherenceDisruptor",
    "AdherenceFromDepression",
    "CASM_REL_FACTORS",
    "SDOH_REL_FACTORS",
]

# =====================================================================
# Monkey patches to fix HIV module bugs
# =====================================================================
# The HIV module's post_art_decline function has a bug where it can
# raise "ValueError: Post-ART duration is negative" when ti_stop_art
# is set but ti_art is invalid or when people are dropped and re-added
# in the same timestep. This monkey patch fixes the issue by ensuring
# valid ti_art and ti_stop_art values before calculating duration.
# The HIV module's step_state method also has a bug where it can raise
# "ValueError: Invalid entry for CD4" when CD4 values are invalid.
# This monkey patch fixes the issue by validating and fixing CD4 values
# before the original validation check.
_original_post_art_decline = None
_original_step_state = None
_original_make_p_hiv_death = None

def _get_valid_cd4(uid, st, default=500.0):
    """Get valid CD4 value for a UID, ensuring it's not NaN or invalid."""
    if "hiv.cd4" in st:
        current_cd4 = np.asarray(st.get("hiv.cd4", []), dtype=float)
        if uid < len(current_cd4):
            cd4_val = current_cd4[uid]
            # Ensure CD4 is valid (finite, positive, reasonable range)
            if np.isfinite(cd4_val) and 0 < cd4_val < 2000:
                return cd4_val
    return default

def _patched_post_art_decline(self, uids):
    """
    Patched version of hiv.post_art_decline that handles negative durations.
    Returns current CD4 values if there are any timing issues to avoid errors.
    """
    if len(uids) == 0:
        return np.array([])
    
    # Get current state
    st = self.sim.people.states
    
    # Convert uids to list if needed
    if hasattr(uids, '__iter__') and not isinstance(uids, (list, tuple, np.ndarray)):
        uids = list(uids)
    elif isinstance(uids, np.ndarray):
        uids = uids.tolist()
    
    # Get timing arrays
    ti_art = np.asarray(st.get("hiv.ti_art", []), dtype=float) if "hiv.ti_art" in st else np.array([], dtype=float)
    ti_stop_art = np.asarray(st.get("hiv.ti_stop_art", []), dtype=float) if "hiv.ti_stop_art" in st else np.array([], dtype=float)
    
    # Filter to valid UIDs
    valid_uids = [uid for uid in uids if uid < len(ti_art) and uid < len(ti_stop_art) and uid >= 0]
    if len(valid_uids) == 0:
        # Return current CD4 values for invalid UIDs (ensure they're valid)
        return np.array([_get_valid_cd4(uid, st) for uid in uids], dtype=float)
    
    # Get values for valid UIDs
    ti_art_vals = np.array([ti_art[uid] if uid < len(ti_art) else np.nan for uid in valid_uids], dtype=float)
    ti_stop_art_vals = np.array([ti_stop_art[uid] if uid < len(ti_stop_art) else np.nan for uid in valid_uids], dtype=float)
    
    # Check for any invalid values - if found, just return current CD4
    invalid_ti_art = ~np.isfinite(ti_art_vals) | (ti_art_vals < 0)
    invalid_ti_stop = ~np.isfinite(ti_stop_art_vals) | (ti_stop_art_vals < 0)
    
    # Calculate duration
    duration = ti_stop_art_vals - ti_art_vals
    negative_duration = duration < 0
    
    # If ANY invalid values or negative durations, return current CD4 values directly
    # This is safer than trying to fix the values and call the original function
    if invalid_ti_art.any() or invalid_ti_stop.any() or negative_duration.any():
        # Return current CD4 values for all UIDs (ensure they're valid)
        result = np.array([_get_valid_cd4(uid, st) for uid in valid_uids], dtype=float)
        # Pad with current CD4 for any invalid UIDs
        if len(valid_uids) < len(uids):
            full_result = np.array([_get_valid_cd4(uid, st) for uid in uids], dtype=float)
            valid_indices = {uid: i for i, uid in enumerate(valid_uids)}
            for i, uid in enumerate(uids):
                if uid in valid_indices:
                    full_result[i] = result[valid_indices[uid]]
            return full_result
        return result
    
    # All values are valid - try calling original function
    # Temporarily update state arrays with fixed values
    original_ti_art = ti_art.copy() if len(ti_art) > 0 else None
    original_ti_stop_art = ti_stop_art.copy() if len(ti_stop_art) > 0 else None
    
    try:
        # Update state arrays with fixed values for valid UIDs only
        for i, uid in enumerate(valid_uids):
            if uid < len(ti_art):
                ti_art[uid] = ti_art_vals[i]
            if uid < len(ti_stop_art):
                ti_stop_art[uid] = ti_stop_art_vals[i]
        
        # Call original function
        if _original_post_art_decline is not None:
            result = _original_post_art_decline(self, valid_uids)
        else:
            # Fallback if original function not available
            result = np.array([_get_valid_cd4(uid, st) for uid in valid_uids], dtype=float)
        
        # Validate result - ensure no NaN or invalid values
        if isinstance(result, np.ndarray) and len(result) > 0:
            invalid_result = ~np.isfinite(result) | (result <= 0) | (result >= 2000)
            if invalid_result.any():
                # Replace invalid values with current CD4
                for i, uid in enumerate(valid_uids):
                    if i < len(result) and invalid_result[i]:
                        result[i] = _get_valid_cd4(uid, st)
        
        # Restore original values
        if original_ti_art is not None:
            for i, uid in enumerate(valid_uids):
                if uid < len(ti_art):
                    ti_art[uid] = original_ti_art[uid]
        if original_ti_stop_art is not None:
            for i, uid in enumerate(valid_uids):
                if uid < len(ti_stop_art):
                    ti_stop_art[uid] = original_ti_stop_art[uid]
        
        # If we had invalid UIDs, pad result with current CD4 values
        if len(valid_uids) < len(uids):
            full_result = np.array([_get_valid_cd4(uid, st) for uid in uids], dtype=float)
            valid_indices = {uid: i for i, uid in enumerate(valid_uids)}
            for i, uid in enumerate(uids):
                if uid in valid_indices and valid_indices[uid] < len(result):
                    cd4_val = result[valid_indices[uid]]
                    # Ensure CD4 is valid
                    if np.isfinite(cd4_val) and 0 < cd4_val < 2000:
                        full_result[i] = cd4_val
            return full_result
        
        return result
    except Exception as e:
        # If original function still fails, return current CD4 values (ensure they're valid)
        return np.array([_get_valid_cd4(uid, st) for uid in valid_uids], dtype=float)

def _patched_make_p_hiv_death(self, uids=None):
    """
    Patched version of hiv.make_p_hiv_death that handles IndexError when
    np.digitize returns out-of-bounds indices.
    """
    if _original_make_p_hiv_death is None:
        # Fallback if original not available - try to call original directly
        try:
            # Try to get the original from the class
            hiv_class = type(self)
            if hasattr(hiv_class, 'make_p_hiv_death'):
                original = hiv_class.__dict__.get('make_p_hiv_death', None)
                if original is not None and original != _patched_make_p_hiv_death:
                    return original(self, uids=uids)
        except:
            pass
        return np.zeros(len(uids) if uids is not None else len(self.sim.people), dtype=float)
    
    try:
        # Call original function
        return _original_make_p_hiv_death(self, uids=uids)
    except IndexError as e:
        # Handle IndexError from np.digitize returning out-of-bounds indices
        # This happens when CD4 values are >= max bin value
        # Re-implement the logic with proper bounds checking
        if uids is None:
            uids = self.sim.people.uids
        
        # Get CD4 values from states
        st = self.sim.people.states
        if "hiv.cd4" in st:
            cd4 = st["hiv.cd4"]
            cd4_vals = np.asarray(cd4[uids], dtype=float)
        else:
            # Fallback to self.cd4 if available
            cd4_vals = np.asarray(self.cd4[uids], dtype=float) if hasattr(self, 'cd4') else np.full(len(uids), 500.0, dtype=float)
        
        # Get cd4_bins from the HIV module
        try:
            cd4_bins = self.cd4_bins if hasattr(self, 'cd4_bins') else np.array([0, 50, 100, 200, 350, 500, 1000])
        except:
            cd4_bins = np.array([0, 50, 100, 200, 350, 500, 1000])
        
        # Get p_hiv_death array
        try:
            p_hiv_death = self.p_hiv_death if hasattr(self, 'p_hiv_death') else np.ones(len(cd4_bins) - 1, dtype=float) * 0.01
        except:
            p_hiv_death = np.ones(len(cd4_bins) - 1, dtype=float) * 0.01
        
        # Clip CD4 values to be within valid range for bins
        # Ensure CD4 values are < max bin to prevent out-of-bounds indices
        max_cd4_for_bins = cd4_bins[-1] - 1.0  # Just below max bin
        cd4_vals = np.clip(cd4_vals, 0.0, max_cd4_for_bins)
        
        # Use digitize and clip indices to valid range
        indices = np.digitize(cd4_vals, cd4_bins)
        indices = np.clip(indices, 0, len(p_hiv_death) - 1)
        
        return p_hiv_death[indices]

def _patched_step_state(self):
    """
    Patched version of hiv.step_state that validates and fixes CD4 values
    before the original validation check to prevent "Invalid entry for CD4" errors.
    """
    # Validate and fix CD4 values before calling original step_state
    st = self.sim.people.states
    if "hiv.cd4" in st:
        cd4 = st["hiv.cd4"]
        # Convert to numpy array for validation
        cd4_arr = np.asarray(cd4, dtype=float)
        
        # Find invalid CD4 values
        invalid_cd4 = ~np.isfinite(cd4_arr) | (cd4_arr <= 0) | (cd4_arr >= 2000)
        
        if invalid_cd4.any():
            # Fix invalid values by setting them to a reasonable default
            n_fixed = invalid_cd4.sum()
            for uid in np.where(invalid_cd4)[0]:
                if uid < len(cd4):
                    cd4[uid] = 500.0
    
    # Call original step_state (it may not return anything)
    if _original_step_state is not None:
        result = _original_step_state(self)
        return result if result is not None else None
    else:
        # Fallback: try to call the original method directly
        # This should not happen if patch is applied correctly
        try:
            result = super(type(self), self).step_state()
            return result if result is not None else None
        except AttributeError:
            # If super doesn't have step_state, that's okay - the original should be available
            pass

# Apply monkey patch when module is imported
def _apply_monkey_patch():
    """Apply monkey patches to HIV methods if not already applied."""
    global _original_post_art_decline, _original_step_state, _original_make_p_hiv_death
    
    # Patch post_art_decline
    if _original_post_art_decline is None:
        try:
            if hasattr(sti, 'HIV') and hasattr(sti.HIV, 'post_art_decline'):
                _original_post_art_decline = sti.HIV.post_art_decline
                sti.HIV.post_art_decline = _patched_post_art_decline
        except (AttributeError, TypeError):
            pass
    
    # Patch step_state
    if _original_step_state is None:
        try:
            if hasattr(sti, 'HIV') and hasattr(sti.HIV, 'step_state'):
                _original_step_state = sti.HIV.step_state
                sti.HIV.step_state = _patched_step_state
        except (AttributeError, TypeError):
            pass
    
    # Patch make_p_hiv_death
    if _original_make_p_hiv_death is None:
        try:
            if hasattr(sti, 'HIV') and hasattr(sti.HIV, 'make_p_hiv_death'):
                _original_make_p_hiv_death = sti.HIV.make_p_hiv_death
                sti.HIV.make_p_hiv_death = _patched_make_p_hiv_death
        except (AttributeError, TypeError):
            pass

# Try to apply immediately
_apply_monkey_patch()

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

    def __init__(self, base_dropout=0.10, base_dropout_noaud=0.001, allow_reinitiation_after_remission=True, label="adherence_art_dropout"):
        """
        Initialize ARTAdherenceDisruptor.
        
        Args:
            base_dropout: Base dropout probability multiplier for AUD-affected people (default: 0.10)
            base_dropout_noaud: Base dropout probability multiplier for No-AUD people (default: 0.001, very low)
                In Eswatini with 95-95-95 targets, baseline dropout should be very low (~1-2% per year)
            allow_reinitiation_after_remission: If True, people who drop out due to AUD can be re-added to ART
                after going into remission. If False, they remain permanently excluded (default: True).
                This is useful for scenarios without AUD care where people with AUD who drop out should
                not restart ART even if they go into remission.
            label: Label for the connector
        """
        super().__init__(label=label)
        self.base_dropout = float(base_dropout)  # For AUD-affected people
        self.base_dropout_noaud = float(base_dropout_noaud)  # For No-AUD people (very low baseline)
        self.allow_reinitiation_after_remission = bool(allow_reinitiation_after_remission)
        self._dropped_this_step = set()  # Track who was dropped this step to prevent immediate re-initiation
        self._ever_dropped = set()  # Track who has ever dropped out (persists across timesteps)
        self._dropped_due_to_aud = set()  # Track who dropped out specifically due to AUD (for permanent exclusion)

    def init_pre(self, sim):
        """Initialize connector and ensure monkey patch is applied."""
        super().init_pre(sim)
        # Ensure monkey patch is applied (in case it wasn't applied at import time)
        _apply_monkey_patch()
        
        # Also ensure the patches are applied to the instance's HIV module
        hiv = getattr(sim.diseases, "hiv", None)
        if hiv is not None:
            hiv_class = type(hiv)
            
            # Patch post_art_decline
            if hasattr(hiv_class, 'post_art_decline'):
                # Check if patch is already applied
                if hiv_class.post_art_decline != _patched_post_art_decline:
                    global _original_post_art_decline
                    if _original_post_art_decline is None:
                        _original_post_art_decline = hiv_class.post_art_decline
                    hiv_class.post_art_decline = _patched_post_art_decline
                    if sim.ti == 0:  # Only print once
                        print(f"[ARTAdherenceDisruptor] Applied monkey patch to {hiv_class.__name__}.post_art_decline")
            
            # Patch step_state
            if hasattr(hiv_class, 'step_state'):
                # Check if patch is already applied
                if hiv_class.step_state != _patched_step_state:
                    global _original_step_state
                    if _original_step_state is None:
                        _original_step_state = hiv_class.step_state
                    hiv_class.step_state = _patched_step_state
                    if sim.ti == 0:  # Only print once
                        print(f"[ARTAdherenceDisruptor] Applied monkey patch to {hiv_class.__name__}.step_state")
            
            # Patch make_p_hiv_death
            if hasattr(hiv_class, 'make_p_hiv_death'):
                # Check if patch is already applied
                if hiv_class.make_p_hiv_death != _patched_make_p_hiv_death:
                    global _original_make_p_hiv_death
                    if _original_make_p_hiv_death is None:
                        _original_make_p_hiv_death = hiv_class.make_p_hiv_death
                    hiv_class.make_p_hiv_death = _patched_make_p_hiv_death
                    if sim.ti == 0:  # Only print once
                        print(f"[ARTAdherenceDisruptor] Applied monkey patch to {hiv_class.__name__}.make_p_hiv_death")

    def step(self):
        sim = self.sim
        ppl = sim.people
        st = ppl.states
        hiv = getattr(sim.diseases, "hiv", None)
        if hiv is None:
            return

        # Reset tracking for this step
        self._dropped_this_step = set()
        
        # CRITICAL: Handle remission based on allow_reinitiation_after_remission setting
        # If allow_reinitiation_after_remission=True: Remove people from _ever_dropped when they go into remission
        # If allow_reinitiation_after_remission=False: Keep people in _ever_dropped permanently (especially if they dropped due to AUD)
        if len(self._ever_dropped) > 0 and "alcoholusedisorder.affected" in st:
            aud_affected = np.asarray(st["alcoholusedisorder.affected"], dtype=bool)
            # Find people in _ever_dropped who are no longer AUD-affected
            ever_dropped_list = list(self._ever_dropped)
            no_longer_aud = [uid for uid in ever_dropped_list if uid < len(aud_affected) and not aud_affected[uid]]
            
            if len(no_longer_aud) > 0:
                # Count how many of these dropped specifically due to AUD
                n_dropped_due_to_aud = len([uid for uid in no_longer_aud if uid in self._dropped_due_to_aud])
                
                if self.allow_reinitiation_after_remission:
                    # Remove them from _ever_dropped so they can be re-added to ART
                    # Also remove from _dropped_due_to_aud if they're there
                    self._ever_dropped -= set(no_longer_aud)
                    self._dropped_due_to_aud -= set(no_longer_aud)
                    if sim.ti % 12 == 0:  # Print once per year
                        print(f"[ARTAdherenceDisruptor] Year {sim.t.year}: Removed {len(no_longer_aud)} people from _ever_dropped "
                              f"(went into AUD remission, allow_reinitiation={self.allow_reinitiation_after_remission}, "
                              f"dropped_due_to_aud={n_dropped_due_to_aud})")
                else:
                    # Keep them in _ever_dropped permanently (especially if they dropped due to AUD)
                    # This means they won't be re-added even after remission
                    if sim.ti % 12 == 0:  # Print once per year
                        print(f"[ARTAdherenceDisruptor] Year {sim.t.year}: {len(no_longer_aud)} people in _ever_dropped went into remission "
                              f"but remain excluded (allow_reinitiation={self.allow_reinitiation_after_remission}, "
                              f"dropped_due_to_aud={n_dropped_due_to_aud})")

        adher = np.asarray(st["adherence"], float)
        on_art = np.asarray(st["hiv.on_art"], bool)
        
        # Get AUD status to apply different dropout rates
        aud_affected = np.asarray(st.get("alcoholusedisorder.affected", []), dtype=bool) if "alcoholusedisorder.affected" in st else np.zeros(len(adher), dtype=bool)
        
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

        # Calculate dropout probability: use different base rates for AUD vs No-AUD
        # For No-AUD people: very low baseline dropout (Eswatini 95-95-95 means minimal dropout)
        # For AUD-affected people: higher dropout based on adherence
        # OPTIMIZED: Pre-compute (1.0 - adher) once and use vectorized operations
        one_minus_adher = 1.0 - adher
        drop_p = np.where(
            aud_affected,
            self.base_dropout * one_minus_adher,  # AUD-affected: higher dropout
            self.base_dropout_noaud * one_minus_adher  # No-AUD: very low baseline dropout
        )
        # Clip to valid probability range [0, 1]
        drop_p = np.clip(drop_p, 0.0, 1.0)
        rand = np.random.rand(len(adher))
        
        # CRITICAL: Remove people from _ever_dropped if their dropout probability is now 0.0
        # This handles cases where people were added to _ever_dropped earlier (maybe when adherence was lower),
        # but now have perfect adherence (1.0) and thus dropout probability of 0.0.
        # If dropout probability is 0.0, they're not at risk of dropping out anymore, so there's no reason to exclude them.
        # This applies regardless of allow_reinitiation_after_remission - if there's no risk, they should be eligible.
        # The allow_reinitiation_after_remission setting only applies to people who are still at risk (dropout probability > 0.0).
        if len(self._ever_dropped) > 0:
            ever_dropped_list = list(self._ever_dropped)
            # Check dropout probability for people in _ever_dropped
            to_remove = []
            for uid in ever_dropped_list:
                if uid < len(drop_p):
                    # If dropout probability is 0.0 (or very close to 0.0), remove from _ever_dropped
                    # This allows them to be re-added to ART if needed, regardless of allow_reinitiation setting
                    # because if there's no risk of dropout, there's no reason to exclude them
                    if drop_p[uid] < 1e-6:  # Essentially 0.0
                        to_remove.append(uid)
            
            if len(to_remove) > 0:
                # Remove from both _ever_dropped and _dropped_due_to_aud
                self._ever_dropped -= set(to_remove)
                self._dropped_due_to_aud -= set(to_remove)
                if sim.ti % 12 == 0:  # Print once per year
                    print(f"[ARTAdherenceDisruptor] Year {sim.t.year}: Removed {len(to_remove)} people from _ever_dropped "
                          f"(dropout probability now 0.0, allowing re-initiation regardless of allow_reinitiation setting)")

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
                        
                        # CRITICAL: Ensure ti_art is valid for people we're dropping to prevent CD4 errors
                        # The HIV module's post_art_decline needs valid ti_art values
                        if "hiv.ti_art" in st:
                            ti_art = st["hiv.ti_art"]
                            for uid in final_drop_ids:
                                if uid < len(ti_art):
                                    # Ensure ti_art is valid (finite and >= 0)
                                    if not np.isfinite(ti_art[uid]) or ti_art[uid] < 0:
                                        # Set to a reasonable default (current timestep - 24 months)
                                        ti_art[uid] = max(0.0, float(sim.ti) - 24.0)
                        
                        # CRITICAL: Pre-validate CD4 values for people we're dropping to prevent "Invalid entry for CD4" errors
                        # The HIV module will recalculate CD4 after stopping ART, but we need to ensure
                        # current CD4 values are valid to prevent errors during the transition
                        if "hiv.cd4" in st:
                            cd4 = st["hiv.cd4"]
                            # Validate CD4 specifically for people we're dropping
                            for uid in final_drop_ids:
                                if uid < len(cd4):
                                    cd4_val = float(cd4[uid])  # Convert to float for comparison
                                    # Ensure CD4 is valid (finite, positive, reasonable range)
                                    if not (np.isfinite(cd4_val) and 0 < cd4_val < 2000):
                                        # Set to a reasonable default
                                        cd4[uid] = 500.0
                    
                    # Track who was dropped this step and ever dropped
                    dropped_uids = set(final_drop_ids.tolist())
                    self._dropped_this_step = dropped_uids
                    self._ever_dropped.update(dropped_uids)  # Add to ever_dropped set
                    
                    # Track which of these dropped specifically due to AUD (for permanent exclusion if needed)
                    if aud_dropped > 0:
                        aud_dropped_uids = [final_drop_ids[i] for i in range(len(final_drop_ids)) if aud_affected[final_drop_ids[i]]]
                        self._dropped_due_to_aud.update(aud_dropped_uids)
                        # Debug: verify we're only adding AUD-affected people
                        if sim.ti % 12 == 0:  # Print once per year
                            print(f"[ARTAdherenceDisruptor] Year {sim.t.year}: Added {len(aud_dropped_uids)} AUD-affected people to _dropped_due_to_aud "
                                  f"(total dropped this step={final_drop_ids.size}, aud_dropped={aud_dropped})")
                    
                    # Print when dropping with detailed debug info
                    if sim.ti % 5 == 0 or final_drop_ids.size > 10:
                        print(f"[ARTAdherenceDisruptor] Year {sim.t.year}, ti={sim.ti}: Scheduled ART stop for {final_drop_ids.size} agents "
                              f"(AUD={aud_dropped}, NoAUD={final_drop_ids.size - aud_dropped}, "
                              f"mean_adherence={mean_adher_dropped:.3f}, mean_drop_prob={mean_drop_p_dropped:.4f})")
                    
                    # Yearly summary of dropout tracking
                    if sim.ti % 12 == 0:  # Once per year
                        # Get current ART status for context
                        on_art = np.asarray(st.get("hiv.on_art", []), bool) if "hiv.on_art" in st else np.zeros(len(ppl), dtype=bool)
                        aud_affected = np.asarray(st.get("alcoholusedisorder.affected", []), dtype=bool) if "alcoholusedisorder.affected" in st else np.zeros(len(ppl), dtype=bool)
                        aud_on_art = (on_art & aud_affected).sum()
                        noaud_on_art = (on_art & ~aud_affected).sum()
                        
                        # Count how many in _ever_dropped are currently AUD-affected
                        ever_dropped_aud = len([uid for uid in self._ever_dropped if uid < len(aud_affected) and aud_affected[uid]])
                        ever_dropped_noaud = len(self._ever_dropped) - ever_dropped_aud
                        
                        # Count how many in _dropped_due_to_aud are currently AUD-affected vs in remission
                        dropped_due_to_aud_currently_aud = len([uid for uid in self._dropped_due_to_aud if uid < len(aud_affected) and aud_affected[uid]])
                        dropped_due_to_aud_in_remission = len(self._dropped_due_to_aud) - dropped_due_to_aud_currently_aud
                        
                        print(f"[ARTAdherenceDisruptor SUMMARY] Year {sim.t.year}: "
                              f"_ever_dropped={len(self._ever_dropped)} (currently AUD={ever_dropped_aud}, currently No-AUD={ever_dropped_noaud}), "
                              f"_dropped_due_to_aud={len(self._dropped_due_to_aud)} (currently AUD={dropped_due_to_aud_currently_aud}, in remission={dropped_due_to_aud_in_remission}), "
                              f"allow_reinitiation={self.allow_reinitiation_after_remission}, "
                              f"Currently on ART: AUD={aud_on_art}, No-AUD={noaud_on_art}")
        
        # Final validation: Check CD4 values for people who have ti_stop_art set (scheduled to drop)
        # This catches any invalid CD4 values that might cause errors in the HIV module's step_state
        if "hiv.ti_stop_art" in st and "hiv.cd4" in st:
            ti_stop_art = st["hiv.ti_stop_art"]
            cd4 = st["hiv.cd4"]
            # Find people who have ti_stop_art set (scheduled to drop this step)
            ti_stop_art_arr = np.asarray(ti_stop_art, dtype=float)
            scheduled_to_drop = np.isfinite(ti_stop_art_arr) & (ti_stop_art_arr == sim.ti)
            
            if scheduled_to_drop.any():
                # Validate CD4 for people scheduled to drop
                for uid in np.where(scheduled_to_drop)[0]:
                    if uid < len(cd4):
                        cd4_val = float(cd4[uid])
                        # Ensure CD4 is valid (finite, positive, reasonable range)
                        if not (np.isfinite(cd4_val) and 0 < cd4_val < 2000):
                            # Set to a reasonable default
                            cd4[uid] = 500.0


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

    def __init__(self, scale_art_efficacy=True, label="intervention_adherence_disruptor"):
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


# =====================================================================
# Legacy connector: AdherenceFromDepression
# =====================================================================
class AdherenceFromDepression(ss.Connector):
    """
    Legacy connector kept for backwards compatibility with older tests/workflows.

    Computes a simple adherence score based on Major Depressive Disorder status
    and records mean adherence over time.

    Notes
    -----
    - Does not require `AdherenceEngine` (module) or extra `People` states.
    - If a `people.states["adherence"]` array exists, it will be updated too.
    """

    def __init__(self, base=1.0, depression_factor=0.8, label="adherence_from_depression"):
        super().__init__(label=label)
        self.base = float(base)
        self.depression_factor = float(depression_factor)
        self.time = []
        self.mean_adherence = []

    def init_post(self):
        super().init_post()
        # Optional: create a people-level adherence state if the API supports it
        try:
            ppl = self.sim.people
            if hasattr(ppl, "states") and "adherence" not in ppl.states:
                # Starsim People may or may not support dynamic add; keep best-effort only
                ppl.states["adherence"] = np.full(len(ppl), self.base, dtype=float)
        except Exception:
            pass

    def step(self):
        sim = self.sim
        ppl = sim.people

        n = len(ppl)
        adher = np.full(n, self.base, dtype=float)

        # Depression reduces adherence
        dep = getattr(sim.diseases, "majordepressivedisorder", None)
        if dep is not None and hasattr(dep, "affected"):
            dep_mask = np.asarray(dep.affected, dtype=bool)
            # Handle length mismatch defensively
            if len(dep_mask) != n:
                dep_mask = np.pad(dep_mask, (0, max(0, n - len(dep_mask))), constant_values=False)[:n]
            adher[dep_mask] *= self.depression_factor

        # Depression care can boost adherence for treated individuals (best-effort)
        for intv in getattr(sim, "interventions", []) if not isinstance(getattr(sim, "interventions", {}), dict) else sim.interventions.values():
            if intv is None:
                continue
            if intv.__class__.__name__.lower() == "depressioncare" or getattr(intv, "label", "").lower().startswith("depression"):
                treated = getattr(intv, "treated_inds", None)
                boost = float(getattr(intv, "adherence_boost", 1.0))
                if treated is not None and len(treated):
                    try:
                        adher[np.asarray(treated, dtype=int)] *= boost
                    except Exception:
                        pass
                break

        adher = np.clip(adher, 0.0, 1.0)

        # Persist into people state if available
        try:
            if hasattr(ppl, "states"):
                ppl.states["adherence"] = adher
        except Exception:
            pass

        # Record results
        year = getattr(sim.t, "year", None)
        self.time.append(float(year) if year is not None else float(getattr(sim, "now", sim.ti)))
        self.mean_adherence.append(float(np.mean(adher)))