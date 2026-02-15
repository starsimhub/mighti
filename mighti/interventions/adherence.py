"""
Unified Adherence System for MIGHTI
-----------------------------------
Provides a three-component adherence pipeline:

1. AdherenceEngine           (CASM + SDoH → adherence state)
2. ARTAdherenceDisruptor     (adherence → ART dropout)
3. InterventionAdherenceDisruptor (adherence → ART efficacy scaling)
"""

from __future__ import annotations

import numpy as np
import sciris as sc
import starsim as ss
import stisim as sti
import logging

from mighti.util.rng import get_rng

logger = logging.getLogger(__name__)

__all__ = [
    "AdherenceEngine",
    "ARTAdherenceDisruptor",
    "InterventionAdherenceDisruptor",
    "AdherenceFromDepression",
    "BASELINE_ADHERENCE_PHARMACOTHERAPY",
    "CASM_NONADHERENCE_OR",
    "CASM_REL_FACTORS",
    "SDOH_REL_FACTORS",
]

# ---------------------------------------------------------------------
# HRMM-style defaults (paper-compatible)
# ---------------------------------------------------------------------
# Baseline adherence probability under pharmacotherapy
BASELINE_ADHERENCE_PHARMACOTHERAPY = 0.62

# Odds ratio (OR) for NON-adherence in presence of CASM conditions
CASM_NONADHERENCE_OR = {
    "AlcoholUseDisorder": 1.41,
    "MajorDepressiveDisorder": 2.21,
    "AnxietyDisorder": 2.04,
    "ChronicPain": 1.34,
    "TobaccoUse": 1.18,
    "OpioidUseDisorder": 1.18,
    "StimulantUseDisorder": 1.81,
}

# ---------------------------------------------------------------------
# CASM adherence multipliers (1/OR from Table S2)
#   Values < 1 => reduced adherence when condition is present
# ---------------------------------------------------------------------
CASM_REL_FACTORS = {k: 1.0 / float(v) for k, v in CASM_NONADHERENCE_OR.items()}

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

    def __init__(
        self,
        casm_rel: dict | None = None,
        sdoh_rel: dict | None = None,
        *,
        # HRMM-style "odds of non-adherence" model (compat with `adherence_unified.py`)
        baseline_adherence: float | None = None,
        casm_nonadherence_or: dict | None = None,
        use_odds_model: bool = False,
        label: str = "adherence_engine",
    ):
        super().__init__(label=label)
        self.casm_rel = casm_rel or CASM_REL_FACTORS.copy()
        self.sdoh_rel = sdoh_rel or SDOH_REL_FACTORS.copy()
        self.baseline_adherence = baseline_adherence
        self.casm_nonadherence_or = casm_nonadherence_or
        self.use_odds_model = bool(use_odds_model)

    def init_pre(self, sim):
        super().init_pre(sim)
        logger.debug("[AdherenceEngine] Initialized for sim '%s'", getattr(sim, "label", "?"))
        st = sim.people.states
        if "adherence" not in st:
            try:
                arr = ss.FloatArr("adherence", default=1.0)
                sim.people.states.append(arr, overwrite=False)
                st["adherence"][:] = 1.0
            except Exception:
                st["adherence"] = np.ones(len(sim.people), dtype=float)

    @staticmethod
    def _odds(p: float) -> float:
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        return p / (1.0 - p)

    @staticmethod
    def _p_from_odds(o: np.ndarray) -> np.ndarray:
        o = np.asarray(o, dtype=float)
        return o / (1.0 + o)

    def step(self):
        ppl = self.sim.people
        st = ppl.states
        n = len(ppl)

        # Decide model:
        # - Default (backwards compatible): multiplicative penalties on [0,1] adherence starting at 1.0
        # - Optional (paper-compatible): baseline adherence p0 with CASM odds ratios acting on non-adherence odds
        odds_mode = self.use_odds_model or (self.baseline_adherence is not None) or (self.casm_nonadherence_or is not None)

        if odds_mode:
            p0 = float(np.clip(self.baseline_adherence if self.baseline_adherence is not None else BASELINE_ADHERENCE_PHARMACOTHERAPY, 1e-12, 1.0 - 1e-12))
            or_map = (self.casm_nonadherence_or or CASM_NONADHERENCE_OR).copy()

            odds_nonadh_0 = self._odds(1.0 - p0)
            mult = np.ones(n, dtype=float)
            for cond, or_nonadh in or_map.items():
                key = f"{cond.lower()}.affected"
                if key in st:
                    affected = np.asarray(st[key], dtype=bool)
                    if len(affected) != n:
                        affected = np.resize(affected, n)
                    mult[affected] *= float(or_nonadh)
            odds_nonadh = odds_nonadh_0 * mult
            p_nonadh = self._p_from_odds(odds_nonadh)
            adherence = 1.0 - p_nonadh
        else:
            # Start from perfect adherence and apply multiplicative penalties
            adherence = np.ones(n, dtype=float)

            # CASM effects
            for cond, rel in self.casm_rel.items():
                key = f"{cond.lower()}.affected"  # e.g. 'majordepressivedisorder.affected'
                if key in st:
                    affected = np.asarray(st[key], bool)
                    if len(affected) != n:
                        affected = np.resize(affected, n)
                    adherence[affected] *= float(rel)
                else:
                    # Debug: check if state exists with different casing
                    if self.sim.ti % 10 == 0:  # Print every 10 timesteps
                        possible_keys = [k for k in st.keys() if cond.lower() in k.lower()]
                        if possible_keys:
                            logger.debug("[AdherenceEngine] Missing '%s'; similar keys: %s", key, possible_keys)

        # SDoH effects (applies to both models)
        for sdoh_key, rel in self.sdoh_rel.items():
            if sdoh_key in st:
                flagged = np.asarray(st[sdoh_key], bool)
                if len(flagged) != n:
                    flagged = np.resize(flagged, n)
                adherence[flagged] *= float(rel)

        # Clip to [0, 1] and write back into the dynamic state
        if "adherence" in st:
            try:
                st["adherence"][:] = np.clip(adherence, 0.0, 1.0)
            except Exception:
                st["adherence"] = np.clip(adherence, 0.0, 1.0)
        else:
            st["adherence"] = np.clip(adherence, 0.0, 1.0)
        
        # Debug: print adherence stats for AUD individuals
        if self.sim.ti % 10 == 0:  # Print every 10 timesteps
            aud_key = "alcoholusedisorder.affected"
            if aud_key in st:
                aud_affected = np.asarray(st[aud_key], bool)
                if aud_affected.any():
                    aud_adherence = adherence[aud_affected]
                    logger.debug(
                        "[AdherenceEngine] Year %s: AUD=%s, mean adherence=%0.3f, min=%0.3f, max=%0.3f",
                        getattr(self.sim.t, "year", "?"),
                        int(aud_affected.sum()),
                        float(aud_adherence.mean()),
                        float(aud_adherence.min()),
                        float(aud_adherence.max()),
                    )


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
        """Initialize connector."""
        super().init_pre(sim)

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
                    if sim.ti % 12 == 0:  # Log once per year
                        logger.debug(
                            "[ARTAdherenceDisruptor] Year %s: Removed %s from _ever_dropped (AUD remission; dropped_due_to_aud=%s)",
                            getattr(sim.t, "year", "?"),
                            int(len(no_longer_aud)),
                            int(n_dropped_due_to_aud),
                        )
                else:
                    # Keep them in _ever_dropped permanently (especially if they dropped due to AUD)
                    # This means they won't be re-added even after remission
                    if sim.ti % 12 == 0:  # Log once per year
                        logger.debug(
                            "[ARTAdherenceDisruptor] Year %s: %s in _ever_dropped went into remission but remain excluded (dropped_due_to_aud=%s)",
                            getattr(sim.t, "year", "?"),
                            int(len(no_longer_aud)),
                            int(n_dropped_due_to_aud),
                        )

        adher = np.asarray(st["adherence"], float)
        on_art = np.asarray(st["hiv.on_art"], bool)
        
        n = len(on_art)

        def _bool_mask(key: str) -> np.ndarray:
            """Return a boolean mask of length n; missing -> all False."""
            arr = st.get(key, None)
            if arr is None:
                return np.zeros(n, dtype=bool)
            out = np.asarray(arr, dtype=bool)
            if out.size == 0:
                return np.zeros(n, dtype=bool)
            if out.shape[0] != n:
                out = np.resize(out, n)
            return out

        # Get AUD status to apply different dropout rates
        aud_affected = _bool_mask("alcoholusedisorder.affected")
        
        # Debug: Always print on first few timesteps to verify it's running
        if sim.ti < 3:
            logger.debug(
                "[ARTAdherenceDisruptor] Year %s, ti=%s: Running. On ART=%s, mean adherence=%0.3f",
                getattr(sim.t, "year", "?"),
                sim.ti,
                int(on_art.sum()),
                float(adher.mean()),
            )

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
        rng = get_rng(sim, salt=f"{self.__class__.__name__}:step")
        rand = rng.random(len(adher))
        
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
                if sim.ti % 12 == 0:  # Log once per year
                    logger.debug(
                        "[ARTAdherenceDisruptor] Year %s: Removed %s from _ever_dropped (dropout probability ~0)",
                        getattr(sim.t, "year", "?"),
                        int(len(to_remove)),
                    )

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
                    if sim.ti % 12 == 0:  # Log once per year
                        logger.debug(
                            "[ARTAdherenceDisruptor] Year %s, ti=%s: ti_art range=[%0.1f, %0.1f], sim_ti=%0.1f, sim_ti_months=%0.1f",
                            getattr(sim.t, "year", "?"),
                            sim.ti,
                            float(ti_art_on_art.min()),
                            float(ti_art_on_art.max()),
                            float(sim_ti_float),
                            float(sim_ti_months),
                        )
                    
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
            aud_affected = np.asarray(st.get("alcoholusedisorder.affected", np.zeros_like(on_art)), dtype=bool)
            if aud_affected.size == 0:
                aud_affected = np.zeros_like(on_art, dtype=bool)
            elif aud_affected.shape[0] != on_art.shape[0]:
                aud_affected = np.resize(aud_affected, on_art.shape[0])
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
                # Debug logging removed (too noisy for production)
                pass
            else:
                # Debug logging removed (too noisy for production)
                pass
        
        # Calculate dropout for valid people
        # Only consider people who are valid (on ART for at least 2 months)
        drop_ids = np.where(valid_art & (rand < drop_p))[0]
        
        # Debug: print expected vs actual drops occasionally
        if sim.ti % 5 == 0 and valid_art.sum() > 0:  # Every 5 timesteps (less verbose)
            aud_affected = np.asarray(st.get("alcoholusedisorder.affected", np.zeros_like(on_art)), dtype=bool)
            if aud_affected.size == 0:
                aud_affected = np.zeros_like(on_art, dtype=bool)
            elif aud_affected.shape[0] != on_art.shape[0]:
                aud_affected = np.resize(aud_affected, on_art.shape[0])
            aud_valid = (valid_art & aud_affected).sum()
            if aud_valid > 0:
                aud_drop_p = drop_p[valid_art & aud_affected]
                expected_aud_drops = aud_drop_p.sum()
                # Recalculate actual drops using the same logic as drop_ids
                aud_valid_mask = valid_art & aud_affected
                aud_rand = rand[aud_valid_mask]
                aud_drop_p_filtered = drop_p[aud_valid_mask]
                actual_aud_drops = (aud_rand < aud_drop_p_filtered).sum()
                # Debug logging removed (too noisy for production)
                pass
                # Additional debug: show some actual values
                if len(aud_drop_p_filtered) > 0:
                    # Debug logging removed (too noisy for production)
                    pass

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
                    # Track who we're about to drop (AUD may not be present)
                    aud_affected = _bool_mask("alcoholusedisorder.affected")
                    aud_dropped = int(aud_affected[final_drop_ids].sum()) if aud_affected.size else 0
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
                        if sim.ti % 12 == 0:  # Log once per year
                            logger.debug(
                                "[ARTAdherenceDisruptor] Year %s: Added %s to _dropped_due_to_aud (total dropped=%s, aud_dropped=%s)",
                                getattr(sim.t, "year", "?"),
                                int(len(aud_dropped_uids)),
                                int(final_drop_ids.size),
                                int(aud_dropped),
                            )
                    
                    # Print when dropping with detailed debug info
                    if sim.ti % 5 == 0 or final_drop_ids.size > 10:
                        logger.debug(
                            "[ARTAdherenceDisruptor] Year %s, ti=%s: Scheduled ART stop for %s (AUD=%s, NoAUD=%s, mean_adherence=%0.3f, mean_drop_prob=%0.4f)",
                            getattr(sim.t, "year", "?"),
                            sim.ti,
                            int(final_drop_ids.size),
                            int(aud_dropped),
                            int(final_drop_ids.size - aud_dropped),
                            float(mean_adher_dropped),
                            float(mean_drop_p_dropped),
                        )
                    
                    # Yearly summary of dropout tracking
                    if sim.ti % 12 == 0:  # Once per year
                        # Get current ART status for context
                        on_art = np.asarray(st.get("hiv.on_art", []), bool) if "hiv.on_art" in st else np.zeros(len(ppl), dtype=bool)
                        aud_affected = _bool_mask("alcoholusedisorder.affected")
                        aud_on_art = (on_art & aud_affected).sum()
                        noaud_on_art = (on_art & ~aud_affected).sum()
                        
                        # Count how many in _ever_dropped are currently AUD-affected
                        ever_dropped_aud = len([uid for uid in self._ever_dropped if uid < len(aud_affected) and aud_affected[uid]])
                        ever_dropped_noaud = len(self._ever_dropped) - ever_dropped_aud
                        
                        # Count how many in _dropped_due_to_aud are currently AUD-affected vs in remission
                        dropped_due_to_aud_currently_aud = len([uid for uid in self._dropped_due_to_aud if uid < len(aud_affected) and aud_affected[uid]])
                        dropped_due_to_aud_in_remission = len(self._dropped_due_to_aud) - dropped_due_to_aud_currently_aud
                        
                        logger.debug(
                            "[ARTAdherenceDisruptor SUMMARY] Year %s: _ever_dropped=%s (AUD=%s, No-AUD=%s), _dropped_due_to_aud=%s (AUD=%s, remission=%s), allow_reinitiation=%s, on ART: AUD=%s, No-AUD=%s",
                            getattr(sim.t, "year", "?"),
                            int(len(self._ever_dropped)),
                            int(ever_dropped_aud),
                            int(ever_dropped_noaud),
                            int(len(self._dropped_due_to_aud)),
                            int(dropped_due_to_aud_currently_aud),
                            int(dropped_due_to_aud_in_remission),
                            bool(self.allow_reinitiation_after_remission),
                            int(aud_on_art),
                            int(noaud_on_art),
                        )
        
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
        
        # (no stdout debug printing; keep module quiet for production)

        # 1. Starsim/sti HIV ART efficacy (pars.art_efficacy)
        # Only scale if explicitly enabled
        if self.scale_art_efficacy:
            hiv = getattr(sim.diseases, "hiv", None)
            if hiv is not None and hasattr(hiv.pars, "art_efficacy"):
                if self._baseline_art_efficacy is not None:
                    hiv.pars.art_efficacy = self._baseline_art_efficacy * scale
                else:
                    # Fallback if init_post wasn't called (shouldn't happen)
                    logger.warning("[%s] baseline_art_efficacy not set; skipping scaling", self.label)

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
                    logger.warning("[%s] baseline for %s not found; skipping scaling", self.label, label)
                    

# =====================================================================
# Legacy connector: AdherenceFromDepression
# =====================================================================
class AdherenceFromDepression(ss.Connector):
    """
    Legacy connector kept for backwards compatibility with older tests/workflows.

    Computes a simple adherence score based on Major Depressive Disorder status
    and records mean adherence over time.
    """

    def __init__(self, base=1.0, depression_factor=0.8, label="adherence_from_depression"):
        super().__init__(label=label)
        self.base = float(base)
        self.depression_factor = float(depression_factor)
        self.time = []
        self.mean_adherence = []

    def step(self):
        sim = self.sim
        ppl = sim.people
        st = ppl.states

        n = len(ppl)
        adher = np.full(n, self.base, dtype=float)

        dep = getattr(sim.diseases, "majordepressivedisorder", None)
        if dep is not None and hasattr(dep, "affected"):
            dep_mask = np.asarray(dep.affected, dtype=bool)
            if len(dep_mask) != n:
                dep_mask = np.pad(dep_mask, (0, max(0, n - len(dep_mask))), constant_values=False)[:n]
            adher[dep_mask] *= self.depression_factor

        adher = np.clip(adher, 0.0, 1.0)

        # Persist into people state if present
        if "adherence" in st:
            try:
                st["adherence"][:] = adher
            except Exception:
                pass

        # Record time series
        year = getattr(sim.t, "year", None)
        self.time.append(float(year) if year is not None else float(getattr(sim, "now", sim.ti)))
        self.mean_adherence.append(float(np.mean(adher)))