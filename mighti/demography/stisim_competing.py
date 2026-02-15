"""
Compatibility wrappers for STIsim modules under MIGHTI competing-risks mortality.

STIsim's `HIV` module directly calls `people.request_death()` inside `step_state()`,
which will double-count deaths if the sim also uses an all-cause mx table.

This wrapper provides a drop-in replacement that:
- computes and reports HIV "death pressure" each timestep, but
- does NOT directly request death when competing mortality mode is enabled.

Deaths are then allocated by `mighti.mortality_competing.CompetingRisksDeaths`.
"""

from __future__ import annotations

import numpy as np
import starsim as ss
import stisim as sti

__all__ = ["HIVCompeting"]


class HIVCompeting(sti.HIV):
    """
    STIsim HIV module adapted for competing-risks death allocation.

    In competing mode (when `sim._mighti_competing_mortality` is True), this module:
    - reports per-agent death pressure via `get_death_pressure()`
    - defers actual death scheduling to `CompetingRisksDeaths`
    - sets `ti_dead` only for deaths attributed to HIV (via sim cause map) during `step_die`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Critical: many MIGHTI connectors/modules expect the HIV module to be named "hiv"
        # (e.g., `sim.diseases.hiv`, `sim.people.hiv`). Ensure we preserve that key.
        self.name = "hiv"
        try:
            self.t.name = "hiv"
        except Exception:
            pass
        self._death_pressure_uids = np.array([], dtype=int)
        self._death_pressure_p = np.array([], dtype=float)
        return

    def _competing_enabled(self) -> bool:
        return bool(getattr(self.sim, "_mighti_competing_mortality", False))

    def _set_death_pressure(self, uids, p) -> None:
        self._death_pressure_uids = np.asarray(uids, dtype=int)
        self._death_pressure_p = np.asarray(p, dtype=float)

    def get_death_pressure(self):
        return self._death_pressure_uids, self._death_pressure_p

    def _attributed_deaths(self, death_uids: np.ndarray) -> np.ndarray:
        cause_map = getattr(self.sim, "_mighti_death_cause", None)
        if not isinstance(cause_map, dict) or not len(death_uids):
            return np.array([], dtype=int)
        my_name = getattr(self, "name", self.__class__.__name__)
        keep = [uid for uid in death_uids if cause_map.get(int(uid)) == my_name]
        return np.asarray(keep, dtype=int)

    def step_state(self):
        """
        Copy of upstream HIV.step_state(), but with mortality handled differently under
        competing mode (report pressure instead of requesting deaths).
        """
        ti = self.ti

        # Clear pressure each step (avoid stale carryover)
        if self._competing_enabled():
            self._set_death_pressure(np.array([], dtype=int), np.array([], dtype=float))

        # Set initial CD4 counts for new agents:
        self.init_cd4()

        # Handle care seeking behavior, including pregnancy scaling
        self.init_care_seeking()
        if self.include_mtct:
            pregnant = self.sim.demographics.pregnancy.pregnant
            self.care_seeking[pregnant] = self.baseline_care_seeking[pregnant] * self.pars.maternal_care_scale
            self.care_seeking[~pregnant] = self.baseline_care_seeking[~pregnant]

        # Adjust CD4 counts for people receiving treatment - logarithmic increase
        if self.on_art.any():
            art_uids = self.on_art.uids
            self.cd4[art_uids] = self.cd4_increase(art_uids)

        # Adjust CD4 counts for people who have gone off treatment - linear decline
        if (~self.on_art & ~self.never_art).any():
            off_art_uids = (~self.on_art & ~self.never_art).uids
            self.cd4[off_art_uids] = self.post_art_decline(off_art_uids)

        # Update states for people who have never been on ART (ART removes these)
        latent = self.acute & (self.ti_latent <= ti)
        falling = self.latent & (self.ti_falling <= ti)
        self.acute[latent] = False
        self.latent[latent] = True
        self.latent[falling] = False
        self.falling[falling] = True

        # Update CD4 counts
        self.cd4[self.acute.uids] = self.acute_decline(self.acute.uids)
        untreated_latent = self.latent
        self.cd4[untreated_latent.uids] = self.cd4_latent[untreated_latent.uids]
        untreated_falling = self.falling
        if untreated_falling.any():
            self.cd4[untreated_falling.uids] = self.falling_decline(untreated_falling.uids)

        # Update CD4 nadir for anyone not on treatment
        untreated = self.infected & ~self.on_art
        self.cd4_nadir[untreated] = np.minimum(self.cd4_nadir[untreated], self.cd4[untreated])

        # Update transmission modifiers
        self.update_transmission()

        # Check CD4
        if np.isnan(self.cd4[self.infected]).any():
            raise ValueError("Invalid entry for CD4")

        # Mortality logic
        off_art = (self.infected & ~self.on_art).uids
        if len(off_art):
            p_death = self.make_p_hiv_death(uids=off_art)
        else:
            p_death = np.array([], dtype=float)

        if self._competing_enabled():
            # Report death pressure instead of requesting death
            uids = off_art.copy()
            probs = np.asarray(p_death, dtype=float).copy()

            if self.pars.include_aids_deaths:
                aids_uids = (self.ti_zero <= ti).uids
                if len(aids_uids):
                    # Treat as "certain" HIV death pressure this timestep; allocator will cap at all-cause
                    uids = np.concatenate([uids, aids_uids]).astype(int, copy=False)
                    probs = np.concatenate([probs, np.ones(len(aids_uids), dtype=float)])

            if len(uids):
                # If duplicates exist, keep max pressure per uid
                order = np.argsort(uids)
                u_sorted = uids[order]
                p_sorted = probs[order]
                uniq, idx_start = np.unique(u_sorted, return_index=True)
                p_max = np.zeros(len(uniq), dtype=float)
                for i, uid in enumerate(uniq):
                    start = idx_start[i]
                    end = idx_start[i + 1] if i + 1 < len(idx_start) else len(u_sorted)
                    p_max[i] = float(np.max(p_sorted[start:end]))
                self._set_death_pressure(uniq, np.clip(p_max, 0.0, 1.0))
            return

        # Legacy: upstream behavior (request deaths directly)
        self.pars.p_hiv_death.set(0)
        self.pars.p_hiv_death.set(p_death)
        hiv_deaths = self.pars.p_hiv_death.filter(off_art)
        if len(hiv_deaths):
            self.ti_dead[hiv_deaths] = ti
            self.sim.people.request_death(hiv_deaths)
        if self.pars.include_aids_deaths:
            aids_deaths = (self.ti_zero <= ti).uids
            if len(aids_deaths):
                self.ti_dead[aids_deaths] = ti
                self.sim.people.request_death(aids_deaths)
        return

    def step_die(self, uids):
        # In competing mode, set ti_dead only for attributed HIV deaths
        if self._competing_enabled():
            uids = np.asarray(uids, dtype=int)
            attributed = self._attributed_deaths(uids)
            if len(attributed):
                self.ti_dead[attributed] = self.ti

        # Always clear states for anyone who died (regardless of cause)
        return super().step_die(uids)

