"""
Competing-risks mortality (all-cause capped with residual bucket).

Why this exists
---------------
In StarSim, *any* module can call ``people.request_death(uids)`` and the People
object will apply the union of all requests on that timestep. If you also run
``ss.Deaths`` using an *all-cause* mortality table (e.g., UN/WPP mx), and your
disease modules also request deaths, you will double-count mortality.

This module provides an alternative pattern:

- Use an observed all-cause mortality table to set the *total* probability of
  death per agent per timestep.
- Ask modeled disease modules to report their "death pressure" (per-agent
  probability weights for this timestep), without directly requesting death.
- Allocate deaths across modeled causes + a residual ("unmodeled") bucket,
  without exceeding all-cause mortality.

This is a pragmatic allocator (not a full causal multi-cause model). It is
designed to prevent death double counting while producing explicit cause
attribution suitable for analyzers.
"""

from typing import Mapping

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

from mighti.util.rng import get_rng

__all__ = ["CompetingRisksDeaths"]


class _DeathPressure:
    def __init__(self, name, uids, p):
        self.name = name
        self.uids = uids  # int
        self.p = p  # float, per-timestep probability-like weight


class CompetingRisksDeaths(ss.Disease):
    """
    Allocate deaths using all-cause mx with a residual bucket.

    Parameters match (a subset of) ``starsim.demographics.Deaths``:

    Args:
        death_rate: number/rate/dataframe/series (typically UN/WPP mx by year/sex/age)
        rel_death: scalar multiplier
        rate_units: units scaling applied to input death rates (e.g., 1 if already per person-year)
        metadata: optional column mapping + sex key mapping for the dataframe format
        residual_cause_name: label used for residual/unmodeled deaths
    """

    def __init__(
        self,
        death_rate,
        *,
        rel_death=1.0,
        rate_units=1e-3,
        metadata=None,
        residual_cause_name="residual",
        **kwargs,
    ):
        super().__init__()
        self.define_pars(
            rel_death=rel_death,
            death_rate=death_rate,
            rate_units=rate_units,
            residual_cause_name=residual_cause_name,
        )
        self.update_pars(kwargs)

        # Defaults align with StarSim UN/WPP format
        self.metadata = sc.mergedicts(
            sc.objdict(
                data_cols=dict(year="Time", sex="Sex", age="AgeGrpStart", value="mx"),
                # Accept both UN/WPP-style labels and Starsim internal labels
                sex_keys={"Female": "f", "Male": "m", "f": "f", "m": "m"},
            ),
            metadata,
        )

        self.death_rate_data = None
        self._cause_map = {}  # uid -> cause name, for *current timestep only*
        return

    # ------------------------------------------------------------------
    # StarSim lifecycle
    # ------------------------------------------------------------------
    def init_pre(self, sim):
        super().init_pre(sim)
        # Enable "competing mortality mode" for MIGHTI modules
        setattr(sim, "_mighti_competing_mortality", True)

        # Standardize death data similarly to ss.Deaths
        drd = ss.standardize_data(data=self.pars.death_rate, metadata=self.metadata)
        if isinstance(drd, (pd.Series, pd.DataFrame)):
            drd = drd.unstack(level="age")
            assert not drd.isna().any(axis=None)
        if sc.isnumber(drd):
            # If user provided a bare number, assume per year
            ss.warn(
                f"CompetingRisksDeaths.death_rate specified as number ({drd}); assuming per year"
            )
            drd = ss.peryear(drd)
        self.death_rate_data = drd
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result(
                "new_deaths",
                dtype=int,
                scale=True,
                summarize_by="sum",
                label="All-cause deaths (competing risks)",
                auto_plot=False,
            ),
            ss.Result(
                "residual_share_mean",
                dtype=float,
                scale=False,
                summarize_by="mean",
                label="Mean residual share of death risk",
                auto_plot=False,
            ),
            ss.Result(
                "modeled_share_mean",
                dtype=float,
                scale=False,
                summarize_by="mean",
                label="Mean modeled share of death risk",
                auto_plot=False,
            ),
        )
        return

    # ------------------------------------------------------------------
    # Core mechanics
    # ------------------------------------------------------------------
    def _make_p_allcause(self, uids):
        """Probability of death (all-cause) for these uids on this timestep."""
        sim = self.sim
        drd = self.death_rate_data
        p = self.pars

        if sc.isnumber(drd):
            death_rate = np.array([drd * p.rate_units * p.rel_death])

        elif isinstance(drd, ss.Rate):
            if drd.unit == 1:
                death_rate = np.array([drd.value * p.rate_units * p.rel_death])
            else:
                death_rate = np.array(
                    [ss.prob(drd.value * p.rate_units * p.rel_death, drd.unit).to_prob(ss.years(1))]
                )

        else:
            ppl = sim.people
            available_years = drd.index.get_level_values("year")
            year_ind = sc.findnearest(available_years, sim.t.now("year"))
            nearest_year = available_years[year_ind]

            death_rate = np.empty(uids.shape, dtype=ss.dtypes.float)

            if "sex" in drd.index.names:
                s = drd.loc[nearest_year, "f"]
                female_mask = np.asarray(ppl.female[uids], dtype=bool)
                female_uids = uids[female_mask]
                if len(female_uids):
                    binned_ages = np.digitize(ppl.age[female_uids], s.index) - 1
                    death_rate[female_mask] = s.values[binned_ages]

                s = drd.loc[nearest_year, "m"]
                male_mask = np.asarray(ppl.male[uids], dtype=bool)
                male_uids = uids[male_mask]
                if len(male_uids):
                    binned_ages = np.digitize(ppl.age[male_uids], s.index) - 1
                    death_rate[male_mask] = s.values[binned_ages]
            else:
                s = drd.loc[nearest_year]
                binned_ages = np.digitize(ppl.age[uids], s.index) - 1
                death_rate[:] = s.values[binned_ages]

            death_rate *= p.rate_units * p.rel_death

        death_rate = ss.peryear(death_rate)
        p_all = death_rate.to_prob(self.t.dt)

        if sc.isnumber(drd) or isinstance(drd, ss.Rate):
            p_all = np.full(uids.shape, float(p_all[0]), dtype=float)

        p_all = np.clip(p_all, 0.0, 1.0)
        return p_all.astype(float)

    def _collect_death_pressures(self):
        pressures = []
        for mod in self.sim.diseases():
            if mod is self:
                continue
            getp = getattr(mod, "get_death_pressure", None)
            if getp is None or not callable(getp):
                continue
            uids, p = getp()
            if uids is None or p is None:
                continue
            uids = np.asarray(uids, dtype=int)
            p = np.asarray(p, dtype=float)
            if len(uids) == 0:
                continue
            pressures.append(_DeathPressure(name=getattr(mod, "name", mod.__class__.__name__), uids=uids, p=p))
        return pressures

    def _categorical_one(self, weights, rng):
        """Return index sampled proportional to nonnegative weights."""
        tot = float(weights.sum())
        if not np.isfinite(tot) or tot <= 0:
            return len(weights) - 1  # last entry (residual)
        r = rng.random() * tot
        c = 0.0
        for i, w in enumerate(weights):
            c += float(w)
            if r <= c:
                return i
        return len(weights) - 1

    def step(self):
        sim = self.sim
        ppl = sim.people
        ti = sim.ti

        # Reset current-timestep cause map
        self._cause_map = {}
        setattr(sim, "_mighti_death_cause", self._cause_map)

        auids = np.asarray(ppl.auids, dtype=int)
        if len(auids) == 0:
            self.results.new_deaths[ti] = 0
            return 0

        p_all = self._make_p_allcause(auids)
        pressures = self._collect_death_pressures()

        # Build dense modeled pressure arrays for simple per-UID allocation
        n = int(ppl.uid.len_used)
        modeled = {}
        for pr in pressures:
            arr = np.zeros(n, dtype=float)
            # Clip and ignore out-of-range uids (can happen with module bookkeeping)
            valid = (pr.uids >= 0) & (pr.uids < n)
            uu = pr.uids[valid]
            pp = np.clip(pr.p[valid], 0.0, 1.0)
            arr[uu] = pp
            modeled[pr.name] = arr

        p_all_full = np.zeros(n, dtype=float)
        p_all_full[auids] = p_all

        if modeled:
            p_model_sum = np.zeros(n, dtype=float)
            for arr in modeled.values():
                p_model_sum += arr

            # If modeled exceeds all-cause for a UID, rescale modeled causes down and set residual to 0
            over = (p_model_sum > p_all_full) & (p_model_sum > 0) & (p_all_full > 0)
            if np.any(over):
                scale = np.ones(n, dtype=float)
                scale[over] = p_all_full[over] / p_model_sum[over]
                for k in list(modeled.keys()):
                    modeled[k] = modeled[k] * scale
                # recompute
                p_model_sum[:] = 0.0
                for arr in modeled.values():
                    p_model_sum += arr

            residual = np.clip(p_all_full - p_model_sum, 0.0, 1.0)
        else:
            p_model_sum = np.zeros(n, dtype=float)
            residual = p_all_full.copy()

        # Draw who dies (all-cause)
        rng = get_rng(sim, salt="CompetingRisksDeaths:step")
        die_flags = rng.random(len(auids)) < p_all
        death_uids = auids[die_flags]

        # Attribute cause conditional on death
        if len(death_uids):
            cause_names = list(modeled.keys()) + [self.pars.residual_cause_name]
            for uid in death_uids:
                weights = np.array([modeled[k][uid] for k in modeled.keys()] + [residual[uid]], dtype=float)
                idx = self._categorical_one(weights, rng)
                self._cause_map[int(uid)] = cause_names[idx]
            ppl.request_death(death_uids)

        # Record simple shares for debugging
        with np.errstate(divide="ignore", invalid="ignore"):
            share_modeled = np.divide(p_model_sum, p_all_full, out=np.zeros_like(p_all_full), where=p_all_full > 0)
            share_resid = np.divide(residual, p_all_full, out=np.zeros_like(p_all_full), where=p_all_full > 0)
        self.results.modeled_share_mean[ti] = float(np.nanmean(share_modeled[auids])) if len(auids) else 0.0
        self.results.residual_share_mean[ti] = float(np.nanmean(share_resid[auids])) if len(auids) else 0.0
        self.results.new_deaths[ti] = int(len(death_uids))

        return int(len(death_uids))

