"""
Additive-hazard mortality with cause attribution.

Motivation
----------
`CompetingRisksDeaths` caps total deaths to an observed all-cause m(x) schedule to
avoid double-counting and to enable clean attribution. That is great for
"decompose observed all-cause into modeled vs residual".

However, for Stevens-style "idealized/reference life expectancy" and for scenario
forecasting, we typically need a mortality mechanism where *removing a modeled
cause reduces all-cause mortality*, rather than simply shifting deaths into a
residual bucket.

This module implements a simple additive-hazard structure:

  hazard_total(uid) = hazard_background(uid) * background_multiplier
                      + sum_i hazard_i(uid)

and converts the total hazard into a per-timestep death probability.

Modeled cause hazards are obtained from disease modules that implement
`get_death_pressure()` (see `mighti.diseases.base_disease._CompetingMortalityMixin`
and `mighti.stisim_competing.HIVCompeting`). Pressures are interpreted as
per-timestep probabilities and converted to hazards via `-log(1-p)`.

This is intentionally pragmatic: it's a bridge that supports
  - calibration of "background" mortality with a chosen disease set enabled, and
  - forecasting/scenario modeling holding the calibrated background constant.
"""

from typing import Mapping

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

from mighti.util.rng import get_rng

__all__ = ["AdditiveHazardDeaths"]


class _DeathPressure:
    def __init__(self, name, uids, p):
        self.name = name
        self.uids = uids
        self.p = p  # per-timestep probability-like pressure


def _p_to_h(p):
    """Convert per-step death probability to hazard: h = -log(1-p)."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, 1.0 - 1e-12)
    return -np.log1p(-p)


class AdditiveHazardDeaths(ss.Disease):
    """
    All-cause deaths from background + modeled hazards, with cause attribution.

    Parameters (similar to StarSim `ss.Deaths` / MIGHTI `CompetingRisksDeaths`)
    ----------
    background_rate:
        Observed all-cause m(x) style dataframe/series/rate/number used as the
        *starting point* for background mortality. Typically you will calibrate
        `background_multiplier` such that the simulated LE matches observed LE
        with a chosen set of disease modules enabled.
    background_multiplier:
        Scalar applied to the background hazard component (not the probability).
        See `multiplier_min_age` for piecewise scoping.
    multiplier_min_age:
        Minimum age (in years) at which `background_multiplier` is applied. For
        agents with ``age < multiplier_min_age``, the background hazard uses
        the *unscaled* schedule (multiplier = 1.0). This lets you assert the
        WPP all-cause schedule at the steep infant/early-childhood ages while
        still calibrating a single global multiplier for the rest of the life
        course. Default is 0.0 (multiplier applies at all ages).
    metadata:
        Column mapping + sex key mapping for the dataframe format.
    background_cause_name:
        Label used for the background/unmodeled cause in the attribution map.
    """

    def __init__(
        self,
        background_rate,
        *,
        background_multiplier=1.0,
        multiplier_min_age=0.0,
        rate_units=1e-3,
        metadata=None,
        background_cause_name="background",
        **kwargs,
    ):
        super().__init__()
        self.define_pars(
            background_rate=background_rate,
            background_multiplier=float(background_multiplier),
            multiplier_min_age=float(multiplier_min_age),
            rate_units=float(rate_units),
            background_cause_name=str(background_cause_name),
        )
        self.update_pars(kwargs)

        self.metadata = sc.mergedicts(
            sc.objdict(
                data_cols=dict(year="Time", sex="Sex", age="AgeGrpStart", value="mx"),
                # Accept both UN/WPP-style labels and Starsim internal labels
                sex_keys={"Female": "f", "Male": "m", "f": "f", "m": "m"},
            ),
            metadata,
        )

        self.background_rate_data = None
        self._cause_map = {}  # uid -> cause name, current timestep
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        # Enable "competing mortality protocol" (modules report death pressure)
        setattr(sim, "_mighti_competing_mortality", True)

        brd = ss.standardize_data(data=self.pars.background_rate, metadata=self.metadata)
        if isinstance(brd, (pd.Series, pd.DataFrame)):
            brd = brd.unstack(level="age")
            assert not brd.isna().any(axis=None)
        if sc.isnumber(brd):
            ss.warn(f"AdditiveHazardDeaths.background_rate specified as number ({brd}); assuming per year")
            brd = ss.peryear(brd)
        self.background_rate_data = brd
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result("new_deaths", dtype=int, scale=True, summarize_by="sum", auto_plot=False),
            ss.Result("background_share_mean", dtype=float, scale=False, summarize_by="mean", auto_plot=False),
            ss.Result("modeled_share_mean", dtype=float, scale=False, summarize_by="mean", auto_plot=False),
        )
        return

    def _make_p_background(self, uids):
        """Per-step background death probability for these uids (before hazard scaling)."""
        sim = self.sim
        brd = self.background_rate_data
        p = self.pars

        if sc.isnumber(brd):
            rate = np.array([brd * p.rate_units])

        elif isinstance(brd, ss.Rate):
            if brd.unit == 1:
                rate = np.array([brd.value * p.rate_units])
            else:
                rate = np.array([ss.prob(brd.value * p.rate_units, brd.unit).to_prob(ss.years(1))])

        else:
            ppl = sim.people
            available_years = brd.index.get_level_values("year")
            year_ind = sc.findnearest(available_years, sim.t.now("year"))
            nearest_year = available_years[year_ind]

            rate = np.empty(uids.shape, dtype=ss.dtypes.float)

            if "sex" in brd.index.names:
                s = brd.loc[nearest_year, "f"]
                female_mask = np.asarray(ppl.female[uids], dtype=bool)
                fu = uids[female_mask]
                if len(fu):
                    # Clamp negative ages (in-utero embryos from ss.Pregnancy) to 0
                    # for the age-bin lookup; their final hazard is zeroed below.
                    ages_f = np.maximum(np.asarray(ppl.age[fu], dtype=float), 0.0)
                    b = np.clip(np.digitize(ages_f, s.index) - 1, 0, len(s.values) - 1)
                    rate[female_mask] = s.values[b]

                s = brd.loc[nearest_year, "m"]
                male_mask = np.asarray(ppl.male[uids], dtype=bool)
                mu = uids[male_mask]
                if len(mu):
                    ages_m = np.maximum(np.asarray(ppl.age[mu], dtype=float), 0.0)
                    b = np.clip(np.digitize(ages_m, s.index) - 1, 0, len(s.values) - 1)
                    rate[male_mask] = s.values[b]
            else:
                s = brd.loc[nearest_year]
                ages_all = np.maximum(np.asarray(ppl.age[uids], dtype=float), 0.0)
                b = np.clip(np.digitize(ages_all, s.index) - 1, 0, len(s.values) - 1)
                rate[:] = s.values[b]

            rate *= p.rate_units

        rate = ss.peryear(rate)
        p_bg = rate.to_prob(self.t.dt)
        if sc.isnumber(brd) or isinstance(brd, ss.Rate):
            p_bg = np.full(uids.shape, float(p_bg[0]), dtype=float)
        p_bg = np.asarray(p_bg, dtype=float).copy()
        # Background mortality should not act on unborn (in-utero) agents: their
        # "age" is negative until delivery (`ss.Pregnancy.make_embryos`).
        try:
            ages_q = np.asarray(sim.people.age[uids], dtype=float)
            p_bg[ages_q < 0.0] = 0.0
        except Exception:
            pass
        return np.clip(p_bg, 0.0, 1.0)

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
        tot = float(weights.sum())
        if not np.isfinite(tot) or tot <= 0:
            return int(len(weights) - 1)
        r = rng.random() * tot
        c = 0.0
        for i, w in enumerate(weights):
            c += float(w)
            if r <= c:
                return int(i)
        return int(len(weights) - 1)

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

        # Background hazard. The multiplier is applied piecewise by age so that
        # the WPP all-cause schedule can be asserted at full strength at the
        # steep infant cliff while a single global multiplier scales the rest of
        # the life course (calibrated to match observed e0).
        p_bg = self._make_p_background(auids)
        h_bg_raw = _p_to_h(p_bg)
        mult_full = float(self.pars.background_multiplier)
        min_age = float(self.pars.multiplier_min_age)
        if min_age > 0.0:
            ages_q = np.asarray(ppl.age[auids], dtype=float)
            mult_arr = np.where(ages_q < min_age, 1.0, mult_full)
            h_bg = h_bg_raw * mult_arr
        else:
            h_bg = h_bg_raw * mult_full

        # Modeled hazards (dense arrays for attribution)
        pressures = self._collect_death_pressures()
        n = int(ppl.uid.len_used)
        modeled_h = {}
        for pr in pressures:
            arr = np.zeros(n, dtype=float)
            valid = (pr.uids >= 0) & (pr.uids < n)
            uu = pr.uids[valid]
            pp = np.clip(pr.p[valid], 0.0, 1.0)
            arr[uu] = _p_to_h(pp)
            modeled_h[pr.name] = arr

        # Total hazard per active uid (vectorized)
        h_total = h_bg.copy()
        if modeled_h:
            for arr in modeled_h.values():
                h_total += arr[auids]

        # Convert to per-step death probability
        p_all = 1.0 - np.exp(-np.clip(h_total, 0.0, np.inf))
        p_all = np.clip(p_all, 0.0, 1.0)

        # Draw deaths
        rng = get_rng(sim, salt="AdditiveHazardDeaths:step")
        die_flags = rng.random(len(auids)) < p_all
        death_uids = auids[die_flags]

        # Attribute causes proportional to hazard components
        if len(death_uids):
            # Map uid -> index into auids/h_bg arrays (fast lookup inside loop)
            uid_to_i = {int(u): int(i) for i, u in enumerate(auids)}
            cause_names = list(modeled_h.keys()) + [self.pars.background_cause_name]
            for uid in death_uids:
                bg_h = float(h_bg[uid_to_i[int(uid)]])
                weights = np.array([float(modeled_h[k][uid]) for k in modeled_h.keys()] + [bg_h], dtype=float)
                idx = self._categorical_one(weights, rng)
                self._cause_map[int(uid)] = str(cause_names[idx])
            ppl.request_death(death_uids)

        # Debug shares (mean over active uids)
        h_bg_full = np.zeros(n, dtype=float)
        h_bg_full[auids] = h_bg
        h_model_sum = np.zeros(n, dtype=float)
        for arr in modeled_h.values():
            h_model_sum += arr
        h_tot_full = h_bg_full + h_model_sum

        with np.errstate(divide="ignore", invalid="ignore"):
            share_model = np.divide(h_model_sum, h_tot_full, out=np.zeros_like(h_tot_full), where=h_tot_full > 0)
            share_bg = np.divide(h_bg_full, h_tot_full, out=np.zeros_like(h_tot_full), where=h_tot_full > 0)
        self.results.modeled_share_mean[ti] = float(np.nanmean(share_model[auids])) if len(auids) else 0.0
        self.results.background_share_mean[ti] = float(np.nanmean(share_bg[auids])) if len(auids) else 0.0
        self.results.new_deaths[ti] = int(len(death_uids))
        return int(len(death_uids))

