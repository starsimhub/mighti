"""
Analyzers for demographic outcomes such as age-specific deaths and survivorship.
"""

import numpy as np
import pandas as pd
import starsim as ss

from mighti.analyzers.disability_weights import resolve_disease_module


__all__ = [
    "DeathsByAgeSexAnalyzer",
    "AgeSexMxAnalyzer",
    "SurvivorshipAnalyzer",
    "ConditionAtDeathAnalyzer",
    "CauseOfDeathYLLAnalyzer",
]

def _raw_view(arr: "ss.Arr", n_used: int, dtype) -> np.ndarray:
    """
    Return a uid-indexed numpy view of the raw values for `[0, n_used)`.

    Starsim's `np.asarray(Arr)` returns ``raw[auids]`` (a positional slice over
    *currently-active* uids). For across-step comparisons (e.g. detecting who
    newly died) and for indexing other uid-keyed arrays, we need the *raw* uid-
    indexed underlying buffer, truncated to the number of uids ever allocated.
    """
    return np.asarray(arr.raw[:n_used], dtype=dtype)


class DeathsByAgeSexAnalyzer(ss.Analyzer):
    """Tracks infant deaths and deaths by age/sex, Starsim-3.0.3 compatible."""

    def __init__(self, max_age=100, **kwargs):
        super().__init__(**kwargs)
        self.max_age = max_age
        self._dead_prev = None  # uid-indexed snapshot of who was dead in previous step

    def init_pre(self, sim):
        super().init_pre(sim)
        n = int(sim.people.uid.len_used)
        self._dead_prev = np.zeros(n, dtype=bool)

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result("infant_deaths", label="Cumulative infant deaths", dtype=int),
            ss.Result("male_deaths_by_age",   label="Male deaths by age",   dtype=int, shape=self.max_age + 1),
            ss.Result("female_deaths_by_age", label="Female deaths by age", dtype=int, shape=self.max_age + 1),
        )

    def _ensure_size(self, n_used: int):
        """Grow `_dead_prev` to cover `n_used` uids (monotonically increasing)."""
        if self._dead_prev is None:
            self._dead_prev = np.zeros(n_used, dtype=bool)
        elif self._dead_prev.size < n_used:
            new = np.zeros(n_used, dtype=bool)
            new[: self._dead_prev.size] = self._dead_prev
            self._dead_prev = new

    def step(self):
        ppl = self.sim.people
        ti = self.sim.ti
        n_used = int(ppl.uid.len_used)
        self._ensure_size(n_used)

        # Use raw, uid-indexed views so that positional indexing into auids cannot
        # mis-map a "position" to the wrong uid after deaths / new embryos.
        alive_raw = _raw_view(ppl.alive, n_used, bool)
        ages_raw = _raw_view(ppl.age, n_used, float)
        fem_raw = _raw_view(ppl.female, n_used, bool)
        dead_raw = ~alive_raw

        # Cumulative infant deaths: any uid currently dead with age in [0, 1).
        # This is a per-step snapshot (overwrite), matching how the analyzer is
        # consumed downstream.
        self.results.infant_deaths[ti] = int(
            np.count_nonzero(dead_raw & (ages_raw >= 0.0) & (ages_raw < 1.0))
        )

        # New deaths this step: dead now but not at the prior snapshot. These
        # indices ARE uids (raw arrays are uid-indexed).
        new_dead_uids = np.where(dead_raw & ~self._dead_prev[:n_used])[0]
        if len(new_dead_uids):
            born = ages_raw[new_dead_uids] >= 0.0
            new_dead_uids = new_dead_uids[born]
        if len(new_dead_uids):
            ages = np.clip(np.floor(ages_raw[new_dead_uids]).astype(int), 0, self.max_age)
            fem = fem_raw[new_dead_uids]
            if np.any(fem):
                idx, cnt = np.unique(ages[fem], return_counts=True)
                self.results.female_deaths_by_age[idx] += cnt
            if np.any(~fem):
                idx, cnt = np.unique(ages[~fem], return_counts=True)
                self.results.male_deaths_by_age[idx] += cnt

        # Update snapshot
        self._dead_prev[:n_used] = dead_raw

    def finalize(self):
        super().finalize() 

    def finalize_results(self):
        super().finalize_results()

    def to_df(self):
        ages = np.arange(self.max_age + 1)
        return pd.concat([
            pd.DataFrame({
                "age": ages, "sex": "Male",
                "deaths": self.results.male_deaths_by_age[:],
            }),
            pd.DataFrame({
                "age": ages, "sex": "Female",
                "deaths": self.results.female_deaths_by_age[:],
            }),
        ], ignore_index=True)


class AgeSexMxAnalyzer(ss.Analyzer):
    """
    Record per-timestep age-sex exposure and deaths to estimate realized m(x).

    - Exposure is counted at the start of the timestep via `start_step()` using alive agents.
    - Deaths are counted after deaths are resolved (analyzer `step()` runs after `people.step_die()`).

    This yields a simple period estimate:

        m_x(t) ≈ deaths_x(t) / exposure_x(t) / dt_year

    where exposure_x(t) is the number of alive agents aged in [x, x+1) at the start of the step.
    """

    def __init__(self, max_age=100, **kwargs):
        super().__init__(**kwargs)
        # Stable key for `sim.analyzers.age_sex_mx_analyzer`
        self.name = "age_sex_mx_analyzer"
        self.max_age = max_age
        self._dead_prev = None  # uid-indexed snapshot of dead state

    def init_pre(self, sim):
        super().init_pre(sim)
        n = int(sim.people.uid.len_used)
        self._dead_prev = np.zeros(n, dtype=bool)

    def init_results(self):
        super().init_results()
        npts = int(self.sim.t.npts)
        shape = (npts, self.max_age + 1)
        self.define_results(
            ss.Result("exposure_male", dtype=float, shape=shape, label="Exposure (male) by age"),
            ss.Result("exposure_female", dtype=float, shape=shape, label="Exposure (female) by age"),
            ss.Result("deaths_male", dtype=int, shape=shape, label="Deaths (male) by age"),
            ss.Result("deaths_female", dtype=int, shape=shape, label="Deaths (female) by age"),
        )

    def _ensure_size(self, n_used: int):
        """Grow `_dead_prev` to cover `n_used` uids (monotonically increasing)."""
        if self._dead_prev is None:
            self._dead_prev = np.zeros(n_used, dtype=bool)
        elif self._dead_prev.size < n_used:
            new = np.zeros(n_used, dtype=bool)
            new[: self._dead_prev.size] = self._dead_prev
            self._dead_prev = new

    def start_step(self):
        """Count exposure by age/sex at start of step."""
        super().start_step()
        ppl = self.sim.people
        ti = self.sim.ti
        n_used = int(ppl.uid.len_used)
        self._ensure_size(n_used)

        auids = np.asarray(ppl.auids, dtype=int)
        if len(auids) == 0:
            return

        # Exclude unborn agents (negative age == in-utero embryos from ss.Pregnancy).
        # Without this, embryos collapse into the age-0 bin via floor(-x)→-1 and the
        # subsequent clip(..., 0, max_age), inflating the denominator of m(0).
        ages_f = np.asarray(ppl.age[auids], dtype=float)
        born = ages_f >= 0.0
        auids = auids[born]
        if len(auids) == 0:
            return

        ages = np.clip(np.floor(ppl.age[auids]).astype(int), 0, self.max_age)
        fem = np.asarray(ppl.female[auids], dtype=bool)

        # Counts for this timestep only (overwrite; shape is time-indexed)
        if np.any(fem):
            idx, cnt = np.unique(ages[fem], return_counts=True)
            self.results.exposure_female[ti, idx] = cnt.astype(float)
        if np.any(~fem):
            idx, cnt = np.unique(ages[~fem], return_counts=True)
            self.results.exposure_male[ti, idx] = cnt.astype(float)
        return

    def step(self):
        """Count new deaths this step by age/sex."""
        ppl = self.sim.people
        ti = self.sim.ti
        n_used = int(ppl.uid.len_used)
        self._ensure_size(n_used)

        # Use raw, uid-indexed views: positional indexing into auids becomes
        # mis-aligned with uids after agents die (remove_dead compacts auids)
        # or new embryos appear. Raw arrays are always uid-indexed.
        alive_raw = _raw_view(ppl.alive, n_used, bool)
        ages_raw = _raw_view(ppl.age, n_used, float)
        fem_raw = _raw_view(ppl.female, n_used, bool)
        dead_raw = ~alive_raw

        new_dead_uids = np.where(dead_raw & ~self._dead_prev[:n_used])[0]
        if len(new_dead_uids):
            # Exclude in-utero losses (negative age at death) from period m(x) tally.
            born = ages_raw[new_dead_uids] >= 0.0
            new_dead_uids = new_dead_uids[born]
        if len(new_dead_uids):
            ages = np.clip(np.floor(ages_raw[new_dead_uids]).astype(int), 0, self.max_age)
            fem = fem_raw[new_dead_uids]
            if np.any(fem):
                idx, cnt = np.unique(ages[fem], return_counts=True)
                self.results.deaths_female[ti, idx] = cnt
            if np.any(~fem):
                idx, cnt = np.unique(ages[~fem], return_counts=True)
                self.results.deaths_male[ti, idx] = cnt

        self._dead_prev[:n_used] = dead_raw
        return

    def to_mx_df(self, *, year=None):
        """Return tidy DataFrame with realized m(x) for the requested year (or last year)."""
        years = np.asarray(self.sim.t.yearvec, dtype=float)
        if year is None:
            ti = int(self.sim.t.npts - 1)
        else:
            ti = int(np.argmin(np.abs(years - float(year))))

        dt_year = float(getattr(self.sim.t, "dt_year", 1.0))
        ages = np.arange(self.max_age + 1)

        def make(sex):
            if sex == "Female":
                deaths = np.asarray(self.results.deaths_female[ti, :], dtype=float)
                expo = np.asarray(self.results.exposure_female[ti, :], dtype=float)
            else:
                deaths = np.asarray(self.results.deaths_male[ti, :], dtype=float)
                expo = np.asarray(self.results.exposure_male[ti, :], dtype=float)
            # Use NaN when exposure is 0 to avoid divide-by-zero artifacts/spikes and
            # to keep log-scale plotting sane.
            mx = np.divide(deaths, expo, out=np.full_like(deaths, np.nan), where=expo > 0) / dt_year
            return pd.DataFrame({"age": ages, "sex": sex, "mx": mx, "deaths": deaths, "exposure": expo})

        out = pd.concat([make("Female"), make("Male")], ignore_index=True)
        out["year"] = float(years[ti])
        return out


class SurvivorshipAnalyzer(ss.Analyzer):
    """
    Computes true survivorship l(x): fraction of original cohort surviving to age x.
    Compatible with all disease types (including neonatal).
    """

    def __init__(self, max_age=100, **kwargs):
        super().__init__(**kwargs)
        self.name = "survivorship_analyzer"
        self.max_age = max_age

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result("lx_male",   shape=self.max_age + 1, dtype=float, label="Male survivorship l(x)"),
            ss.Result("lx_female", shape=self.max_age + 1, dtype=float, label="Female survivorship l(x)"),
        )

    def step(self):
        """No per-step logic (needed only to silence lifecycle warning)."""
        pass

    def finalize(self):
        """Compute l(x) at the end of the sim."""
        super().finalize() 
        ppl = self.sim.people
        n0_m = max(np.sum(~ppl.female), 1)
        n0_f = max(np.sum(ppl.female), 1)

        lx_m = np.zeros(self.max_age + 1)
        lx_f = np.zeros(self.max_age + 1)

        for a in range(self.max_age + 1):
            alive_m = (~ppl.female) & (~ppl.dead) & (ppl.age >= a)
            alive_f = (ppl.female) & (~ppl.dead) & (ppl.age >= a)
            lx_m[a] = np.sum(alive_m) / n0_m
            lx_f[a] = np.sum(alive_f) / n0_f

        self.results.lx_male[:] = lx_m
        self.results.lx_female[:] = lx_f

    def finalize_results(self):
        super().finalize_results()

    def to_df(self):
        records = []
        year = int(self.sim.t.yearvec[-1])
        for sex, key in (("Male", "lx_male"), ("Female", "lx_female")):
            lx = getattr(self.results, key)
            for age, val in enumerate(lx):
                records.append({
                    "year": year, "age": age, "sex": sex, "survival": float(val)
                })
        return pd.DataFrame(records)
    

class ConditionAtDeathAnalyzer(ss.Analyzer):
    """
    Simple analyzer that records who died, their age/sex/YLL,
    and which non-HIV conditions they had or died from.

    Notes:
    - HIV deaths are excluded (use the HIV module directly).
    - Analyzer still records HIV-positive status at death.
    - No cause grouping logic here (handled in main scripts).
    """

    def __init__(self, conditions=None, ex_life_expectancy=80.0, **kwargs):
        super().__init__(**kwargs)
        # Store non-HIV conditions only (HIV tracked separately)
        self.conditions = [c.lower() for c in (conditions or []) if c.lower() != "hiv"]
        # Back-compat: this used to be a single scalar "reference age". It can now
        # also be a callable(sex, age)->remaining LE, or a tidy DataFrame with
        # columns ['age','sex','ex'] representing remaining LE e(x).
        self.ex_life_expectancy = ex_life_expectancy
        self._ex_lookup = None
        self.records = []
        self.name = "condition_at_death_analyzer"

    @staticmethod
    def _make_ex_lookup_from_df(df_ex):
        cols = {c.lower(): c for c in df_ex.columns}
        for req in ("age", "sex", "ex"):
            if req not in cols:
                raise ValueError("ex_life_expectancy DataFrame must include columns: age, sex, ex")
        d = df_ex.rename(columns={cols["age"]: "age", cols["sex"]: "sex", cols["ex"]: "ex"}).copy()
        d["age"] = pd.to_numeric(d["age"], errors="coerce").fillna(0).astype(int)
        d["sex"] = d["sex"].astype(str).str.strip().str.title()
        d["ex"] = pd.to_numeric(d["ex"], errors="coerce").fillna(0.0).astype(float)
        max_age = int(d["age"].max()) if len(d) else 0
        table = {(r.sex, int(r.age)): float(r.ex) for r in d.itertuples(index=False)}

        def lookup(sex, age):
            s = str(sex).strip().title()
            a = int(np.floor(float(age)))
            if a < 0:
                a = 0
            if a > max_age:
                return 0.0
            return float(table.get((s, a), 0.0))

        return lookup

    def init_pre(self, sim):
        super().init_pre(sim)
        ex = self.ex_life_expectancy
        if callable(ex):
            self._ex_lookup = ex  # callable(sex, age)->remaining LE
        elif isinstance(ex, pd.DataFrame):
            self._ex_lookup = self._make_ex_lookup_from_df(ex)
        else:
            self._ex_lookup = None

    def init_results(self):
        """Initialize analyzer results."""
        super().init_results()
        # starsim ≥3.2 locks ``results``; do not replace the container.
        # Primary output is ``self.records`` / ``to_df()``.
        try:
            self.results["n_deaths"] = 0
            self.results["by_cause"] = {}
            self.results["by_sex"] = {}
            self.results["by_age"] = {}
        except Exception:
            self._death_summary = {
                "n_deaths": 0,
                "by_cause": {},
                "by_sex": {},
                "by_age": {},
            }

    def _had_condition(self, disease, uid):
        """Return True if agent had this condition at death."""
        if disease is None:
            return False
        for attr in ("infected", "affected", "active"):
            if hasattr(disease, attr):
                arr = getattr(disease, attr)
                try:
                    return bool(arr[uid])
                except Exception:
                    return False
        return False

    def _died_of_condition(self, disease, uid, ti):
        """Return True if this module has ti_dead and matches current step."""
        if disease is None or not hasattr(disease, "ti_dead"):
            return False
        ti_dead = getattr(disease, "ti_dead")
        try:
            return bool((ti_dead == ti)[uid])
        except Exception:
            raw = getattr(ti_dead, "raw", None)
            if raw is None:
                return False
            return bool(np.isfinite(raw[uid]) and raw[uid] == ti)

    def step(self):
        ppl = self.sim.people
        ti = self.sim.t.ti
        year = float(self.sim.t.yearvec[ti])
        died_now = ppl.dead.uids
        if not len(died_now):
            return

        # Reference HIV module to check HIV infection at death
        hiv_mod = getattr(self.sim.diseases, "hiv", None)

        for uid in died_now:
            age = float(ppl.age[uid])
            sex = "Female" if ppl.female[uid] else "Male"

            # Years of life lost (YLL)
            # - If `ex_life_expectancy` is callable or a DataFrame: interpret as remaining LE e(x)
            # - If numeric: interpret as a constant reference age (back-compat)
            if self._ex_lookup is not None:
                yll = max(0.0, float(self._ex_lookup(sex, age)))
            else:
                ref_age = 75.0 if sex == "Female" else 70.0
                if isinstance(self.ex_life_expectancy, (int, float)):
                    ref_age = float(self.ex_life_expectancy)
                yll = max(0.0, ref_age - age)

            rec = dict(uid=int(uid), year=year, age=age, sex=sex, yll=yll)

            # Record presence and cause flags for each disease.
            # Resolve via alias-tolerant lookup (labels may be PascalCase).
            for cond in self.conditions:
                disease, _resolved = resolve_disease_module(self.sim.diseases, cond)
                rec[f"died_{cond}"] = self._had_condition(disease, uid)
                rec[f"cause_{cond}"] = self._died_of_condition(disease, uid, ti)

            # Record HIV infection status (not death cause)
            rec["hiv_positive"] = False
            if hiv_mod is not None:
                try:
                    if hasattr(hiv_mod, "infected"):
                        rec["hiv_positive"] = bool(hiv_mod.infected[uid])
                except Exception:
                    pass
                if not rec["hiv_positive"] and hasattr(hiv_mod, "ti_infected"):
                    try:
                        raw = getattr(hiv_mod.ti_infected, "raw", None)
                        ti_val = float(raw[uid]) if raw is not None else float(hiv_mod.ti_infected[uid])
                        rec["hiv_positive"] = bool(np.isfinite(ti_val))
                    except Exception:
                        pass

            # Determine primary cause (non-HIV only)
            causes = [c for c in self.conditions if rec.get(f"cause_{c}", False)]
            if len(causes) == 1:
                rec["primary_cause"] = causes[0]
            elif len(causes) > 1:
                rec["primary_cause"] = "multiple"
            else:
                rec["primary_cause"] = None

            self.records.append(rec)

    def to_df(self):
        """Return DataFrame of recorded deaths and conditions."""
        return pd.DataFrame(self.records)


class CauseOfDeathYLLAnalyzer(ss.Analyzer):
    """
    Record deaths with cause-of-death labels (if available) and YLL against a
    reference remaining life expectancy table.

    Cause labels
    ------------
    If `CompetingRisksDeaths` is used, it writes a per-timestep dict:
      `sim._mighti_death_cause: {uid -> cause_name}`
    This analyzer reads from that dict for each newly-dead uid. If absent, the
    cause is recorded as "unknown".

    Reference life expectancy
    -------------------------
    `reference_ex` may be:
    - callable(sex, age)->remaining LE e(x)
    - tidy DataFrame with columns ['age','sex','ex'] representing e(x)
    """

    def __init__(self, reference_ex, *, max_age=100, **kwargs):
        super().__init__(**kwargs)
        self.name = "cause_of_death_yll_analyzer"
        self.max_age = int(max_age)
        self.reference_ex = reference_ex
        self._ex_lookup = None
        # uid-indexed snapshot of who was dead at the previous step
        self._dead_prev = None
        self.records = []

    def init_pre(self, sim):
        super().init_pre(sim)
        n = int(sim.people.uid.len_used)
        self._dead_prev = np.zeros(n, dtype=bool)
        ex = self.reference_ex
        if callable(ex):
            self._ex_lookup = ex
        elif isinstance(ex, pd.DataFrame):
            self._ex_lookup = ConditionAtDeathAnalyzer._make_ex_lookup_from_df(ex)
        else:
            raise ValueError("reference_ex must be a callable or a DataFrame with columns: age, sex, ex")

    def _ensure_size(self, n_used: int):
        """Grow `_dead_prev` to cover `n_used` uids (monotonically increasing)."""
        if self._dead_prev is None:
            self._dead_prev = np.zeros(n_used, dtype=bool)
        elif self._dead_prev.size < n_used:
            new = np.zeros(n_used, dtype=bool)
            new[: self._dead_prev.size] = self._dead_prev
            self._dead_prev = new

    def step(self):
        ppl = self.sim.people
        ti = self.sim.ti
        year = float(self.sim.t.yearvec[ti])
        n_used = int(ppl.uid.len_used)
        self._ensure_size(n_used)

        # Use raw, uid-indexed views. Without this, np.asarray(ppl.dead) returns
        # a positional slice over auids, np.where(...) returns positions, and the
        # subsequent cause_map[int(uid)] / ppl.age[uid] / ppl.female[uid] lookups
        # mis-map almost every newly-dead agent to cause="unknown".
        alive_raw = _raw_view(ppl.alive, n_used, bool)
        ages_raw = _raw_view(ppl.age, n_used, float)
        fem_raw = _raw_view(ppl.female, n_used, bool)
        dead_raw = ~alive_raw

        new_dead_uids = np.where(dead_raw & ~self._dead_prev[:n_used])[0]
        if len(new_dead_uids):
            # Exclude in-utero losses (negative age at death) so that cause
            # attribution and YLL match the m(x) / deaths analyzers' scope.
            born = ages_raw[new_dead_uids] >= 0.0
            new_dead_uids = new_dead_uids[born]

        if len(new_dead_uids):
            cause_map = getattr(self.sim, "_mighti_death_cause", {}) or {}
            ages_dead = ages_raw[new_dead_uids]
            fem_dead = fem_raw[new_dead_uids]
            for uid, age, is_female in zip(new_dead_uids, ages_dead, fem_dead):
                sex = "Female" if bool(is_female) else "Male"
                yll = max(0.0, float(self._ex_lookup(sex, float(age)))) if self._ex_lookup is not None else 0.0
                cause = str(cause_map.get(int(uid), "unknown"))
                self.records.append(
                    dict(uid=int(uid), year=year, age=float(age), sex=sex, cause=cause, yll=yll)
                )

        self._dead_prev[:n_used] = dead_raw

    def to_df(self):
        return pd.DataFrame(self.records)
    