"""
Analyzers for demographic outcomes: deaths by age and sex, realized m(x), survivorship, and deaths by cause with years of life lost
"""

import numpy as np
import pandas as pd
import starsim as ss

from mighti.analyzers.disability_weights import resolve_disease_module
from mighti.analysis.life_expectancy import make_ex_lookup


__all__ = [
    "DeathsByAgeSexAnalyzer",
    "AgeSexMxAnalyzer",
    "SurvivorshipAnalyzer",
    "ConditionAtDeathAnalyzer",
    "CauseOfDeathYLLAnalyzer",
]


def new_deaths(sim):
    """ The agents who died in this step (not those who left otherwise), excluding the unborn """
    ppl = sim.people
    dying = ppl.dying()
    dead = dying[ppl.ti_dead.raw[dying] <= sim.ti]
    return dead[ppl.age[dead] >= 0]


def by_age(ages, max_age):
    """ The number of people at each single year of age, from 0 to max_age (the last open-ended) """
    return np.bincount(np.clip(np.floor(ages).astype(int), 0, max_age), minlength=max_age + 1)


class DeathsByAgeSexAnalyzer(ss.Analyzer):
    """ Deaths by single year of age and sex over the run, and the cumulative number of infant deaths """

    def __init__(self, max_age=100, **kwargs):
        super().__init__(**kwargs)
        self.max_age = max_age
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result("infant_deaths", label="Cumulative infant deaths", dtype=int),
            ss.Result("male_deaths_by_age", label="Male deaths by age", dtype=int, shape=self.max_age + 1),
            ss.Result("female_deaths_by_age", label="Female deaths by age", dtype=int, shape=self.max_age + 1),
        )
        return

    def step(self):
        ppl = self.sim.people
        ti = self.ti
        dead = new_deaths(self.sim)
        ages = ppl.age[dead]
        fem = ppl.female[dead]
        self.results.female_deaths_by_age[:] += by_age(ages[fem], self.max_age)
        self.results.male_deaths_by_age[:] += by_age(ages[~fem], self.max_age)
        prev = self.results.infant_deaths[ti-1] if ti > 0 else 0
        self.results.infant_deaths[ti] = prev + np.count_nonzero(ages < 1)
        return

    def to_df(self):
        ages = np.arange(self.max_age + 1)
        return pd.concat([
            pd.DataFrame({"age": ages, "sex": "Male", "deaths": self.results.male_deaths_by_age[:]}),
            pd.DataFrame({"age": ages, "sex": "Female", "deaths": self.results.female_deaths_by_age[:]}),
        ], ignore_index=True)


class AgeSexMxAnalyzer(ss.Analyzer):
    """
    Exposure and deaths by single year of age and sex at each time point, to estimate the realized m(x):

        m_x(t) ≈ deaths_x(t) / exposure_x(t) / dt

    where exposure_x(t) is the number alive aged [x, x+1) at the start of the step (born agents only).
    """

    def __init__(self, max_age=100, **kwargs):
        kwargs.setdefault('name', 'age_sex_mx_analyzer')
        super().__init__(**kwargs)
        self.max_age = max_age
        return

    def init_results(self):
        super().init_results()
        shape = (self.t.npts, self.max_age + 1)
        self.define_results(
            ss.Result("exposure_male", dtype=float, shape=shape, label="Exposure (male) by age"),
            ss.Result("exposure_female", dtype=float, shape=shape, label="Exposure (female) by age"),
            ss.Result("deaths_male", dtype=int, shape=shape, label="Deaths (male) by age"),
            ss.Result("deaths_female", dtype=int, shape=shape, label="Deaths (female) by age"),
        )
        return

    def start_step(self):
        """ Exposure by age and sex at the start of the step """
        super().start_step()
        ppl = self.sim.people
        uids = ppl.auids[ppl.age[ppl.auids] >= 0] # Not the unborn
        fem = ppl.female[uids]
        ages = ppl.age[uids]
        self.results.exposure_female[self.ti, :] = by_age(ages[fem], self.max_age)
        self.results.exposure_male[self.ti, :] = by_age(ages[~fem], self.max_age)
        return

    def step(self):
        """ Deaths by age and sex in the step """
        ppl = self.sim.people
        dead = new_deaths(self.sim)
        fem = ppl.female[dead]
        ages = ppl.age[dead]
        self.results.deaths_female[self.ti, :] = by_age(ages[fem], self.max_age)
        self.results.deaths_male[self.ti, :] = by_age(ages[~fem], self.max_age)
        return

    def to_mx_df(self, *, year=None):
        """ The realized m(x) by age and sex at the time point nearest `year` (default: the last) """
        years = np.asarray(self.t.yearvec, dtype=float)
        ti = len(years) - 1 if year is None else int(np.argmin(np.abs(years - float(year))))
        ages = np.arange(self.max_age + 1)

        def make(sex):
            deaths = np.asarray(self.results[f'deaths_{sex.lower()}'][ti, :], dtype=float)
            expo = np.asarray(self.results[f'exposure_{sex.lower()}'][ti, :], dtype=float)
            mx = np.divide(deaths, expo, out=np.full_like(deaths, np.nan), where=expo > 0)/self.t.dt_year # NaN with no exposure
            return pd.DataFrame({"age": ages, "sex": sex, "mx": mx, "deaths": deaths, "exposure": expo})

        out = pd.concat([make("Female"), make("Male")], ignore_index=True)
        out["year"] = float(years[ti])
        return out


class SurvivorshipAnalyzer(ss.Analyzer):
    """ Survivorship l(x) at the end of the run: the share of each sex alive and aged at least x """

    def __init__(self, max_age=100, **kwargs):
        kwargs.setdefault('name', 'survivorship_analyzer')
        super().__init__(**kwargs)
        self.max_age = max_age
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result("lx_male", shape=self.max_age + 1, dtype=float, label="Male survivorship l(x)"),
            ss.Result("lx_female", shape=self.max_age + 1, dtype=float, label="Female survivorship l(x)"),
        )
        return

    def step(self):
        pass

    def finalize(self):
        """ l(x) at the end of the run, of everyone who was ever in it """
        super().finalize()
        ppl = self.sim.people
        n = ppl.uid.len_used
        female = ppl.female.raw[:n]
        alive = ppl.alive.raw[:n]
        age = ppl.age.raw[:n]
        for sex, mask in (('female', female), ('male', ~female)):
            n0 = max(np.count_nonzero(mask), 1)
            counts = by_age(age[mask & alive & (age >= 0)], self.max_age)
            self.results[f'lx_{sex}'][:] = np.cumsum(counts[::-1])[::-1]/n0 # Alive and at least x years old
        return

    def to_df(self):
        year = int(self.t.yearvec[-1])
        return pd.DataFrame([dict(year=year, age=age, sex=sex, survival=float(val))
                             for sex, key in (("Male", "lx_male"), ("Female", "lx_female")) for age, val in enumerate(self.results[key])])


class ConditionAtDeathAnalyzer(ss.Analyzer):
    """
    Each death: age, sex, years of life lost (YLL), whether the person had each condition and died of it, and whether they had HIV

    Args:
        conditions (list): the conditions to record (HIV is recorded as `hiv_positive`)
        ex_life_expectancy (float/df/func): the remaining life expectancy at death, for the YLL: a table of e(x) by age and sex (columns age, sex, ex), a function of sex and age, or a number, the age to which life is lost (70 for men and 75 for women if not a number)
    """

    def __init__(self, conditions=None, ex_life_expectancy=80.0, **kwargs):
        kwargs.setdefault('name', 'condition_at_death_analyzer')
        super().__init__(**kwargs)
        self.conditions = [c.lower() for c in (conditions or []) if c.lower() != "hiv"]
        self.ex_life_expectancy = ex_life_expectancy
        self.records = []
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        ex = self.ex_life_expectancy
        self._ex_lookup = ex if callable(ex) else make_ex_lookup(ex) if isinstance(ex, pd.DataFrame) else None
        return

    def yll(self, sex, age):
        if self._ex_lookup is not None:
            return max(0.0, float(self._ex_lookup(sex, age)))
        ref_age = float(self.ex_life_expectancy) if isinstance(self.ex_life_expectancy, (int, float)) else (75.0 if sex == "Female" else 70.0)
        return max(0.0, ref_age - age)

    def step(self):
        sim = self.sim
        ppl = sim.people
        dead = new_deaths(sim)
        if not len(dead):
            return
        year = float(self.t.yearvec[self.ti])
        hiv = sim.diseases.get('hiv')
        mods = {cond: resolve_disease_module(sim.diseases, cond)[0] for cond in self.conditions}
        causes = cause_names(ppl, dead)
        for i, uid in enumerate(dead):
            age = float(ppl.age[uid])
            sex = "Female" if ppl.female[uid] else "Male"
            rec = dict(uid=int(uid), year=year, age=age, sex=sex, yll=self.yll(sex, age))
            for cond, mod in mods.items():
                rec[f"died_{cond}"] = bool(mod.active[uid]) if mod is not None else False # Had the condition at death
                rec[f"cause_{cond}"] = mod is not None and causes[i] == mod.name # Died of it
            rec["hiv_positive"] = bool(hiv.infected[uid]) if hiv is not None else False
            died_of = [c for c in self.conditions if rec[f"cause_{c}"]]
            rec["primary_cause"] = died_of[0] if died_of else None
            self.records.append(rec)
        return

    def to_df(self):
        """ The deaths recorded, one row each """
        return pd.DataFrame(self.records)


def cause_names(ppl, uids):
    """ The cause of death of each agent: the name of the module that killed them, or 'unknown' """
    codes = np.asarray(ppl.cause_of_death[uids], dtype=int)
    names = [str(c) if c is not None else 'unknown' for c in ppl.causes] + ['unknown']
    return [names[c] for c in codes] # -1 (none) is the last


class CauseOfDeathYLLAnalyzer(ss.Analyzer):
    """
    Each death: age, sex, cause, and years of life lost (YLL) against a reference life table

    The cause is the module whose death request it was (`people.cause_of_death`): e.g. the name of a
    condition, of `ss.Deaths` ('deaths'), or of the background or residual of `mi.mortality_additive` and
    `mi.mortality_competing`; 'unknown' for a death without one (e.g. STIsim's HIV deaths).

    Args:
        reference_ex (df/func): a table of e(x) by age and sex (columns age, sex, ex), or a function of sex and age
        max_age (int): the oldest age
    """

    def __init__(self, reference_ex, *, max_age=100, **kwargs):
        kwargs.setdefault('name', 'cause_of_death_yll_analyzer')
        super().__init__(**kwargs)
        self.max_age = int(max_age)
        self.reference_ex = reference_ex
        self.records = []
        if not (callable(reference_ex) or isinstance(reference_ex, pd.DataFrame)):
            raise ValueError("reference_ex must be a callable or a DataFrame with columns: age, sex, ex")
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        ex = self.reference_ex
        self._ex_lookup = ex if callable(ex) else make_ex_lookup(ex)
        return

    def step(self):
        ppl = self.sim.people
        dead = new_deaths(self.sim)
        year = float(self.t.yearvec[self.ti])
        for uid, cause in zip(dead, cause_names(ppl, dead)):
            age = float(ppl.age[uid])
            sex = "Female" if ppl.female[uid] else "Male"
            self.records.append(dict(uid=int(uid), year=year, age=age, sex=sex, cause=cause, yll=max(0.0, float(self._ex_lookup(sex, age)))))
        return

    def to_df(self):
        """ The deaths recorded, one row each """
        return pd.DataFrame(self.records)
