"""
The base classes of MIGHTI's health conditions, with their parameters read from a CSV file

Each condition acquired over life is declared from states and transitions (see `ss.Disease`):
people acquire it at an annual probability by sex (`p_acquire_female`, `p_acquire_male`, times
`p_acquire_multiplier`), die of it at an annual probability (`p_death`), and, for a remitting
condition, remit at an annual probability (`remission_rate`). Starsim converts the annual
probabilities to the timestep (`ss.probperyear`). Other conditions change acquisition with
`ss.modify()` of the condition's `rel_sus` (see `mighti.interactions`), and treatments change its
transitions (e.g. `ss.Treatment(efficacy=dict(death=0.5))`). Conditions present from birth are
written as code (`NonAcquiredDisease`).
"""

import numpy as np
import pandas as pd
import starsim as ss


__all__ = ['get_disease_parameters', 'ChronicDisease', 'AcuteDisease', 'RemittingDisease', 'GenericSIS', 'GenericSIR',
           'AcuteSurgicalDisease', 'NonAcquiredDisease', 'StaticCondition']


defaults = dict(p_death=0.0001, remission_rate=0.0, affected_sex='both', p_acquire=0.01) # For a value missing from the CSV file


def get_disease_parameters(csv_path, disease_name):
    """
    Load the parameters of one condition from a CSV file with a row per condition (column `condition`)

    The probabilities (`p_death`, `remission_rate`, `p_acquire`, `p_acquire_female`, `p_acquire_male`) are
    annual; a missing value takes the default in `defaults` (and `p_acquire_female` and `p_acquire_male`
    take `p_acquire`). The other columns (e.g. `rel_sus`, read by `mighti.interactions`) are not read here.

    Args:
        csv_path (str): path to the parameter CSV file
        disease_name (str): name of the condition, as in the `condition` column

    Returns:
        dict: `p_death`, `remission_rate`, `affected_sex`, `p_acquire_female`, `p_acquire_male`, and those of `AcuteSurgicalDisease` if given
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if 'condition' not in df.columns:
        raise KeyError(f"'condition' column missing in {csv_path}. Available columns: {df.columns.tolist()}")
    row = df[df['condition'] == disease_name]
    if row.empty:
        raise ValueError(f"Disease '{disease_name}' not found in parameter file: {csv_path}")
    row = row.iloc[0]

    def get(field, default):
        val = row.get(field, np.nan)
        if field != 'affected_sex':
            val = pd.to_numeric(val, errors='coerce')
        return default if pd.isna(val) else val

    p_acquire = get('p_acquire', defaults['p_acquire'])
    out = dict(
        p_death = get('p_death', defaults['p_death']),
        remission_rate = get('remission_rate', defaults['remission_rate']),
        affected_sex = str(get('affected_sex', defaults['affected_sex'])).strip().lower(),
        p_acquire_female = get('p_acquire_female', p_acquire),
        p_acquire_male = get('p_acquire_male', p_acquire),
    )
    for field in ['p_surgery', 'rel_mortality_treated', 'rel_mortality_untreated']: # AcuteSurgicalDisease
        val = get(field, None)
        if val is not None:
            out[field] = val
    return out


def annual(value):
    """ An annual probability, given as a number, an `ss.bernoulli` (Starsim 3's form), or a typed probability """
    if isinstance(value, ss.bernoulli):
        value = value.pars.p
    return value if isinstance(value, (ss.TimePar, ss.par)) else ss.probperyear(float(value))


def hazard(p):
    """ The annual hazard of an annual probability (a number or a typed probability, or a parameter of one) """
    p = getattr(p, 'raw', p)
    p = getattr(p, 'value', p)
    return -np.log1p(-np.minimum(float(p), 1 - 1e-12))


class Condition(ss.Disease):
    """
    A condition acquired over life, with its parameters read from a CSV file (see `get_disease_parameters()`)

    The name of the condition in the file is the class's name (e.g. `Type2Diabetes`), or `disease_name`.
    Parameters given in `pars` or as keywords override the file's; `init_prev` (a share, or an `ss.bernoulli`
    whose `p` may be a function of the sim and the agents, e.g. `mi.age_sex_dependent_prevalence`) is the
    share of people affected at the start, and `affected_sex` ('both', 'female', or 'male') says who acquires it.

    Args:
        csv_path (str): path to the parameter CSV file
        pars (dict): parameters, e.g. `dict(init_prev=ss.bernoulli(0.1), p_death=0.02)`
        kwargs (dict): parameters, as for `pars`, or `name`/`label`
    """
    disease_name = None # The condition's name in the CSV file, if not the class's name
    annual_pars = ['p_acquire_female', 'p_acquire_male', 'p_death', 'remission_rate'] # Annual probabilities

    def __init__(self, csv_path=None, pars=None, **kwargs):
        self.disease_name = self.disease_name or type(self).__name__
        values = get_disease_parameters(csv_path, self.disease_name) if csv_path is not None else {}
        values.update(pars or {})
        values.update(kwargs)
        self.affected_sex = str(values.pop('affected_sex', 'both')).lower()
        known = self.declared_pars() + ['init_prev', 'init', 'init_n', 'name', 'label']
        if csv_path is not None: # Columns of the file for another kind of condition
            for key in ['remission_rate', 'p_surgery', 'rel_mortality_treated', 'rel_mortality_untreated']:
                if key not in known and key not in (pars or {}) and key not in kwargs:
                    values.pop(key, None)
        for key in self.annual_pars:
            if key in values:
                values[key] = annual(values[key])
        if self.affected_sex in ['male', 'female']: # Only one sex acquires it
            other = 'male' if self.affected_sex == 'female' else 'female'
            values[f'p_acquire_{other}'] = ss.probperyear(0.0)
        super().__init__(**values)
        return

    @classmethod
    def declared_pars(cls):
        """ The names of the parameters declared by the class """
        return [k for k,v in ss.transitions.class_members(cls).items() if isinstance(v, ss.par)]

    @property
    def active(self):
        """ The state of having the condition: `affected` (or `infected` for a generic infection) """
        return self.affected

    def death_hazard(self, uids):
        """ The annual hazard of death from the condition of each of these agents (0 for those it cannot kill now), with the modifiers of its deaths (e.g. a treatment's efficacy), for `mi.mortality_competing.CompetingRisksDeaths` """
        h = np.zeros(len(uids))
        for tr in self.transitions.values():
            if tr.outflow and not tr.migration: # Its deaths: death, and death_treated for AcuteSurgicalDisease
                at_risk = np.logical_or.reduce([self[state.name][uids] for state in tr.src])
                h += np.where(at_risk, tr.hazard(uids), 0.0)
        return h/ss.dur(1, self.sim.unit).years # Per unit of the sim's time (e.g. months, with STIsim's HIV) to per year

    @property
    def duration(self):
        """ Years since onset of the current episode, by UID (0 for those without the condition) """
        dur = np.zeros(self.sim.people.uid.len_used)
        uids = self.active.uids
        dur[uids] = (self.ti - self[f'ti_{self.active.name}'][uids])*self.t.dt_year # The time of onset is kept (keep_time=True)
        return dur


class ChronicDisease(Condition):
    """ A chronic condition: acquired at an annual probability by sex, with excess mortality; no remission """
    p_acquire_female = ss.par(ss.probperyear(defaults['p_acquire']), doc='Annual probability that a woman acquires the condition')
    p_acquire_male = ss.par(ss.probperyear(defaults['p_acquire']), doc='Annual probability that a man acquires the condition')
    p_acquire_multiplier = ss.par(1.0, doc='Multiplier on acquisition, e.g. for calibration')
    p_death = ss.par(ss.probperyear(defaults['p_death']), doc='Annual probability of death from the condition')
    susceptible = ss.State()
    affected = ss.State(keep_time=True) # Keeps the time of onset (ti_affected), for duration
    onset = ss.Transition(susceptible, affected, rate=p_acquire_multiplier*ss.by(sex=dict(female=p_acquire_female, male=p_acquire_male)))
    death = ss.death(affected, rate=p_death)
    prevalence = affected/(susceptible + affected)


class AcuteDisease(ChronicDisease):
    """ An acute condition: in MIGHTI, the same course as a chronic one (no remission) """
    pass


class RemittingDisease(Condition):
    """ A remitting condition: as a chronic one, and those affected remit (become susceptible again) at an annual probability """
    p_acquire_female = ss.par(ss.probperyear(defaults['p_acquire']), doc='Annual probability that a woman acquires the condition')
    p_acquire_male = ss.par(ss.probperyear(defaults['p_acquire']), doc='Annual probability that a man acquires the condition')
    p_acquire_multiplier = ss.par(1.0, doc='Multiplier on acquisition, e.g. for calibration')
    p_death = ss.par(ss.probperyear(defaults['p_death']), doc='Annual probability of death from the condition')
    remission_rate = ss.par(ss.probperyear(defaults['remission_rate']), doc='Annual probability of remission')
    susceptible = ss.State()
    affected = ss.State(keep_time=True)
    onset = ss.Transition(susceptible, affected, rate=p_acquire_multiplier*ss.by(sex=dict(female=p_acquire_female, male=p_acquire_male)))
    remission = ss.Transition(affected, susceptible, rate=remission_rate)
    death = ss.death(affected, rate=p_death)
    prevalence = affected/(susceptible + affected)


class GenericSIS(Condition):
    """ An infection acquired at an annual probability by sex (not by contact), from which people recover to susceptible, with excess mortality """
    p_acquire_female = ss.par(ss.probperyear(defaults['p_acquire']), doc='Annual probability that a woman is infected')
    p_acquire_male = ss.par(ss.probperyear(defaults['p_acquire']), doc='Annual probability that a man is infected')
    p_acquire_multiplier = ss.par(1.0, doc='Multiplier on acquisition, e.g. for calibration')
    p_death = ss.par(ss.probperyear(defaults['p_death']), doc='Annual probability of death from the infection')
    remission_rate = ss.par(ss.probperyear(defaults['remission_rate']), doc='Annual probability of recovery')
    susceptible = ss.State()
    infected = ss.State(keep_time=True)
    onset = ss.Transition(susceptible, infected, rate=p_acquire_multiplier*ss.by(sex=dict(female=p_acquire_female, male=p_acquire_male)))
    recovery = ss.Transition(infected, susceptible, rate=remission_rate)
    death = ss.death(infected, rate=p_death)
    prevalence = infected/(susceptible + infected)

    @property
    def active(self):
        return self.infected


class GenericSIR(Condition):
    """ As `GenericSIS`, but people recover to immunity (`recovered`) """
    p_acquire_female = ss.par(ss.probperyear(defaults['p_acquire']), doc='Annual probability that a woman is infected')
    p_acquire_male = ss.par(ss.probperyear(defaults['p_acquire']), doc='Annual probability that a man is infected')
    p_acquire_multiplier = ss.par(1.0, doc='Multiplier on acquisition, e.g. for calibration')
    p_death = ss.par(ss.probperyear(defaults['p_death']), doc='Annual probability of death from the infection')
    remission_rate = ss.par(ss.probperyear(defaults['remission_rate']), doc='Annual probability of recovery')
    susceptible = ss.State()
    infected = ss.State(keep_time=True)
    recovered = ss.State()
    onset = ss.Transition(susceptible, infected, rate=p_acquire_multiplier*ss.by(sex=dict(female=p_acquire_female, male=p_acquire_male)))
    recovery = ss.Transition(infected, recovered, rate=remission_rate)
    death = ss.death(infected, rate=p_death)
    prevalence = infected/(susceptible + infected + recovered)

    @property
    def active(self):
        return self.infected


class AcuteSurgicalDisease(Condition):
    """
    An acute condition that surgery can treat: those affected have surgery at an annual probability (`p_surgery`),
    after which their mortality is `rel_mortality_treated` times `p_death`, rather than `rel_mortality_untreated` times it
    """
    annual_pars = Condition.annual_pars + ['p_surgery']
    p_acquire_female = ss.par(ss.probperyear(defaults['p_acquire']), doc='Annual probability that a woman acquires the condition')
    p_acquire_male = ss.par(ss.probperyear(defaults['p_acquire']), doc='Annual probability that a man acquires the condition')
    p_acquire_multiplier = ss.par(1.0, doc='Multiplier on acquisition, e.g. for calibration')
    p_death = ss.par(ss.probperyear(defaults['p_death']), doc='Annual probability of death from the condition, before the relative mortality')
    p_surgery = ss.par(ss.probperyear(0.3), doc='Annual probability of surgery for those affected')
    rel_mortality_untreated = ss.par(2.0, doc='Relative mortality before surgery')
    rel_mortality_treated = ss.par(0.5, doc='Relative mortality after surgery')
    susceptible = ss.State()
    affected = ss.State(keep_time=True)
    treated = ss.State()
    onset = ss.Transition(susceptible, affected, rate=p_acquire_multiplier*ss.by(sex=dict(female=p_acquire_female, male=p_acquire_male)))
    surgery = ss.Transition(affected, treated, rate=p_surgery)
    death = ss.death(affected, rate=rel_mortality_untreated*p_death)
    death_treated = ss.death(treated, rate=rel_mortality_treated*p_death)
    prevalence = (affected + treated)/(susceptible + affected + treated)


class NonAcquiredDisease(ss.Disease):
    """
    A condition present from birth (congenital or neonatal), written as code

    At the start, and at each birth (from `ss.Pregnancy`), each person has the condition with probability
    `init_prev` (for a neonatal condition, only those under 28 days old at the start). Those affected die of it
    at an annual probability `p_death` (for a neonatal condition, only while under 28 days old).

    Args:
        csv_path (str): path to the parameter CSV file (see `get_disease_parameters()`)
        pars (dict): parameters, e.g. `dict(init_prev=ss.bernoulli(0.01))`
        is_neonatal (bool): whether the condition, and its mortality, is limited to the first 28 days of life
    """
    disease_name = None
    is_neonatal = False

    def __init__(self, csv_path=None, pars=None, is_neonatal=None, **kwargs):
        super().__init__()
        self.disease_name = self.disease_name or type(self).__name__
        if is_neonatal is not None:
            self.is_neonatal = is_neonatal
        p = get_disease_parameters(csv_path, self.disease_name) if csv_path is not None else defaults
        self.define_pars(
            p_death = ss.bernoulli(p=ss.probperyear(p['p_death'])),
            init_prev = ss.bernoulli(p=0.01),
        )
        pars = {**(pars or {}), **kwargs}
        if 'p_death' in pars:
            pars['p_death'] = ss.bernoulli(p=annual(pars['p_death']))
        self.update_pars(pars)
        self.define_states(
            ss.BoolState('affected', label='Affected'),
            ss.FloatArr('ti_affected', label='Time of becoming affected'),
        )
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        preg = sim.get_module(ss.Pregnancy, die=False)
        if preg is not None:
            preg.add_callback(preg.birth, self.on_birth)
        return

    def init_post(self):
        super().init_post()
        uids = self.sim.people.auids
        if self.is_neonatal:
            uids = uids[self.sim.people.age[uids] < 28/365]
        self.on_birth(uids)
        return

    def on_birth(self, uids):
        """ Each newborn (or person at the start) has the condition with probability init_prev """
        affected = self.pars.init_prev.filter(ss.uids(uids))
        self.affected[affected] = True
        self.ti_affected[affected] = self.ti
        return

    def step(self):
        """ Deaths among those affected (for a neonatal condition, those under 28 days old) """
        uids = self.affected.uids
        if self.is_neonatal:
            uids = uids[self.sim.people.age[uids] < 28/365]
        deaths = self.pars.p_death.filter(uids)
        self.sim.people.request_death(deaths, cause=self)
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('prevalence', dtype=float, scale=False, label='Prevalence'),
            ss.Result('death', dtype=int, scale=True, label='Deaths'),
        )
        return

    def update_results(self):
        super().update_results()
        ti = self.ti
        self.results.death[ti] = len(self.sim.people.died_of(self)) # Once each death has one cause
        self.results.prevalence[ti] = np.count_nonzero(self.affected)/max(self.sim.people.n_alive, 1)
        return

    @property
    def active(self):
        return self.affected

    def death_hazard(self, uids):
        """ The annual hazard of death from the condition of each of these agents, for `mi.mortality_competing.CompetingRisksDeaths` """
        at_risk = self.affected[uids]
        if self.is_neonatal:
            at_risk &= self.sim.people.age[uids] < 28/365
        return np.where(at_risk, hazard(self.pars.p_death.pars.p), 0.0)

    @property
    def duration(self):
        """ Years since onset (birth, or the start), by UID (0 for those without the condition) """
        dur = np.zeros(self.sim.people.uid.len_used)
        uids = self.affected.uids
        dur[uids] = (self.ti - self.ti_affected[uids])*self.t.dt_year
        return np.nan_to_num(dur)


class StaticCondition(NonAcquiredDisease):
    """ A lifelong condition present from birth (e.g. Down syndrome), at all ages """
    is_neonatal = False
