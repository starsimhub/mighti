"""
Competing-risks mortality: all-cause mortality as observed, with the modeled causes as part of it

With an observed all-cause m(x) schedule (e.g. UN/WPP) as the background (`ss.Deaths`), and diseases that
also kill, deaths from the modeled causes would be counted twice. `CompetingRisksDeaths` instead kills at
the rest of the all-cause hazard: the all-cause hazard less that of each modeled cause (not below 0), so that
the total is the observed all-cause mortality while the modeled causes are below it. Each death has one
cause (`people.cause_of_death`): a modeled cause, or this module's name ('residual').

The hazard of each modeled cause is that of the conditions of `mighti.diseases` (their `death_hazard()`), and
for a declared disease (e.g. STIsim's HIV), the rates of its transitions to death (see `death_hazard()`).
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['CompetingRisksDeaths', 'death_hazard']


def death_hazard(disease, uids):
    """
    The annual hazard of death from a disease of each agent: its own `death_hazard()` (MIGHTI's conditions), or for a
    declared disease, the rates of its transitions to death (e.g. HIV's `hiv_death` and `art_death`), for those in their
    source states. A death at a set time (e.g. HIV's `aids_death`, at the end of a stage) has no hazard to count.
    """
    uids = ss.uids(uids)
    if hasattr(disease, 'death_hazard'):
        return disease.death_hazard(uids)
    h = np.zeros(len(uids))
    graph = getattr(disease, 'graph', None)
    if graph is None: # A disease written as code with no death_hazard(): none modeled
        return h
    for tr in graph.transitions.values():
        if (tr.outflow or tr.dies) and tr.rate is not None:
            inside = np.zeros(len(uids), dtype=bool)
            for state in tr.src:
                inside |= np.asarray(disease[state.name][uids], dtype=bool)
            h[inside] += tr.hazard(uids[inside])
    return h/disease.sim.unit_base.years # Per unit of the sim's time, to per year


class CompetingRisksDeaths(ss.Deaths):
    """
    Deaths at the all-cause hazard less the hazards of the modeled causes (the residual)

    Args:
        death_rate: the all-cause death rates, as for `ss.Deaths` (typically UN/WPP mx by year, sex, and age)
        rel_death (float): multiplier on the all-cause rates
        rate_units (float): as for `ss.Deaths` (1 if the rates are per person)
        metadata (dict): as for `ss.Deaths`; sexes may be given as 'Female'/'Male' or 'f'/'m'
        name (str): the name of the module, which is the cause of the deaths it causes (default 'residual')

    Unlike MIGHTI's Starsim 3 version, a modeled cause whose hazard is above the all-cause hazard for someone
    is not scaled down to it: the residual is then 0, and that person's mortality is the modeled cause's.
    """
    def __init__(self, death_rate=None, *, rel_death=1.0, rate_units=1e-3, metadata=None, name='residual', **kwargs):
        metadata = sc.mergedicts(dict(sex_keys={'Female':'f', 'Male':'m', 'f':'f', 'm':'m'}), metadata)
        super().__init__(death_rate=death_rate, rel_death=rel_death, rate_units=rate_units, metadata=metadata, name=name, **kwargs)
        return

    def modeled_hazard(self, uids):
        """ The total annual hazard of death of the modeled causes for each agent """
        h = np.zeros(len(uids))
        for dis in self.sim.diseases():
            h += death_hazard(dis, uids)
        return h

    def make_p_death(self):
        """ The probability of death of each agent in this step from the residual hazard """
        p_all = super().make_p_death()
        uids = self.sim.people.auids
        p_all = np.broadcast_to(np.asarray(p_all, dtype=float), uids.shape)
        h_all = -np.log1p(-np.clip(p_all, 0, 1 - 1e-12))
        h_mod = self.modeled_hazard(uids)*self.t.dt_year
        h_res = np.maximum(h_all - h_mod, 0)
        self.residual_share = np.mean(sc.safedivide(h_res, h_all, default=0)) if len(uids) else 0
        return -np.expm1(-h_res)

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('residual_share', dtype=float, scale=False, label='Mean residual share of the death hazard', auto_plot=False))
        return

    def update_results(self):
        super().update_results()
        self.results.residual_share[self.ti] = getattr(self, 'residual_share', np.nan)
        return
