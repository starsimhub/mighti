"""
Additive-hazard mortality: background deaths, with each modeled cause of death adding its own hazard

Each disease module kills at its own rate and records itself as the cause (`people.request_death(uids,
cause=...)`, as Starsim's diseases do), so the hazard of death is the background hazard plus each
modeled cause's, and removing a modeled cause lowers all-cause mortality (as needed for a reference
life expectancy and for scenarios). Each death has one cause (`people.cause_of_death`, `people.causes`).

The background is an observed all-cause m(x) schedule (e.g. UN/WPP) times `background_multiplier`,
which can be calibrated so that the life expectancy with the modeled causes matches the observed one.
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['AdditiveHazardDeaths']


class AdditiveHazardDeaths(ss.Deaths):
    """
    Background deaths from an all-cause m(x) schedule, with the hazard scaled by `background_multiplier`

    Args:
        background_rate: the background death rates, as for `ss.Deaths(death_rate=...)` (typically UN/WPP mx by year, sex, and age)
        background_multiplier (float): multiplier on the background hazard (not the probability)
        multiplier_min_age (float): the age from which the multiplier applies; younger people have the unscaled schedule (e.g. to keep the infant rates as observed)
        rate_units (float): as for `ss.Deaths` (1 if the rates are per person)
        metadata (dict): as for `ss.Deaths`; sexes may be given as 'Female'/'Male' or 'f'/'m'
        name (str): the name of the module, which is the cause of the deaths it causes (default 'background')
    """
    def __init__(self, background_rate=None, *, background_multiplier=1.0, multiplier_min_age=0.0, rate_units=1e-3, metadata=None, name='background', **kwargs):
        metadata = sc.mergedicts(dict(sex_keys={'Female':'f', 'Male':'m', 'f':'f', 'm':'m'}), metadata)
        super().__init__(death_rate=background_rate, rate_units=rate_units, metadata=metadata, name=name, **kwargs)
        self.define_pars(
            background_multiplier = float(background_multiplier),
            multiplier_min_age = float(multiplier_min_age),
        )
        return

    def make_p_death(self):
        """ The probability of death of each agent in this step, with the background hazard scaled from multiplier_min_age """
        p = super().make_p_death()
        ppl = self.sim.people
        uids = ppl.auids
        p = np.broadcast_to(np.asarray(p, dtype=float), uids.shape)
        mult = np.where(ppl.age[uids] >= self.pars.multiplier_min_age, self.pars.background_multiplier, 1.0)
        return 1 - (1 - p)**mult # The hazard times the multiplier
