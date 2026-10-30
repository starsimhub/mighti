"""
Interventions: ART for people with co-occurring conditions, and housing support
"""

import numpy as np
import starsim as ss
import stisim as sti

__all__ = ["ARTwithCASM", "HousingSupport", "GiveHousingToDepressed", "GiveHousingSupport", "HousingSupportForAUD"]


class ARTwithCASM(sti.ART):
    """
    STIsim's ART, for use with MIGHTI's adherence modules (see `mi.interventions.adherence`)

    Args:
        casm_sensitivity (str): how the effect of co-occurring conditions (CASM) is interpreted: 'pharma' (pharmacotherapy)
        kwargs (dict): passed to `sti.ART`, e.g. `coverage=`
    """
    def __init__(self, *args, casm_sensitivity='pharma', **kwargs):
        super().__init__(*args, **kwargs)
        self.casm_sensitivity = casm_sensitivity
        return


class HousingSupport(ss.Intervention):
    """
    Give stable housing (`people.neighbourhood_situation`) to a share of people without it, each step from `start`

    Args:
        coverage (float): the share of those eligible housed in each step
        start (float/date): when it starts (default: the start of the sim)
        label (str): the label
    """
    def __init__(self, coverage=0.5, start=None, label=None, **kwargs):
        super().__init__(label=label, **kwargs)
        self.define_pars(p_house=ss.bernoulli(p=coverage))
        self.start = start
        return

    def eligible(self):
        """ Who is eligible: people without stable housing """
        return ~self.sim.people.neighbourhood_situation

    def step(self):
        if self.start is not None and self.now < ss.date(self.start):
            return
        housed = self.pars.p_house.filter(self.eligible().uids)
        self.sim.people.neighbourhood_situation[housed] = True
        return housed


class GiveHousingToDepressed(HousingSupport):
    """ Housing support for people with major depressive disorder who are unstably housed (see `HousingSupport`) """
    def eligible(self):
        return super().eligible() & self.sim.diseases.majordepressivedisorder.affected


class GiveHousingSupport(HousingSupport):
    """ Housing support for adults (15 and over) who are unstably housed (see `HousingSupport`) """
    def eligible(self):
        return super().eligible() & (self.sim.people.age >= 15)


class HousingSupportForAUD(HousingSupport):
    """ Housing support for adults (15 and over) with alcohol use disorder who are unstably housed, from 2010 by default (see `HousingSupport`) """
    def __init__(self, coverage=0.5, start=2010, label=None, **kwargs):
        super().__init__(coverage=coverage, start=start, label=label, **kwargs)
        return

    def eligible(self):
        return super().eligible() & self.sim.diseases.alcoholusedisorder.affected & (self.sim.people.age >= 15)
