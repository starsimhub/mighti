"""
Adherence to ART, as affected by co-occurring conditions (CASM) and social determinants of health (SDoH)

1. `AdherenceEngine`: each person's adherence (`people.adherence`), from their conditions and SDoH
2. `ARTAdherenceDisruptor`: people with low adherence stop ART (a modifier of HIV's `rate_art_dropout`)
3. `InterventionAdherenceDisruptor`: ART is less effective when mean adherence is low (a modifier of HIV's `effective_art_efficacy`)
"""

import numpy as np
import starsim as ss

__all__ = [
    "AdherenceEngine",
    "ARTAdherenceDisruptor",
    "InterventionAdherenceDisruptor",
    "BASELINE_ADHERENCE_PHARMACOTHERAPY",
    "CASM_NONADHERENCE_OR",
    "CASM_REL_FACTORS",
    "SDOH_REL_FACTORS",
]

# Baseline adherence to pharmacotherapy (HRMM)
BASELINE_ADHERENCE_PHARMACOTHERAPY = 0.62

# Odds ratios of non-adherence with each co-occurring condition (Table S2)
CASM_NONADHERENCE_OR = {
    "AlcoholUseDisorder": 1.41,
    "MajorDepressiveDisorder": 2.21,
    "AnxietyDisorder": 2.04,
    "ChronicPain": 1.34,
    "TobaccoUse": 1.18,
    "OpioidUseDisorder": 1.18,
    "StimulantUseDisorder": 1.81,
}

# Multipliers on adherence with each condition (1/OR)
CASM_REL_FACTORS = {k: 1.0/float(v) for k, v in CASM_NONADHERENCE_OR.items()}

# Multipliers on adherence with each SDoH (by the name of the people's property)
SDOH_REL_FACTORS = {
    "neighbourhood_situation": 0.90,
    "social_context": 0.85,
    "education_situation": 0.92,
    "economic_situation": 0.80,
    "healthcare_system": 0.88,
}


def has_condition(sim, cond):
    """ Who has a condition (by its name, e.g. 'MajorDepressiveDisorder'), as a boolean array over the active agents; None if it is not in the sim """
    dis = sim.diseases.get(str(cond).lower())
    if dis is None:
        return None
    return np.asarray(dis.active.values if hasattr(dis, 'active') else dis.affected.values, dtype=bool)


class AdherenceEngine(ss.Module):
    """
    Each person's adherence to ART (`people.adherence`), from their conditions and social determinants of health

    By default, adherence starts at 1 and is multiplied by `casm_rel` for each condition a person has, and by `sdoh_rel` for
    each SDoH property they have. With `use_odds_model=True` (or `baseline_adherence` or `casm_nonadherence_or` given), it is
    instead the baseline adherence with the odds of non-adherence multiplied by each condition's odds ratio, then multiplied by `sdoh_rel`.

    Args:
        casm_rel (dict): the multiplier on adherence of each condition (default `CASM_REL_FACTORS`)
        sdoh_rel (dict): the multiplier on adherence of each SDoH property (default `SDOH_REL_FACTORS`)
        baseline_adherence (float): the baseline adherence of the odds model (default `BASELINE_ADHERENCE_PHARMACOTHERAPY`)
        casm_nonadherence_or (dict): the odds ratio of non-adherence of each condition (default `CASM_NONADHERENCE_OR`)
        use_odds_model (bool): whether to use the odds model
    """
    def __init__(self, casm_rel=None, sdoh_rel=None, *, baseline_adherence=None, casm_nonadherence_or=None, use_odds_model=False, name='adherence_engine', **kwargs):
        super().__init__(name=name, **kwargs)
        self.casm_rel = CASM_REL_FACTORS.copy() if casm_rel is None else casm_rel
        self.sdoh_rel = SDOH_REL_FACTORS.copy() if sdoh_rel is None else sdoh_rel
        self.baseline_adherence = baseline_adherence
        self.casm_nonadherence_or = casm_nonadherence_or
        self.use_odds_model = use_odds_model or baseline_adherence is not None or casm_nonadherence_or is not None
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        sim.people.add_state(ss.FloatArr('adherence', default=1.0))
        return

    def step(self):
        sim = self.sim
        ppl = sim.people
        n = len(ppl.auids)
        if self.use_odds_model:
            p0 = np.clip(BASELINE_ADHERENCE_PHARMACOTHERAPY if self.baseline_adherence is None else self.baseline_adherence, 1e-12, 1 - 1e-12)
            mult = np.ones(n)
            for cond, odds_ratio in (self.casm_nonadherence_or or CASM_NONADHERENCE_OR).items():
                affected = has_condition(sim, cond)
                if affected is not None:
                    mult[affected] *= odds_ratio
            odds = (1 - p0)/p0*mult # Of non-adherence
            adherence = 1/(1 + odds)
        else:
            adherence = np.ones(n)
            for cond, rel in self.casm_rel.items():
                affected = has_condition(sim, cond)
                if affected is not None:
                    adherence[affected] *= rel
        for key, rel in self.sdoh_rel.items():
            if key in ppl.states:
                adherence[np.asarray(ppl.states[key].values, dtype=bool)] *= rel
        ppl.adherence[ppl.auids] = np.clip(adherence, 0, 1)
        return


class ARTAdherenceDisruptor(ss.modify):
    """
    People on ART stop it at a monthly probability that rises as their adherence falls

        p_i = base_dropout*(1 - adherence_i) for people with alcohol use disorder, and base_dropout_noaud*(1 - adherence_i) for others

    Only people who have been on ART for at least `min_time_on_art` stop. It sets HIV's rate of stopping ART early
    (`rate_art_dropout`, the transition `art_dropout`) for each of them.

    Args:
        base_dropout (float): the monthly probability of stopping for people with alcohol use disorder and no adherence
        base_dropout_noaud (float): the same for people without alcohol use disorder
        min_time_on_art (dur): the time on ART before people can stop
    """
    def __init__(self, base_dropout=0.10, base_dropout_noaud=0.001, min_time_on_art=ss.months(6), name='adherence_art_dropout', **kwargs):
        super().__init__('hiv.pars.rate_art_dropout', set=self.dropout_rate, where=self.eligible, name=name, **kwargs)
        self.define_pars(
            base_dropout = float(base_dropout),
            base_dropout_noaud = float(base_dropout_noaud),
            min_time_on_art = min_time_on_art,
        )
        return

    def eligible(self, sim):
        """ Those who have been on ART for at least min_time_on_art """
        hiv = sim.diseases.hiv
        return hiv.on_art & ((hiv.ti - hiv.ti_art)*hiv.t.dt_year >= self.pars.min_time_on_art.years)

    def dropout_rate(self, sim):
        """ The probability of stopping ART of each person, per month """
        aud = sim.diseases.get('alcoholusedisorder')
        aud = aud.affected.values if aud is not None else np.zeros(len(sim.people), dtype=bool)
        base = np.where(aud, self.pars.base_dropout, self.pars.base_dropout_noaud)
        return ss.probpermonth(np.clip(base*(1 - sim.people.adherence.values), 0, 1))


class InterventionAdherenceDisruptor(ss.modify):
    """
    ART is less effective when adherence is low: HIV's `effective_art_efficacy` (the efficacy of suppressive ART against
    transmission) is multiplied by the mean adherence, and the `rel_effect` of any intervention that has one is its value
    at the start times adherence (the mean, for a scalar)
    """
    def __init__(self, name='intervention_adherence_disruptor', **kwargs):
        super().__init__('hiv.pars.effective_art_efficacy', multiply=self.mean_adherence, name=name, **kwargs)
        return

    def mean_adherence(self, sim):
        return float(sim.people.adherence.values.mean())

    def init_post(self):
        """ The values at the start """
        super().init_post()
        self.baseline_rel_effects = {intv.name: np.array(intv.rel_effect, dtype=float) for intv in self.sim.interventions() if hasattr(intv, 'rel_effect')}
        return

    def step(self):
        super().step()
        adherence = self.sim.people.adherence.values
        scale = float(adherence.mean())
        for intv in self.sim.interventions():
            baseline = self.baseline_rel_effects.get(intv.name)
            if baseline is not None:
                intv.rel_effect = baseline*scale if baseline.ndim == 0 else baseline*adherence
        return
