"""
Module defining the Type 2 Diabetes remitting disease model.
"""


import starsim as ss
from mighti.diseases.base_disease import RemittingDisease
from mighti.interventions.ncd_treatment import ReduceMortalityTx


class Type2Diabetes(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Type2Diabetes'
        super().__init__(csv_path, pars, **kwargs)

        self.define_pars(label='type2diabetes')
        return


class T2D_ReduceMortalityTx(ReduceMortalityTx):
    """Back-compat wrapper; Paper 2 uses ``ReduceMortalityTx(disease='type2diabetes')``."""

    def __init__(self, *args, product=None, rel_death_reduction=0.5, eligibility=None, **kwargs):
        kwargs.setdefault("disease", "type2diabetes")
        kwargs.setdefault("rel_death_reduction", rel_death_reduction)
        if eligibility is not None and "eligibility_mode" not in kwargs:
            kwargs["eligibility_mode"] = "affected"
        super().__init__(*args, product=product, **kwargs)
