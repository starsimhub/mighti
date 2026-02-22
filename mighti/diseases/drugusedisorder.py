"""
Module defining drug use disorder as a remitting disease model.
Drug use disorders include: Opioid, Cocaine, Amphetamines, Cannabis and other substances.
"""


from mighti.diseases.base_disease import RemittingDisease


class DrugUseDisorder(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'DrugUseDisorder'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label = 'DrugUseDisorder')

        return

