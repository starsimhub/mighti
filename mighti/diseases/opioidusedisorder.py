"""
Module defining opioid use disorder as a remitting disease model.
"""


from mighti.diseases.base_disease import RemittingDisease


class OpioidUseDisorder(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'OpioidUseDisorder'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label = 'OpioidUseDisorder')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 1
        return
