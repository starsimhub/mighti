"""
Module defining the Alcohol Use Disorder remitting disease model.
"""


from mighti.diseases.base_disease import RemittingDisease


class AlcoholUseDisorder(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'AlcoholUseDisorder'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'AlcoholUseDisorder')
        if not hasattr(self.pars, 'p_acquire_multiplier'):
            self.pars.p_acquire_multiplier = 0.001
        return

