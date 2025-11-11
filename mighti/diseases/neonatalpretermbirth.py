"""
Module defining neonatal preterm birth as an acute disease model.
"""

from mighti.diseases.base_disease import NonAcquiredDisease


class NeonatalPretermBirth(NonAcquiredDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'NeonatalPretermBirth'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label='NeonatalPretermBirth')
        return
    