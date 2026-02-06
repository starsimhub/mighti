"""
Module defining neonatal encephalopathy as an acute disease model.
"""

from mighti.diseases.base_disease import NonAcquiredDisease


class NeonatalEncephalopathy(NonAcquiredDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'NeonatalEncephalopathy'
        super().__init__(csv_path, pars, is_neonatal=True, **kwargs)
        self.define_pars(label='NeonatalEncephalopathy')
        return
    