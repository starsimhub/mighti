"""
Module defining neonatal jaundice as an acute disease model.
"""

from mighti.diseases.base_disease import AcuteDisease


class NeonatalJaundice(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'NeonatalJaundice'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label='NeonatalJaundice')
        return
    