"""
Module defining neonatal sepsis as an acute disease model.
"""

from mighti.diseases.base_disease import AcuteDisease


class NeonatalSepsis(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'NeonatalSepsis'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label='NeonatalSepsis')
        return
    