"""
Module defining digestive congenital anomalies as an acute surgical disease model.
"""

from mighti.diseases.base_disease import AcuteSurgicalDisease


class DigestiveCongenitalAnomalies(AcuteSurgicalDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'DigestiveCongenitalAnomalies'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label='DigestiveCongenitalAnomalies')
        return
    