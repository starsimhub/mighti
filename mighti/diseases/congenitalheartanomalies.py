"""
Module defining congenital heart anomalies as an acute surgical disease model.
"""

from mighti.diseases.base_disease import NonAcquiredDisease


class CongenitalHeartAnomalies(NonAcquiredDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'CongenitalHeartAnomalies'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label='CongenitalHeartAnomalies')
        return
    