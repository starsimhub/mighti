"""
Module defining congenital musculoskeletal anomalies as an acute surgical disease model.
"""

from mighti.diseases.base_disease import AcuteSurgicalDisease


class CongenitalMusculoskeletal(AcuteSurgicalDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'CongenitalMusculoskeletal'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label='CongenitalMusculoskeletal')
        return
    