"""
Module defining Self Harm as an acute disease model.

This is intended to represent a self-harm/suicide cause category from GBD.
"""

from mighti.diseases.base_disease import AcuteDisease


class SelfHarm(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = "SelfHarm"
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label="SelfHarm")
        return

