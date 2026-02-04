"""
Module defining Bipolar Disorder as a remitting disease model.
"""

from mighti.diseases.base_disease import RemittingDisease


class BipolarDisorder(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = "BipolarDisorder"
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label="BipolarDisorder")
        return

