"""
Module defining Lower respiratory infections as an infectious disease model.

WHO cause-of-death data reports "Lower respiratory infections" separately from
COVID-19, so we model it as its own SIS condition.
"""

from mighti.diseases.base_disease import GenericSIS


class LowerRespiratoryInfections(GenericSIS):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = "LowerRespiratoryInfections"
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label="LowerRespiratoryInfections")
        return

