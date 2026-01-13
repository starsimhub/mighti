"""
Module defining diarrheal disease as a GenericSIR (infectious) model.
"""

from mighti.diseases.base_disease import GenericSIR


class DiarrhealDisease(GenericSIR):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'DiarrhealDisease'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label='DiarrhealDisease')
        return
    