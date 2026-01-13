"""
Module defining Protein-Energy Malnutrition as a chronic disease model.
"""

from mighti.diseases.base_disease import AcuteDisease

class ProteinEnergyMalnutrition(AcuteDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'ProteinEnergyMalnutrition'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label='ProteinEnergyMalnutrition')
        return
    