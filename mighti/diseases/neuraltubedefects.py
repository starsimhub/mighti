"""
Module defining neural tube defects as an acute surgical disease model.
"""

from mighti.diseases.base_disease import AcuteSurgicalDisease


class NeuralTubeDefects(AcuteSurgicalDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'NeuralTubeDefects'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label='NeuralTubeDefects')
        return
    