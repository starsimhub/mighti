"""
Module defining obesity as a chronic disease model.
"""


from mighti.diseases.base_disease import ChronicDisease


class Obesity(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Obesity'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label = 'Obesity')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 1
        return

