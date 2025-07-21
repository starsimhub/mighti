"""
Module defining prostate cancer as a chronic disease model.
"""


from mighti.diseases.base_disease import ChronicDisease


class ProstateCancer(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'ProstateCancer'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label = 'ProstateCancer')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 1
        return

