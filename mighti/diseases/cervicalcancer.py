"""
Module defining cervical cancer as a chronic disease model.
"""


from mighti.diseases.base_disease import ChronicDisease


class CervicalCancer(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'CervicalCancer'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label = 'CervicalCancer')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 1
        return

