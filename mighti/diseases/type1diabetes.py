"""
Module defining type 1 diabetes as a chronic disease model.
"""


from mighti.diseases.base_disease import ChronicDisease


class Type1Diabetes(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Type1Diabetes'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label = 'Type1Diabetes')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 1
        return

