"""
Module defining Flu as an infectious disease model.
"""


from mighti.diseases.base_disease import GenericSIS



class Flu(GenericSIS):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Flu'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label = 'Flu')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 1
        return

