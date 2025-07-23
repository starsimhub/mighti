"""
Module defining HPV infection as an infectious disease model.
"""


from mighti.diseases.base_disease import GenericSIS



class HPV(GenericSIS):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'HPV'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars( label = 'HPV')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 1
        return

