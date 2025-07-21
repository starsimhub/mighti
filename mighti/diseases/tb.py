"""
Module defining TB infectiou as a infectious disease model.
"""


from mighti.diseases.base_disease import GenericSIS


class TB(GenericSIS):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'TB'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label = 'TB')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 1
        return
    