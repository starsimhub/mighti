"""
Module defining depressin as a remitting disease model.
"""


from mighti.diseases.base_disease import RemittingDisease


class Depression(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'Depression'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label = 'Depression')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 0.018  
        return

