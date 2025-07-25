"""
Module defining Alzheimer's disease as a chronic disease model.
"""


from mighti.diseases.base_disease import ChronicDisease


class AlzheimersDisease(ChronicDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'AlzheimersDisease'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'AlzheimersDisease')
        if not hasattr(self.pars, 'p_acquire'):
            self.pars.p_acquire = 1

        return

