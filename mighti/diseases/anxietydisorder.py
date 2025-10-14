"""
Module defining the Anxiety Disorder remitting disease model.
"""


from mighti.diseases.base_disease import RemittingDisease


class AnxietyDisorder (RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'AnxietyDisorder'
        super().__init__(csv_path, pars, **kwargs)
        
        self.define_pars(label = 'AnxietyDisorder')

        return

