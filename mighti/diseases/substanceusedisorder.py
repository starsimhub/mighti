"""
Module defining substance use disorder as a remitting disease model.
"""


from mighti.diseases.base_disease import RemittingDisease


class SubstanceUseDisorder(RemittingDisease):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'SubstanceUseDisorder'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(
            label = 'SubstanceUseDisorder'
        )
        return

