"""
Module defining viral hepatitis infectious disease model.
"""


from mighti.diseases.base_disease import GenericSIS


class ViralHepatitis(GenericSIS):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'ViralHepatitis'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(
            label = 'ViralHepatitis'
        )
        return

