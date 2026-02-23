"""
Module defining acute hepatitis infectious disease model.
Acute hepatitis includes: Hepatitis A, Hepatitis B, Hepatitis C, and Hepatitis E
"""


from mighti.diseases.base_disease import GenericSIS


class AcuteHepatitis(GenericSIS):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'AcuteHepatitis'
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label = 'AcuteHepatitis')

        return

