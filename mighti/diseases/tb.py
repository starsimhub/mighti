"""
Module defining Tuberculosis (TB) as an infectious disease model.

Note: This is currently a placeholder SIR model. A more realistic TB model
would include latent infection, prolonged duration, and treatment dynamics.
"""


from mighti.diseases.base_disease import GenericSIR


class Tuberculosis(GenericSIR):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = "Tuberculosis"
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label="Tuberculosis")
        return
