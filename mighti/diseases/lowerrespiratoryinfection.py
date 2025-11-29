"""
Module defining Lower Respiratory Infections as an infectious disease model.

This disease demonstrates the dynamic severity framework:
- Acquisition probability increases with HIV severity (50% per severity level above 1)
- Severity tracks HIV severity
- Mortality increases with severity (40% per severity level)
"""

from mighti.diseases.base_disease import GenericSIS
import starsim as ss


class LowerRespiratoryInfections(GenericSIS):
    def __init__(self, csv_path, pars=None, **kwargs):
        self.disease_name = 'LowerRespiratoryInfections'
        
        # Configure severity tracking and multipliers
        if pars is None:
            pars = {}
        pars['track_severity_from'] = 'hiv'  # Track severity from HIV
        pars['severity_acquisition_per_level'] = 0.5  # 50% increase per severity level above 1
        pars['severity_mortality_per_level'] = 0.4  # 40% increase per severity level above 1
        
        super().__init__(csv_path, pars, **kwargs)
        self.define_pars(label='LowerRespiratoryInfections')
        return
