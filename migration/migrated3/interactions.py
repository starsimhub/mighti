
import starsim as ss
import mighti as mi
import pandas as pd
import sciris as sc
from collections import defaultdict

# Specify all externally visible classes this file defines
__all__ = [
    'hiv_hypertension', 'hiv_obesity', 'hiv_type1diabetes', 'hiv_type2diabetes',
    'hiv_depression', 'hiv_accident', 'hiv_alzheimers', 'hiv_assault',
    'hiv_cerebrovasculardisease', 'hiv_chronicliverdisease',
    'hiv_chroniclowerrespiratorydisease', 'hiv_heartdisease',
    'hiv_chronickidneydisease', 'hiv_flu', 'hiv_hpv', 'hiv_parkinsons',
    'hiv_smoking', 'hiv_alcohol', 'hiv_brca', 'hiv_viralhepatitis',
    'hiv_cervicalcancer', 'hiv_colorectalcancer', 'hiv_breastcancer', 'hiv_lungcancer',
    'hiv_prostatecancer', 'hiv_othercancer', 'hiv_poverty',
    'GenericNCDConnector', 'read_interactions'
]


class HIVConnector(ss.Connector):
    """ Base class for connectors that adjust susceptibility for people with HIV """

    def __init__(self, disease_label, disease_class, rel_susceptibility, pars=None, **kwargs):
        label = f'HIV-{disease_label}'
        super().__init__(label=label, requires=[ss.HIV, disease_class])
        self.define_pars(
            rel_susceptibility=rel_susceptibility
        )
        self.update_pars(pars, **kwargs)
        return

    def step(self):
        sim = self.sim
        disease_attr = self.label.split('-')[-1].lower()
        # Apply the increased susceptibility to those with HIV
        getattr(sim.diseases, disease_attr).rel_sus[sim.people.hiv.infected] = self.pars.rel_susceptibility
        return


class hiv_hypertension(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Hypertension', mi.Hypertension, 1.3, pars, **kwargs)


class hiv_obesity(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Obesity', mi.Obesity, 1.2, pars, **kwargs)


class hiv_type1diabetes(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Type1Diabetes', mi.Type1Diabetes, 1.5, pars, **kwargs)


class hiv_type2diabetes(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Type2Diabetes', mi.Type2Diabetes, 1.5, pars, **kwargs)


class hiv_depression(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Depression', mi.Depression, 2.0, pars, **kwargs)


class hiv_accident(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Accident', mi.Accident, 1.1, pars, **kwargs)


class hiv_alzheimers(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Alzheimers', mi.Alzheimers, 1.1, pars, **kwargs)


class hiv_assault(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Assault', mi.Assault, 1.1, pars, **kwargs)


class hiv_cerebrovasculardisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Cerebrovascular', mi.CerebrovascularDisease, 1.2, pars, **kwargs)


class hiv_chronicliverdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Liver', mi.ChronicLiverDisease, 1.2, pars, **kwargs)


class hiv_chroniclowerrespiratorydisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Respiratory', mi.ChronicLowerRespiratoryDisease, 1.2, pars, **kwargs)


class hiv_heartdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Heart', mi.HeartDisease, 1.3, pars, **kwargs)


class hiv_chronickidneydisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Kidney', mi.ChronicKidneyDisease, 1.3, pars, **kwargs)


class hiv_flu(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Flu', mi.Flu, 1.2, pars, **kwargs)


class hiv_hpv(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HPV', mi.HPV, 1.5, pars, **kwargs)


class hiv_parkinsons(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Parkinsons', mi.Parkinsons, 1.3, pars, **kwargs)


class hiv_smoking(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Smoking', mi.Smoking, 1.5, pars, **kwargs)


class hiv_alcohol(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Alcohol', mi.Alcohol, 1.4, pars, **kwargs)


class hiv_brca(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('BRCA', mi.BRCA, 1.3, pars, **kwargs)


class hiv_cervicalcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Cervical', mi.CervicalCancer, 1.3, pars, **kwargs)


class hiv_colorectalcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Colorectal', mi.ColorectalCancer, 1.3, pars, **kwargs)


class hiv_breastcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Breast', mi.BreastCancer, 1.3, pars, **kwargs)


class hiv_lungcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Lung', mi.LungCancer, 1.4, pars, **kwargs)


class hiv_prostatecancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Prostate', mi.ProstateCancer, 1.3, pars, **kwargs)


class hiv_othercancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Other', mi.OtherCancer, 1.3, pars, **kwargs)


class hiv_viralhepatitis(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Hepatitis', mi.ViralHepatitis, 1.3, pars, **kwargs)


class hiv_poverty(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Poverty', mi.Poverty, 1.3, pars, **kwargs)


# Functions to read in datafiles
def read_interactions(datafile=None):
    """
    Read in datafile with risk/condition interactions.
    At some point, this can be adjusted to automatically create the Connectors.
    """
    if datafile is None:
        datafile = sc.thispath() / '../mighti/data/rel_sus.csv'
    df = pd.read_csv(datafile)

    rel_sus = defaultdict(dict)

    for cond in df.has_condition.unique():
        conddf = df.loc[df.has_condition == cond]
        conddf.reset_index(inplace=True, drop=True)
        interacting_conds = conddf.columns[~conddf.isna().any()].tolist()
        interacting_conds.remove('has_condition')
        for interacting_cond in interacting_conds:
            rel_sus[interacting_cond][cond] = conddf.loc[0, interacting_cond]

    return rel_sus


class GenericNCDConnector(ss.Connector):
    """
    A generic connector to model interactions between two diseases.
    This class adjusts susceptibility based on relative risk from the interaction matrix.
    """

    def __init__(self, condition1, condition2, relative_risk, pars=None, **kwargs):
        """
        Initialize the connector with the two interacting conditions and their relative risk.
        """
        label = f'{condition1}-{condition2}'  # Create a unique label for each connector
        super().__init__(label=label, requires=[getattr(mi, condition1), getattr(mi, condition2)])
        self.condition1 = condition1
        self.condition2 = condition2
        self.relative_risk = relative_risk
        self.define_pars(rel_sus=relative_risk)  # Use the passed relative risk
        self.update_pars(pars, **kwargs)
        return

    def step(self):
        sim = self.sim
        cond1_obj = getattr(sim.diseases, self.condition1.lower())  # Get the first disease object
        cond2_obj = getattr(sim.diseases, self.condition2.lower())  # Get the second disease object

        # Determine if the second condition uses 'infected' or 'affected' and adjust accordingly
        if hasattr(cond1_obj, 'infected'):
            # If the first disease uses 'infected'
            condition1_uids = cond1_obj.infected.uids
        elif hasattr(cond1_obj, 'affected'):
            # If the first disease uses 'affected'
            condition1_uids = cond1_obj.affected.uids
        else:
            raise AttributeError(f"{self.condition1} does not have 'infected' or 'affected' attribute.")

        # Now, apply the susceptibility adjustment to condition2 based on condition1
        if hasattr(cond2_obj, 'infected'):
            # If the second disease uses 'infected'
            sim.diseases[self.condition2.lower()].rel_sus[condition1_uids] = self.pars.rel_sus
        elif hasattr(cond2_obj, 'affected'):
            # If the second disease uses 'affected'
            sim.diseases[self.condition2.lower()].rel_sus[condition1_uids] = self.pars.rel_sus
        else:
            raise AttributeError(f"{self.condition2} does not have 'infected' or 'affected' attribute.")
        return
