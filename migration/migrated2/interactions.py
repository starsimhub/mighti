
import starsim as ss
import mighti as mi
import pandas as pd
import sciris as sc
from collections import defaultdict

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


class BaseHIVConnector(ss.Connector):
    """ Base class for connectors increasing susceptibility due to HIV """
    
    def __init__(self, disease_name, rel_sus, requires, pars=None, **kwargs):
        label = f'HIV-{disease_name}'
        super().__init__(name=label.lower(), label=label, requires=requires)
        self.define_pars(rel_sus=rel_sus)
        self.update_pars(pars, **kwargs)
        return

    def step(self):
        sim = self.sim
        disease = self.name.split('-')[1]  # Assume the format is 'hiv-disease'
        sim.diseases[disease].rel_sus[sim.people.hiv.infected] = self.pars.rel_sus
        return


class hiv_hypertension(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Hypertension', 1.3, [ss.HIV, mi.Hypertension], pars, **kwargs)


class hiv_obesity(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Obesity', 1.2, [ss.HIV, mi.Obesity], pars, **kwargs)


class hiv_type1diabetes(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Type1Diabetes', 1.5, [ss.HIV, mi.Type1Diabetes], pars, **kwargs)


class hiv_type2diabetes(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Type2Diabetes', 1.5, [ss.HIV, mi.Type2Diabetes], pars, **kwargs)


class hiv_depression(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Depression', 2.0, [ss.HIV, mi.Depression], pars, **kwargs)


class hiv_accident(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Accident', 1.1, [ss.HIV, mi.Accident], pars, **kwargs)


class hiv_alzheimers(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Alzheimers', 1.1, [ss.HIV, mi.Alzheimers], pars, **kwargs)


class hiv_assault(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Assault', 1.1, [ss.HIV, mi.Assault], pars, **kwargs)


class hiv_cerebrovasculardisease(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('CerebrovascularDisease', 1.2, [ss.HIV, mi.CerebrovascularDisease], pars, **kwargs)


class hiv_chronicliverdisease(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('ChronicLiverDisease', 1.2, [ss.HIV, mi.ChronicLiverDisease], pars, **kwargs)


class hiv_chroniclowerrespiratorydisease(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('ChronicLowerRespiratoryDisease', 1.2, [ss.HIV, mi.ChronicLowerRespiratoryDisease], pars, **kwargs)


class hiv_heartdisease(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HeartDisease', 1.3, [ss.HIV, mi.HeartDisease], pars, **kwargs)


class hiv_chronickidneydisease(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('ChronicKidneyDisease', 1.3, [ss.HIV, mi.ChronicKidneyDisease], pars, **kwargs)


class hiv_flu(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Flu', 1.2, [ss.HIV, mi.Flu], pars, **kwargs)


class hiv_hpv(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HPV', 1.5, [ss.HIV, mi.HPV], pars, **kwargs)


class hiv_parkinsons(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Parkinsons', 1.3, [ss.HIV, mi.Parkinsons], pars, **kwargs)


class hiv_smoking(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Smoking', 1.5, [ss.HIV, mi.Smoking], pars, **kwargs)


class hiv_alcohol(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Alcohol', 1.4, [ss.HIV, mi.Alcohol], pars, **kwargs)


class hiv_brca(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('BRCA', 1.3, [ss.HIV, mi.BRCA], pars, **kwargs)


class hiv_cervicalcancer(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('CervicalCancer', 1.3, [ss.HIV, mi.CervicalCancer], pars, **kwargs)


class hiv_colorectalcancer(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('ColorectalCancer', 1.3, [ss.HIV, mi.ColorectalCancer], pars, **kwargs)


class hiv_breastcancer(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('BreastCancer', 1.3, [ss.HIV, mi.BreastCancer], pars, **kwargs)


class hiv_lungcancer(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('LungCancer', 1.4, [ss.HIV, mi.LungCancer], pars, **kwargs)


class hiv_prostatecancer(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('ProstateCancer', 1.3, [ss.HIV, mi.ProstateCancer], pars, **kwargs)


class hiv_othercancer(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('OtherCancer', 1.3, [ss.HIV, mi.OtherCancer], pars, **kwargs)


class hiv_viralhepatitis(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('ViralHepatitis', 1.3, [ss.HIV, mi.ViralHepatitis], pars, **kwargs)


class hiv_poverty(BaseHIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('Poverty', 1.3, [ss.HIV, mi.Poverty], pars, **kwargs)


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
        super().__init__(name=label.lower(), label=label, requires=[getattr(mi, condition1), getattr(mi, condition2)])
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
            condition1_uids = cond1_obj.infected.uids
        elif hasattr(cond1_obj, 'affected'):
            condition1_uids = cond1_obj.affected.uids
        else:
            raise AttributeError(f"{self.condition1} does not have 'infected' or 'affected' attribute.")
        
        # Apply the susceptibility adjustment to condition2 based on condition1
        if hasattr(cond2_obj, 'infected'):
            sim.diseases[self.condition2.lower()].rel_sus[condition1_uids] = self.pars.rel_sus
        elif hasattr(cond2_obj, 'affected'):
            sim.diseases[self.condition2.lower()].rel_sus[condition1_uids] = self.pars.rel_sus
        else:
            raise AttributeError(f"{self.condition2} does not have 'infected' or 'affected' attribute.")
        return
