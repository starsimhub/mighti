import starsim as ss
import mighti as mi
import pandas as pd
import sciris as sc
from collections import defaultdict

# Specify all externally visible classes this file defines
__all__ = [
    'hiv_hypertension','hiv_obesity','hiv_type1diabetes','hiv_type2diabetes','hiv_depression',
    'hiv_accident', 'hiv_alzheimers', 'hiv_assault','hiv_cerebro', 'hiv_liver', 'hiv_resp', 
    'hiv_heart', 'hiv_kidney', 'hiv_flu','hiv_hpv', 'hiv_parkinsons', 
    'hiv_smoking', 'hiv_alcohol', 'hiv_brca',
    'hiv_cervical','hiv_colorectal', 'hiv_breast', 'hiv_lung', 'hiv_prostate','hiv_other',
    'GenericNCDConnector','read_interactions'
]


class hiv_hypertension(ss.Connector):
    """ Simple connector to make people with HIV more likely to contract Hypertension """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Hypertension', requires=[ss.HIV, mi.Hypertension])
        self.default_pars(
            rel_sus_hiv_hypertension=1.3,  # People with HIV are 1.3x more likely to acquire Hypertension
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        # Apply the increased susceptibility to those with HIV
        sim.diseases.hypertension.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_hypertension
        return


class hiv_obesity(ss.Connector):
    """ Simple connector to make people with HIV more likely to contract Obesity """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Obesity', requires=[ss.HIV, mi.Obesity])
        self.default_pars(
            rel_sus_hiv_obesity=1.2,  # People with HIV are 1.2x more likely to acquire Obesity
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        # Apply the increased susceptibility to those with HIV
        sim.diseases.obesity.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_obesity
        return
    
    
class hiv_type1diabetes(ss.Connector):
    """ Simple connector to make people with HIV more likely to contract Type 1 Diabetes """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Type1Diabetes', requires=[ss.HIV, mi.Type1Diabetes])
        self.default_pars(
            rel_sus_hiv_type1diabetes=1.5,  # People with HIV are 1.5x more likely to acquire Type 1 Diabetes
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        # Apply the increased susceptibility to those with HIV
        sim.diseases.type1diabetes.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_type1diabetes
        return
    
    
class hiv_type2diabetes(ss.Connector):
    """ Simple connector to make people with HIV more likely to contract Type 2 Diabetes """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Type2Diabetes', requires=[ss.HIV, mi.Type2Diabetes])
        self.default_pars(
            rel_sus_hiv_type2diabetes=1.5,  # People with HIV are 1.5x more likely to acquire Type 2 Diabetes
        )
        self.update_pars(pars, **kwargs)
        return

    # def update(self):
    #     sim = self.sim
    #     print(f"Type2Diabetes object: {sim.diseases.type2diabetes}")
    #     print(f"Attributes of Type2Diabetes: {dir(sim.diseases.type2diabetes)}")
    #     # Apply the increased susceptibility to those with HIV
    #     sim.diseases.type2diabetes.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_type2diabetes
    #     return
    def update(self):
        sim = self.sim
        if sim.diseases.type2diabetes.rel_sus is None:
            print("rel_sus is still None during update.")
        else:
            print(f"rel_sus is initialized with values: {sim.diseases.type2diabetes.rel_sus}")
            sim.diseases.type2diabetes.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_type2diabetes
        return


class hiv_depression(ss.Connector):
    """ Simple connector to make people with HIV more likely to contract Depression """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Depression', requires=[ss.HIV, mi.Depression])  # Use ss.HIV class
        self.default_pars(
            rel_sus_hiv_depression=2,  # People with HIV are 2x more likely to acquire Depression
        )
        self.update_pars(pars, **kwargs)
        return
    
    def update(self):
        sim = self.sim
        # Apply the increased susceptibility to those with HIV
        sim.diseases.depression.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_depression
        return


class hiv_accident(ss.Connector):
    """ Connector to make people with HIV more likely to suffer from accidents """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Accident', requires=[ss.HIV, mi.Accident])
        self.default_pars(rel_sus_hiv_accident=1.1)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        # Apply the increased susceptibility to those with HIV
        sim.diseases.accident.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_accident
        return


class hiv_alzheimers(ss.Connector):
    """ Connector to make people with HIV more likely to contract Alzheimer's """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Alzheimers', requires=[ss.HIV, mi.Alzheimers])
        self.default_pars(rel_sus_hiv_alzheimers=1.1)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        # Apply the increased susceptibility to those with HIV
        sim.diseases.alzheimers.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_alzheimers
        return


class hiv_assault(ss.Connector):
    """ Connector for people with HIV more likely to be involved in Assault """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Assault', requires=[ss.HIV, mi.Assault])
        self.default_pars(rel_sus_hiv_assault=1.1)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.assault.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_assault
        return
    

class hiv_cerebro(ss.Connector):
    """ Connector for people with HIV more likely to acquire Cerebrovascular diseases """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Cerebrovascular', requires=[ss.HIV, mi.CerebrovascularDisease])
        self.default_pars(rel_sus_hiv_cerebro=1.2)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.cerebrovasculardisease.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_cerebro
        return


class hiv_liver(ss.Connector):
    """ Connector for people with HIV more likely to acquire Chronic Liver Disease """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Liver', requires=[ss.HIV, mi.ChronicLiverDisease])
        self.default_pars(rel_sus_hiv_liver=1.2)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.chronicliverdisease.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_liver
        return


class hiv_resp(ss.Connector):
    """ Connector for people with HIV more likely to acquire Chronic Lower Respiratory Disease """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Respiratory', requires=[ss.HIV, mi.ChronicLowerRespiratoryDisease])
        self.default_pars(rel_sus_hiv_resp=1.2)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.chroniclowerrespiratorydisease.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_resp
        return
    

class hiv_heart(ss.Connector):
    """ Connector for people with HIV more likely to acquire Heart Disease """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Heart', requires=[ss.HIV, mi.HeartDiseases])
        self.default_pars(rel_sus_hiv_heart=1.3)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.heartdiseases.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_heart
        return


class hiv_kidney(ss.Connector):
    """ Connector for people with HIV more likely to acquire Chronic Kidney Disease """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Kidney', requires=[ss.HIV, mi.ChronicKidneyDisease])
        self.default_pars(rel_sus_hiv_kidney=1.3)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.chronickidneydisease.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_kidney
        return


class hiv_flu(ss.Connector):
    """ Connector for people with HIV more likely to acquire Influenza """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Flu', requires=[ss.HIV, mi.Flu])
        self.default_pars(rel_sus_hiv_flu=1.2)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.flu.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_flu
        return


class hiv_hpv(ss.Connector):
    """ Connector for people with HIV more likely to acquire HPV """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-HPV', requires=[ss.HIV, mi.HPV])
        self.default_pars(rel_sus_hiv_hpv=1.5)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.hpv.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_hpv
        return


class hiv_parkinsons(ss.Connector):
    """ Connector for people with HIV more likely to acquire Parkinson's disease """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Parkinsons', requires=[ss.HIV, mi.Parkinsons])
        self.default_pars(rel_sus_hiv_parkinsons=1.3)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.parkinsons.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_parkinsons
        return


class hiv_smoking(ss.Connector):
    """ Connector for people with HIV more likely to be infected by smoking """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Smoking', requires=[ss.HIV, mi.Smoking])
        self.default_pars(rel_sus_hiv_smoking=1.5)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.smoking.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_smoking
        return


class hiv_alcohol(ss.Connector):
    """ Connector for people with HIV more likely to suffer from alcohol-related conditions """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Alcohol', requires=[ss.HIV, mi.Alcohol])
        self.default_pars(rel_sus_hiv_alcohol=1.4)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.alcohol.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_alcohol
        return


class hiv_brca(ss.Connector):
    """ Connector for people with HIV more likely to be infected by BRCA mutation """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-BRCA', requires=[ss.HIV, mi.BRCA])
        self.default_pars(rel_sus_hiv_brca=1.3)
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.brca.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_brca
        return
    
    
class hiv_cervical(ss.Connector):
    """ Connector for people with HIV more likely to acquire Cervical Cancer """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Cervical', requires=[ss.HIV, mi.CervicalCancer])
        self.default_pars(
            rel_sus_hiv_cervical=1.3  # People with HIV are 1.3x more likely to acquire Cervical Cancer
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.cervicalcancer.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_cervical
        return

    
class hiv_colorectal(ss.Connector):
    """ Connector for people with HIV more likely to acquire Colorectal Cancer """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Colorectal', requires=[ss.HIV, mi.ColorectalCancer])
        self.default_pars(
            rel_sus_hiv_colorectal=1.3  # People with HIV are 1.3x more likely to acquire Colorectal Cancer
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.colorectalcancer.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_colorectal
        return


class hiv_breast(ss.Connector):
    """ Connector for people with HIV more likely to acquire Breast Cancer """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Breast', requires=[ss.HIV, mi.BreastCancer])
        self.default_pars(
            rel_sus_hiv_breast=1.3  # People with HIV are 1.3x more likely to acquire Breast Cancer
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.breastcancer.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_breast
        return


class hiv_lung(ss.Connector):
    """ Connector for people with HIV more likely to acquire Lung Cancer """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Lung', requires=[ss.HIV, mi.LungCancer])
        self.default_pars(
            rel_sus_hiv_lung=1.4  # People with HIV are 1.4x more likely to acquire Lung Cancer
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.lungcancer.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_lung
        return


class hiv_prostate(ss.Connector):
    """ Connector for people with HIV more likely to acquire Prostate Cancer """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Prostate', requires=[ss.HIV, mi.ProstateCancer])
        self.default_pars(
            rel_sus_hiv_prostate=1.3  # People with HIV are 1.3x more likely to acquire Prostate Cancer
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.prostatecancer.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_prostate
        return


class hiv_other(ss.Connector):
    """ Connector for people with HIV more likely to acquire Other Cancers """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Other', requires=[ss.HIV, mi.OtherCancer])
        self.default_pars(
            rel_sus_hiv_other=1.3  # People with HIV are 1.3x more likely to acquire Other Cancers
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        sim.diseases.othercancer.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_other
        return

    
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
        # print(f"Creating connector with label: {label}")  # For debugging purposes
        super().__init__(label=label, requires=[getattr(mi, condition1), getattr(mi, condition2)])
        self.condition1 = condition1
        self.condition2 = condition2
        self.relative_risk = relative_risk
        self.default_pars(rel_sus=relative_risk)  # Use the passed relative risk
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        cond1_obj = getattr(sim.diseases, self.condition1.lower())  # Get the first disease object
        
        # Apply the relative risk from condition1 to condition2's susceptible individuals
        sim.diseases[self.condition2.lower()].rel_sus[cond1_obj.infected] = self.pars.rel_sus
        return