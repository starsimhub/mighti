import starsim as ss
import mighti as mi
import pandas as pd
import sciris as sc
from collections import defaultdict

# Specify all externally visible classes this file defines
__all__ = [
    'hiv_hypertension', 'hiv_obesity', 'hiv_type1diabetes', 'hiv_type2diabetes',
    'hiv_depression','hiv_alzheimersdisease', 'hiv_parkinsonsdisease','hiv_ptsd',
    'hiv_hivassociateddimentia','hiv_cardiovasculardiseases', 'hiv_chronicliverdisease',
    'hiv_asthma','hiv_roadinjuries','hiv_domesticviolence',
    'hiv_chronickidneydisease', 'hiv_flu', 'hiv_hpvvaccination', 'hiv_tobaccouse', 'hiv_alcoholusedisorder',
    'hiv_viralhepatitis','hiv_copd','hiv_hyperlipidemia',
    'hiv_cervicalcancer', 'hiv_colorectalcancer', 'hiv_breastcancer', 'hiv_lungcancer',
    'hiv_prostatecancer', 
    'GenericNCDConnector', 'read_interactions'
]


# Base class for HIV-related connectors
class HIVConnector(ss.Connector):
    """ Base class for connectors that increase susceptibility due to HIV """
    def __init__(self, label, requires, susceptibility_key, default_susceptibility, pars=None, **kwargs):
        super().__init__(label=label)
        self.define_pars(**{susceptibility_key: default_susceptibility})
        self.update_pars(pars, **kwargs)
        self.susceptibility_key = susceptibility_key
        self.requires = requires

    def step(self):
        sim = self.sim
        disease_name = self.susceptibility_key.split('_')[-1].lower()
        disease_obj = getattr(sim.diseases, disease_name, None)
        hiv_infected_uids = sim.people.hiv.infected.uids
        if disease_obj is None:
            print(f"[WARNING] {disease_name} not found in simulation diseases.")
            return

    # Check if rel_sus exists
        if not hasattr(disease_obj, 'rel_sus'):
            print(f"[ERROR] Disease {disease_name} does not have `rel_sus`. Fix initialization.")
            return
    
        print(f"[DEBUG] Adjusting susceptibility for {disease_name}. HIV-infected individuals: {len(hiv_infected_uids)}")
    
        # Check for NaN values in rel_sus
        print(f"[DEBUG] Initial rel_sus values for {disease_name}: {disease_obj.rel_sus}")
    
        # Ensure rel_sus is properly initialized
        if disease_obj.rel_sus is None or isinstance(disease_obj.rel_sus, float):
            print(f"[ERROR] `rel_sus` is not initialized correctly for {disease_name}.")
            return
    
        if len(hiv_infected_uids) > 0:
            print(f"[DEBUG] Before: {disease_name} rel_sus first 5 values: {list(disease_obj.rel_sus[:5])}")
            disease_obj.rel_sus[hiv_infected_uids] = self.pars[self.susceptibility_key]
            print(f"[DEBUG] After: {disease_name} rel_sus first 5 values: {list(disease_obj.rel_sus[:5])}")
    
        return

class hiv_type2diabetes(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Type2Diabetes', [ss.HIV, mi.Type2Diabetes], 'rel_sus_hiv_type2diabetes', 1.95, pars, **kwargs)

class hiv_type1diabetes(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Type1Diabetes', [ss.HIV, mi.Type1Diabetes], 'rel_sus_hiv_type1diabetes', 1.95, pars, **kwargs)

class hiv_hypertension(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Hypertension', [ss.HIV, mi.Hypertension], 'rel_sus_hiv_hypertension', 1.3, pars, **kwargs)

class hiv_hyperlipidemia(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Hyperlipidemia', [ss.HIV, mi.Hyperlipidemia], 'rel_sus_hiv_hyperlipidemia', 1.3, pars, **kwargs)

class hiv_asthma(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Asthma', [ss.HIV, mi.Asthma], 'rel_sus_hiv_asthma', 1.3, pars, **kwargs)

class hiv_copd(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-COPD', [ss.HIV, mi.COPD], 'rel_sus_hiv_copd', 1.3, pars, **kwargs)

class hiv_obesity(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Obesity', [ss.HIV, mi.Obesity], 'rel_sus_hiv_obesity', 1.2, pars, **kwargs)

class hiv_depression(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Depression', [ss.HIV, mi.Depression], 'rel_sus_hiv_depression', 2, pars, **kwargs)

class hiv_ptsd(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-PTSD', [ss.HIV, mi.PTSD], 'rel_sus_hiv_ptsd', 2, pars, **kwargs)

class hiv_hivassociateddimentia(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-HIVAssociatedDimentia', [ss.HIV, mi.HIVAssociatedDementia], 'rel_sus_hiv_hivassociateddimentia', 2, pars, **kwargs)

class hiv_roadinjuries(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Accident', [ss.HIV, mi.RoadInjuries], 'rel_sus_hiv_roadinjuries', 1.1, pars, **kwargs)

class hiv_domesticviolence(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-DomesticViolence', [ss.HIV, mi.DomesticViolence], 'rel_sus_hiv_domesticviolence', 1.1, pars, **kwargs)

class hiv_alzheimersdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-AlzheimersDisease', [ss.HIV, mi.AlzheimersDisease], 'rel_sus_hiv_alzheimersdisease', 1.1, pars, **kwargs)

class hiv_cardiovasculardiseases(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-CardiovascularDiseases', [ss.HIV, mi.CardiovascularDiseases], 'rel_sus_hiv_cardiovasculardiseases', 1.2, pars, **kwargs)

class hiv_chronicliverdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-ChronicLiverDisease', [ss.HIV, mi.ChronicLiverDisease], 'rel_sus_hiv_chronicliverdisease', 1.2, pars, **kwargs)

class hiv_chroniclowerrespiratorydisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Respiratory', [ss.HIV, mi.ChronicLowerRespiratoryDisease], 'rel_sus_hiv_resp', 1.2, pars, **kwargs)

class hiv_heartdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Heart', [ss.HIV, mi.HeartDisease], 'rel_sus_hiv_heart', 1.3, pars, **kwargs)

class hiv_chronickidneydisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-ChronicKidneyDisease', [ss.HIV, mi.ChronicKidneyDisease], 'rel_sus_hiv_chronickidneydisease', 1.3, pars, **kwargs)

class hiv_flu(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Flu', [ss.HIV, mi.Flu], 'rel_sus_hiv_flu', 1.2, pars, **kwargs)

class hiv_hpvvaccination(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-HPV', [ss.HIV, mi.HPVVaccination], 'rel_sus_hiv_hpv', 1.5, pars, **kwargs)

class hiv_parkinsonsdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-ParkinsonsDisease', [ss.HIV, mi.ParkinsonsDisease], 'rel_sus_hiv_parkinsonsdisease', 1.3, pars, **kwargs)

class hiv_tobaccouse(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-TobaccoUse', [ss.HIV, mi.TobaccoUse], 'rel_sus_hiv_tobaccouse', 1.5, pars, **kwargs)

class hiv_alcoholusedisorder(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-AlcoholUseDisorder', [ss.HIV, mi.AlcoholUseDisorder], 'rel_sus_hiv_alcoholusedisorder', 1.4, pars, **kwargs)

class hiv_cervicalcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Cervical', [ss.HIV, mi.CervicalCancer], 'rel_sus_hiv_cervical', 1.3, pars, **kwargs)

class hiv_colorectalcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Colorectal', [ss.HIV, mi.ColorectalCancer], 'rel_sus_hiv_colorectal', 1.3, pars, **kwargs)

class hiv_breastcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Breast', [ss.HIV, mi.BreastCancer], 'rel_sus_hiv_breast', 1.3, pars, **kwargs)

class hiv_lungcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Lung', [ss.HIV, mi.LungCancer], 'rel_sus_hiv_lung', 1.4, pars, **kwargs)

class hiv_prostatecancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Prostate', [ss.HIV, mi.ProstateCancer], 'rel_sus_hiv_prostate', 1.3, pars, **kwargs)

class hiv_othercancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Other', [ss.HIV, mi.OtherCancer], 'rel_sus_hiv_other', 1.3, pars, **kwargs)

class hiv_viralhepatitis(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Hepatitis', [ss.HIV, mi.ViralHepatitis], 'rel_sus_hiv_hepatitis', 1.3, pars, **kwargs)

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


# import starsim as ss
# import mighti as mi
# import pandas as pd
# import sciris as sc

# # -------------------------
# # Read HIV Interactions
# # -------------------------

# def read_hiv_interactions(df_params):
#     """ Read HIV interactions from `eswatini_parameters.csv`. """
#     if df_params is None:
#         raise ValueError("[ERROR] `df_params` has not been initialized. Call `initialize_conditions(df_params)` first.")

#     # Debugging: Check available columns
#     print(f"[DEBUG] Available columns in df_params: {df_params.columns}")

#     # Ensure 'rel_sus' column exists
#     if 'rel_sus' not in df_params.columns:
#         raise ValueError(f"[ERROR] 'rel_sus' column is missing in parameter file. Available columns: {df_params.columns}")

#     # Read HIV interactions (filtering out NaN values)
#     rel_sus = df_params['rel_sus'].dropna().to_dict()

#     print(f"[DEBUG] Read HIV interactions: {rel_sus}")
#     return rel_sus


# # -------------------------
# # Generic HIV-NCD Connector
# # -------------------------

# class HIVConnector(ss.Connector):
#     """
#     Generic connector to model increased susceptibility due to HIV.
#     """

#     def __init__(self, condition, relative_risk, pars=None, **kwargs):
#         label = f'HIV-{condition}'
#         condition_obj = globals().get(condition, None)  # Ensure the disease class exists

#         if condition_obj is None:
#             raise ValueError(f"[ERROR] Condition {condition} not found in `mighti` module.")

#         super().__init__(label=label)
#         self.condition = condition
#         self.relative_risk = relative_risk
#         self.define_pars(rel_sus=relative_risk)
#         self.update_pars(pars, **kwargs)

#     def step(self):
#         sim = self.sim
#         condition_obj = getattr(sim.diseases, self.condition.lower(), None)

#         if not condition_obj:
#             return  # Skip if the disease object is not initialized

#         # Get HIV-infected individuals
#         hiv_infected_uids = sim.people.hiv.infected.uids

#         # Adjust relative susceptibility
#         for uid in hiv_infected_uids:
#             condition_obj.rel_sus[uid] = self.pars.rel_sus
#         return


# # -------------------------
# # Create HIV-NCD Connectors Dynamically
# # -------------------------

# def create_hiv_connectors(df_params):
#     """
#     Reads HIV interaction data from `eswatini_parameters.csv` and dynamically creates HIV-NCD connectors.
#     """
#     rel_sus_data = read_hiv_interactions(df_params)
#     connectors = []

#     for condition, relative_risk in rel_sus_data.items():
#         print(f"[DEBUG] Creating HIV connector for {condition} with RR={relative_risk}")

#         try:
#             connector = HIVConnector(condition, relative_risk)
#             connectors.append(connector)
#         except ValueError as e:
#             print(f"[WARNING] Skipping {condition}: {e}")

#     print(f"[DEBUG] Created {len(connectors)} HIV-NCD connectors: {connectors}")
#     return connectors