import starsim as ss
import mighti as mi
import pandas as pd
import sciris as sc
from collections import defaultdict
import numpy as np

# Specify all externally visible classes this file defines
__all__ = [
    'hiv_hypertension', 'hiv_obesity', 'hiv_type1diabetes', 'hiv_type2diabetes',
    'hiv_cardiovasculardiseases', 'hiv_chronickidneydisease', 'hiv_hyperlipidemia',
    'hiv_cervicalcancer', 'hiv_colorectalcancer', 'hiv_breastcancer', 'hiv_lungcancer',
    'hiv_prostatecancer', 'hiv_alcoholusedisorder', 'hiv_tobaccouse',
    'hiv_hivassociateddementia', 'hiv_ptsd', 'hiv_depression', 'hiv_hpv', 'hiv_flu',
    'hiv_viralhepatitis', 'hiv_domesticviolence', 'hiv_roadinjuries',
    'hiv_chronicliverdisease', 'hiv_asthma', 'hiv_copd', 'hiv_alzheimersdisease',
    'hiv_parkinsonsdisease', 'GenericNCDConnector', 'read_interactions'
]

# Base class for HIV-related connectors


class HIVConnector(ss.Connector):
    """ 
    Connector that increases susceptibility to diseases due to HIV infection.
    
    This connector modifies the `rel_sus` (relative susceptibility) attribute 
    of a target disease for individuals who are HIV-infected.
    """

    def __init__(self, label, requires, susceptibility_key, default_susceptibility, pars=None, **kwargs):
        super().__init__(label=label)
        self.define_pars(**{susceptibility_key: default_susceptibility})
        self.update_pars(pars, **kwargs)
        self.susceptibility_key = susceptibility_key
        self.requires = requires

    def step(self):
        sim = self.sim
        disease_name = self.susceptibility_key.split('_')[-1]  # Extract disease name from key

        # Get the disease object
        disease_obj = getattr(sim.diseases, disease_name.lower(), None)
        if disease_obj is None:
            print(f"[ERROR] Disease {disease_name} not found in sim.diseases")
            return

        # Ensure `sim.people.hiv` and `sim.people.hiv.infected` exist
        if not hasattr(sim.people, 'hiv') or not hasattr(sim.people.hiv, 'infected'):
            print("[ERROR] sim.people.hiv.infected is missing")
            return

        # Get HIV-infected individuals
        hiv_infected_uids = sim.people.hiv.infected.uids
        if len(hiv_infected_uids) == 0:
            print(f"[DEBUG] No HIV-infected individuals found at time {sim.ti}. Skipping update.")
            return

        # Ensure the disease object has `rel_sus` attribute
        if not hasattr(disease_obj, 'rel_sus'):
            print(f"[ERROR] {disease_name} does not have 'rel_sus' attribute.")
            return
        
        print(f"[DEBUG] {self.label}: Before update, mean rel_sus={disease_obj.rel_sus.mean()}")
        

        # Debug info before update
        first_n = 10  # Number of elements to print for debugging
        print(f"[DEBUG] {self.label}: {disease_name} rel_sus BEFORE update: {disease_obj.rel_sus[:first_n]}")
        
        # Apply susceptibility update
        disease_obj.rel_sus[hiv_infected_uids] = self.pars[self.susceptibility_key]
        print(f"[DEBUG] {self.label}: After update, mean rel_sus={disease_obj.rel_sus.mean()}")

        # Ensure no NaNs were introduced
        if np.isnan(disease_obj.rel_sus).any():
            print(f"[ERROR] {disease_name} rel_sus contains NaN AFTER update!")
            disease_obj.rel_sus = np.nan_to_num(disease_obj.rel_sus, nan=1.0)  # Replace NaNs with 1.0

        # Debug info after update
        print(f"[DEBUG] {self.label}: {disease_name} rel_sus AFTER update: {disease_obj.rel_sus[:first_n]}")
    
class hiv_type2diabetes(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Type2Diabetes', [ss.HIV, mi.Type2Diabetes], 'rel_sus_hiv_type2diabetes', 20, pars, **kwargs)
    
    def step(self):        
        sim = self.sim
        disease_name = self.susceptibility_key.split('_')[-1]
        disease_obj = getattr(sim.diseases, disease_name.lower(), None)
    
        if disease_obj is None:
            print(f"[ERROR] Disease {disease_name} not found in sim.diseases")
            return
    
        hiv_infected_uids = sim.people.hiv.infected.uids
    
        if len(hiv_infected_uids) == 0:
            print("No HIV-infected individuals found.")
            return
        
        # print(f"HIV-infected UIDs: {list(hiv_infected_uids)[:5]}")  # Print first 5 for sanity check
    
        # Print before update
        # print(f"Before update, rel_sus (first 5 values): {disease_obj.rel_sus[hiv_infected_uids[:5]]}")
    
        # Apply update
        disease_obj.rel_sus.set(hiv_infected_uids, self.pars[self.susceptibility_key])
    
        # Print after update
        # print(f"After update, rel_sus (first 5 values): {disease_obj.rel_sus[hiv_infected_uids[:5]]}")
         

class hiv_type1diabetes(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Type1Diabetes', [ss.HIV, mi.Type1Diabetes], 'rel_sus_hiv_type1diabetes', 1.95, pars, **kwargs)

class hiv_hypertension(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Hypertension', [ss.HIV, mi.Hypertension], 'rel_sus_hiv_hypertension', 1.3, pars, **kwargs)

class hiv_hyperlipidemia(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Hyperlipidemia', [ss.HIV, mi.Hyperlipidemia], 'rel_sus_hiv_hyperlipidemia', 1.3, pars, **kwargs)

class hiv_cardiovasculardiseases(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-CardiovascularDiseases', [ss.HIV, mi.CardiovascularDiseases], 'rel_sus_hiv_cardiovasculardiseases', 1.2, pars, **kwargs)


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

class hiv_hivassociateddementia(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-HIVAssociatedDementia', [ss.HIV, mi.HIVAssociatedDementia], 'rel_sus_hiv_hivassociateddementia', 2, pars, **kwargs)

class hiv_roadinjuries(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-RoadInjuries', [ss.HIV, mi.RoadInjuries], 'rel_sus_hiv_roadinjuries', 1.1, pars, **kwargs)

class hiv_domesticviolence(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-DomesticViolence', [ss.HIV, mi.DomesticViolence], 'rel_sus_hiv_domesticviolence', 1.1, pars, **kwargs)

class hiv_alzheimersdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-AlzheimersDisease', [ss.HIV, mi.AlzheimersDisease], 'rel_sus_hiv_alzheimersdisease', 1.1, pars, **kwargs)

class hiv_chronicliverdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-ChronicLiverDisease', [ss.HIV, mi.ChronicLiverDisease], 'rel_sus_hiv_chronicliverdisease', 1.2, pars, **kwargs)

class hiv_heartdisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Heart', [ss.HIV, mi.HeartDisease], 'rel_sus_hiv_heart', 1.3, pars, **kwargs)

class hiv_chronickidneydisease(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-ChronicKidneyDisease', [ss.HIV, mi.ChronicKidneyDisease], 'rel_sus_hiv_chronickidneydisease', 1.3, pars, **kwargs)

class hiv_flu(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Flu', [ss.HIV, mi.Flu], 'rel_sus_hiv_flu', 1.2, pars, **kwargs)

class hiv_hpv(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-HPV', [ss.HIV, mi.HPV], 'rel_sus_hiv_hpv', 1.5, pars, **kwargs)

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
        super().__init__('HIV-Cervical', [ss.HIV, mi.CervicalCancer], 'rel_sus_hiv_cervicalcancer', 1.3, pars, **kwargs)

class hiv_colorectalcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Colorectal', [ss.HIV, mi.ColorectalCancer], 'rel_sus_hiv_colorectalcancer', 1.3, pars, **kwargs)

class hiv_breastcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Breast', [ss.HIV, mi.BreastCancer], 'rel_sus_hiv_breastcancer', 1.3, pars, **kwargs)

class hiv_lungcancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Lung', [ss.HIV, mi.LungCancer], 'rel_sus_hiv_lungcancer', 1.4, pars, **kwargs)

class hiv_prostatecancer(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Prostate', [ss.HIV, mi.ProstateCancer], 'rel_sus_hiv_prostatecancer', 1.3, pars, **kwargs)

class hiv_viralhepatitis(HIVConnector):
    def __init__(self, pars=None, **kwargs):
        super().__init__('HIV-Hepatitis', [ss.HIV, mi.ViralHepatitis], 'rel_sus_hiv_viralhepatitis', 1.3, pars, **kwargs)


class GenericNCDConnector(ss.Connector):
    """
    A generic connector to model interactions between two diseases.
    Adjusts susceptibility (`rel_sus`) based on relative risk from the interaction matrix.
    """

    def __init__(self, condition1, condition2, relative_risk, pars=None, **kwargs):
        label = f'{condition1}-{condition2}'  
        name = label.lower().replace(" ", "_")

        super().__init__(name=name, label=label)
        self.condition1 = condition1
        self.condition2 = condition2
        self.relative_risk = relative_risk
        self.define_pars(rel_sus=relative_risk)
        self.update_pars(pars, **kwargs)

    def step(self):
        """Applies interaction effects between diseases during each simulation step."""
        sim = self.sim
        cond1_obj = getattr(sim.diseases, self.condition1.lower(), None)
        cond2_obj = getattr(sim.diseases, self.condition2.lower(), None)
    
        if cond1_obj is None or cond2_obj is None:
            print(f"[ERROR] {self.name}: One or both disease objects not found in simulation! Skipping step.")
            return  

        print(f"[DEBUG] {self.name}: Applying interaction at time {self.sim.ti}")
        
        # Print number of already affected individuals
        num_already_affected = np.sum(cond2_obj.ti_affected != -1)
        print(f"[DEBUG] {self.name}: {self.condition2} individuals already affected before step: {num_already_affected}")

        # Print incidence probability for debugging
        print(f"[DEBUG] {self.name}: {self.condition2} incidence probability → {sim.pars[self.condition2.lower()]['incidence_prob']}")

        # Ensure `rel_sus` array exists and has no NaNs
        if cond2_obj.rel_sus is None or len(cond2_obj.rel_sus) == 0:
            print(f"[WARNING] {self.name}: rel_sus for {self.condition2} is EMPTY. Skipping interaction.")
            return

        cond2_obj.rel_sus.raw = np.nan_to_num(cond2_obj.rel_sus.raw, nan=1.0)

        before = cond2_obj.rel_sus.mean()
        print(f"[DEBUG] {self.name}: {self.condition2} rel_sus mean BEFORE: {before}, min: {cond2_obj.rel_sus.min()}, max: {cond2_obj.rel_sus.max()}")

        # Determine whether `cond2_obj` uses "affected" (NCD) or "infected" (SIS)
        status_attr = "infected" if hasattr(cond2_obj, "infected") else "affected"
        
        uids = getattr(cond2_obj, status_attr).uids  

        if len(uids) == 0:
            print(f"[WARNING] {self.name}: No {status_attr} individuals in {self.condition2}. Skipping update.")
            return
        
        # Print `ti_affected` values before update
        print(f"[DEBUG] {self.name}: {self.condition2} ti_affected values BEFORE update: {cond2_obj.ti_affected[:20]}")

        # Fix NaNs in `ti_affected`
        if np.isnan(cond2_obj.ti_affected).any():
            print(f"[CRITICAL WARNING] {self.name}: {self.condition2} ti_affected contains NaNs! Fixing.")
            cond2_obj.ti_affected = np.nan_to_num(cond2_obj.ti_affected, nan=-1)  # Replace NaNs with -1

        # Find newly affected individuals safely
        valid_uids = uids[uids < len(cond2_obj.ti_affected)]
        if len(valid_uids) == 0:
            print(f"[WARNING] {self.name}: No valid indices for {self.condition2}. Skipping update.")
            return
        
        new_cases = valid_uids[np.where(cond2_obj.ti_affected[valid_uids] == self.sim.ti)]
        
        if len(new_cases) == 0:
            print(f"[DEBUG] {self.name}: No new cases detected for {self.condition2}. Check incidence calculation.")
            return

        # Apply relative risk adjustment
        new_rel_sus = cond2_obj.rel_sus.raw[new_cases] * self.relative_risk
        new_rel_sus = np.nan_to_num(new_rel_sus, nan=1.0)
        new_rel_sus = np.clip(new_rel_sus, 1.0, 10)  # Keep values reasonable

        cond2_obj.rel_sus.set(new_cases, new_rel_sus)  

        # Compute adjusted incidence probability
        adjusted_incidence = sim.pars[self.condition2.lower()]['incidence_prob'] * cond2_obj.rel_sus.raw

        print(f"[DEBUG] {self.condition2}: Adjusted incidence probability → mean={np.nanmean(adjusted_incidence)}, min={np.nanmin(adjusted_incidence)}, max={np.nanmax(adjusted_incidence)}")

        # Check `rel_sus` after update
        if np.isnan(cond2_obj.rel_sus.raw).any():
            print(f"[ERROR] {self.name}: {self.condition2} rel_sus contains NaN AFTER update!")
            print(f"[CRITICAL INFO] {self.name}: rel_sus first 10 values AFTER update: {cond2_obj.rel_sus.raw[:10]}")

# Function to read interaction data
def read_interactions(datafile=None):
    """
    Reads interaction data from a CSV file.
    Automatically creates the Connectors based on relative risk.
    """
    if datafile is None:
        datafile = sc.thispath() / '../mighti/data/rel_sus.csv'
    df = pd.read_csv(datafile, index_col=0)

    rel_sus = defaultdict(dict)

    for condition1 in df.index:
        for condition2 in df.columns:
            if condition1 != condition2:
                value = df.at[condition1, condition2]
                if not pd.isna(value):
                    rel_sus[condition1][condition2] = value

    return rel_sus