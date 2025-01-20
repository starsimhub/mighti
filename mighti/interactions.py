import numpy as np
import starsim as ss
import mighti as mi

__all__ = [
    'hiv_obesity', 'hiv_type2diabetes'
]


class hiv_obesity(ss.Connector):
    """ Simple connector to make people with HIV more likely to contract Obesity """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Obesity', requires=[ss.HIV, mi.Obesity])
        self.default_pars(
            rel_sus_hiv_obesity=10,  # People with HIV are 1.2x more likely to acquire Obesity
        )
        self.update_pars(pars, **kwargs)
        return

    def update(self):
        sim = self.sim
        # Apply the increased susceptibility to those with HIV
        sim.diseases.obesity.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_obesity
        return

class hiv_type2diabetes(ss.Connector):
    """ Simple connector to make people with HIV more likely to contract Type 2 Diabetes """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Type2Diabetes', requires=[ss.HIV, mi.Type2Diabetes])
        self.default_pars(
            rel_sus_hiv_type2diabetes=5,  # People with HIV are 10x more likely to acquire Type 2 Diabetes
        )
        self.update_pars(pars, **kwargs)

    def update(self):
        sim = self.sim

        # Log current time step
        print(f"Running update for HIV-Type2Diabetes at time step {sim.ti}")

        # Ensure rel_sus is initialized and reset to default (1.0)
        if sim.diseases.type2diabetes.rel_sus is None or len(sim.diseases.type2diabetes.rel_sus) != len(sim.people):
            print("Reinitializing rel_sus for Type2Diabetes.")
            sim.diseases.type2diabetes.rel_sus = np.ones(len(sim.people))  # Initialize to 1.0
        else:
            sim.diseases.type2diabetes.rel_sus[:] = 1.0  # Reset to default

        # Apply increased susceptibility for HIV-infected individuals
        hiv_infected = sim.people.hiv.infected
        before_update = np.unique(sim.diseases.type2diabetes.rel_sus)
        sim.diseases.type2diabetes.rel_sus[hiv_infected] = self.pars.rel_sus_hiv_type2diabetes
        after_update = np.unique(sim.diseases.type2diabetes.rel_sus)

        # Debugging output
        print(f"rel_sus values before update: {before_update}")
        print(f"Updated rel_sus for HIV-positive individuals: {sim.diseases.type2diabetes.rel_sus[hiv_infected]}")
        print(f"rel_sus values after update: {after_update}")
    # def update(self):
    #     sim = self.sim
    #     if sim.diseases.type2diabetes.rel_sus is None:
    #         print("rel_sus is still None during update.")
    #     else:
    #         print(f"rel_sus is initialized with values: {sim.diseases.type2diabetes.rel_sus}")
    #         sim.diseases.type2diabetes.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_type2diabetes
    #     return
# import starsim as ss
# import mighti as mi
# import pandas as pd
# import sciris as sc
# import numpy as np
# from collections import defaultdict

# # Specify all externally visible classes this file defines
# __all__ = [
#     'hiv_hypertension','hiv_obesity','hiv_type1diabetes','hiv_type2diabetes','hiv_depression',
#      'hiv_alzheimers', 'hiv_cerebrovasculardisease', 'hiv_chronicliverdisease',
#      'hiv_asthma', 'hiv_parkinsons', 'hiv_tobaccouse', 'hiv_alcoholusedisorder', 
#     'hiv_ischemicheartdisease', 'hiv_chronickidneydisease', 'hiv_flu','hiv_hpvvaccination', 
#     'hiv_trafficaccident','hiv_domesticviolence','hiv_ptsd','hiv_hivassociateddimentia',
#     'hiv_viralhepatitis', 'hiv_copd','hiv_hyperlipidemia',
#     'hiv_cervicalcancer','hiv_colorectalcancer', 'hiv_breastcancer', 'hiv_lungcancer', 
#     'hiv_prostatecancer','hiv_othercancer',
#     'GenericNCDConnector','read_interactions'
# ]    


# class hiv_obesity(ss.Connector):
#     """ Connector to increase susceptibility for HIV and Obesity """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Obesity', requires=[ss.HIV, mi.Obesity])
        
#         # Define and update parameters
#         default_pars = {
#             'rel_sus_hiv_obesity': 1.83  # Default relative susceptibility
#         }
#         self.default_pars(**default_pars)
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         if not hasattr(sim.diseases.obesity, 'rel_sus') or sim.diseases.obesity.rel_sus is None:
#             print("Initializing rel_sus for Obesity.")
#             sim.diseases.obesity.rel_sus = np.ones(len(sim.people))  # Default to 1.0
#         # Update susceptibility
#         sim.diseases.obesity.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_obesity
#         print(f"Updated rel_sus for Obesity: {sim.diseases.obesity.rel_sus[sim.people.hiv.infected]}")
#         return
    
# # class hiv_type2diabetes(ss.Connector):
# #     """Connector to make people with HIV more likely to contract Type 2 Diabetes."""
# #     def __init__(self, pars=None, **kwargs):
# #         super().__init__(label='HIV-Type2Diabetes', requires=[ss.HIV, mi.Type2Diabetes])
# #         self.default_pars(rel_sus_hiv_type2diabetes=1.61)
# #         self.update_pars(pars, **kwargs)
# #         return

# #     def update(self):
# #         sim = self.sim
# #         # Ensure rel_sus matches the population size
# #         if not hasattr(sim.diseases.type2diabetes, 'rel_sus') or len(sim.diseases.type2diabetes.rel_sus) != len(sim.people):
# #             print("Initializing or resizing rel_sus for Type2Diabetes.")
# #             sim.diseases.type2diabetes.rel_sus = np.ones(len(sim.people))

# #         # Update rel_sus for HIV-infected individuals
# #         hiv_infected = sim.people.hiv.infected
# #         sim.diseases.type2diabetes.rel_sus[hiv_infected] = self.pars.rel_sus_hiv_type2diabetes
# #         print(f"Updated rel_sus for Type2Diabetes: {sim.diseases.type2diabetes.rel_sus[hiv_infected]}")
# #         return

# class hiv_type2diabetes(ss.Connector):
#     """Connector to make people with HIV more likely to contract Type 2 Diabetes."""
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Type2Diabetes', requires=[ss.HIV, mi.Type2Diabetes])
#         self.default_pars(rel_sus_hiv_type2diabetes=1.61)
#         self.update_pars(pars, **kwargs)
#         if self.pars.rel_sus_hiv_type2diabetes <= 0:
#             raise ValueError("rel_sus_hiv_type2diabetes must be greater than 0.")
#         return

#     def update(self):
#         sim = self.sim
#         rel_sus = ensure_rel_sus(sim.diseases.type2diabetes, sim)
#         hiv_infected_indices = np.where(sim.people.hiv.infected)[0]
#         rel_sus[hiv_infected_indices] = self.pars.rel_sus_hiv_type2diabetes
#         print(f"Updated rel_sus for Type2Diabetes: {rel_sus[hiv_infected_indices]}")
#         print(f"Number of HIV-infected individuals updated: {len(hiv_infected_indices)}")
#         return

# class hiv_type1diabetes(ss.Connector):
#     """ Simple connector to make people with HIV more likely to contract Type 1 Diabetes """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Type1Diabetes', requires=[ss.HIV, mi.Type1Diabetes])
#         self.default_pars(
#             rel_sus_hiv_type1diabetes=1.5,  # People with HIV are 1.5x more likely to acquire Type 1 Diabetes
#         )
#         self.update_pars(pars, **kwargs)

#         print(f"Parameter rel_sus_hiv_type1diabetes: {self.pars.rel_sus_hiv_type1diabetes}")
#         return

#     def update(self):
#         sim = self.sim
#         # Apply the increased susceptibility to those with HIV
#         sim.diseases.type1diabetes.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_type1diabetes
#         return



# class hiv_viralhepatitis(ss.Connector):
#     """ Simple connector to make people with HIV more likely to contract Hypertension """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-ViralHepatitis', requires=[ss.HIV, mi.ViralHepatitis])
#         self.default_pars(
#             rel_sus_hiv_viralhepatitis=1.3,  # People with HIV are 1.3x more likely to acquire Hypertension
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         # Apply the increased susceptibility to those with HIV
#         sim.diseases.viralhepatitis.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_viralhepatitis
#         return

# class hiv_copd(ss.Connector):
#     """ Simple connector to make people with HIV more likely to contract Hypertension """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-COPD', requires=[ss.HIV, mi.COPD])
#         self.default_pars(
#             rel_sus_hiv_copd=1.3,  # People with HIV are 1.3x more likely to acquire Hypertension
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         # Apply the increased susceptibility to those with HIV
#         sim.diseases.copd.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_copd
#         return


# class hiv_hyperlipidemia(ss.Connector):
#     """ Simple connector to make people with HIV more likely to contract Hyperlipidemia """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Hyperlipidemia', requires=[ss.HIV, mi.Hyperlipidemia])
#         self.default_pars(
#             rel_sus_hiv_hyperlipidemia=1.3,  # People with HIV are 1.3x more likely to acquire Hypertension
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         # Apply the increased susceptibility to those with HIV
#         sim.diseases.hyperlipidemia.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_hyperlipidemia
#         return


# class hiv_ptsd(ss.Connector):
#     """ Simple connector to make people with HIV more likely to contract PTSD """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Hypertension', requires=[ss.HIV, mi.PTSD])
#         self.default_pars(
#             rel_sus_hiv_ptsd=1.3,  # People with HIV are 1.3x more likely to acquire Hypertension
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         # Apply the increased susceptibility to those with HIV
#         sim.diseases.ptsd.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_ptsd
#         return


# class hiv_hivassociateddimentia(ss.Connector):
#     """ Simple connector to make people with HIV more likely to contract Dimentia """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-HIVAssociatedDimentia', requires=[ss.HIV, mi.HIVAssociatedDementia])
#         self.default_pars(
#             rel_sus_hiv_hivassociateddimentia=1.3,  # People with HIV are 1.3x more likely to acquire Hypertension
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         # Apply the increased susceptibility to those with HIV
#         sim.diseases.hivassociateddimentia.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_hiv_hivassociateddimentia
#         return



# class hiv_hypertension(ss.Connector):
#     """ Simple connector to make people with HIV more likely to contract Hypertension """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Hypertension', requires=[ss.HIV, mi.Hypertension])
#         self.default_pars(
#             rel_sus_hiv_hypertension=1.3,  # People with HIV are 1.3x more likely to acquire Hypertension
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         # Apply the increased susceptibility to those with HIV
#         sim.diseases.hypertension.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_hypertension
#         return




# class hiv_depression(ss.Connector):
#     """ Simple connector to make people with HIV more likely to contract Depression """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Depression', requires=[ss.HIV, mi.Depression])  # Use ss.HIV class
#         self.default_pars(
#             rel_sus_hiv_depression=2,  # People with HIV are 2x more likely to acquire Depression
#         )
#         self.update_pars(pars, **kwargs)
#         return
    
#     def update(self):
#         sim = self.sim
#         # Apply the increased susceptibility to those with HIV
#         sim.diseases.depression.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_depression
#         return


# class hiv_trafficaccident(ss.Connector):
#     """ Connector to make people with HIV more likely to suffer from accidents """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-TrafficAccident', requires=[ss.HIV, mi.TrafficAccident])
#         self.default_pars(rel_sus_hiv_trafficaccident=1.1)
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         # Apply the increased susceptibility to those with HIV
#         sim.diseases.trafficaccident.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_trafficaccident
#         return


# class hiv_alzheimers(ss.Connector):
#     """ Connector to make people with HIV more likely to contract Alzheimer's """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Alzheimers', requires=[ss.HIV, mi.Alzheimers])
#         self.default_pars(rel_sus_hiv_alzheimers=1.1)
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         # Apply the increased susceptibility to those with HIV
#         sim.diseases.alzheimers.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_alzheimers
#         return


# class hiv_domesticviolence(ss.Connector):
#     """ Connector for people with HIV more likely to be involved in Assault """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-DomesticViolence', requires=[ss.HIV, mi.DomesticViolence])
#         self.default_pars(rel_sus_hiv_domesticviolence=1.1)
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.domesticviolence.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_domesticviolence
#         return
    

# class hiv_cerebrovasculardisease(ss.Connector):
#     """ Connector for people with HIV more likely to acquire Cerebrovascular diseases """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-CerebrovascularDisease', requires=[ss.HIV, mi.CerebrovascularDisease])
#         self.default_pars(rel_sus_hiv_cerebrovasculardisease=1.2)
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.cerebrovasculardisease.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_cerebrovasculardisease
#         return


# class hiv_chronicliverdisease(ss.Connector):
#     """ Connector for people with HIV more likely to acquire Chronic Liver Disease """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-ChronicLiverDisease', requires=[ss.HIV, mi.ChronicLiverDisease])
#         self.default_pars(rel_sus_hiv_liver=1.2)
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.chronicliverdisease.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_chronicliverdisease
#         return


# class hiv_asthma(ss.Connector):
#     """ Connector for people with HIV more likely to acquire Asthma"""
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Asthma', requires=[ss.HIV, mi.Asthma])
#         self.default_pars(rel_sus_hiv_resp=1.2)
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.asthma.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_resp
#         return
    

# class hiv_ischemicheartdisease(ss.Connector):
#     """ Connector for people with HIV more likely to acquire Heart Disease """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-IschemicHeartDisease', requires=[ss.HIV, mi.IschemicHeartDisease])
#         self.default_pars(rel_sus_hiv_ischemicheartdisease=1.3)
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.ischemicheartdisease.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_ischemicheartdisease
#         return


# class hiv_chronickidneydisease(ss.Connector):
#     """ Connector for people with HIV more likely to acquire Chronic Kidney Disease """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-ChronicKidneyDisease', requires=[ss.HIV, mi.ChronicKidneyDisease])
#         self.default_pars(rel_sus_hiv_chronickidneydisease=1.3)
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.chronickidneydisease.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_chronickidneydisease
#         return


# class hiv_flu(ss.Connector):
#     """ Connector for people with HIV more likely to acquire Influenza """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Flu', requires=[ss.HIV, mi.Flu])
#         self.default_pars(rel_sus_hiv_flu=1.2)
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.flu.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_flu
#         return


# class hiv_hpvvaccination(ss.Connector):
#     """ Connector for people with HIV more likely to acquire HPV """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-HPVVaccination', requires=[ss.HIV, mi.HPVVaccination])
#         self.default_pars(rel_sus_hiv_hpvvaccination=1.5)
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.hpvvaccination.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_hpvvaccination
#         return


# class hiv_parkinsons(ss.Connector):
#     """ Connector for people with HIV more likely to acquire Parkinson's disease """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Parkinsons', requires=[ss.HIV, mi.Parkinsons])
#         self.default_pars(rel_sus_hiv_parkinsons=1.3)
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.parkinsons.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_parkinsons
#         return


# class hiv_tobaccouse(ss.Connector):
#     """ Connector for people with HIV more likely to be infected by smoking """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-TobaccoUse', requires=[ss.HIV, mi.TobaccoUse])
#         self.default_pars(rel_sus_hiv_tobaccouse=1.5)
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.tobaccouse.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_tobaccouse
#         return


# class hiv_alcoholusedisorder(ss.Connector):
#     """ Connector for people with HIV more likely to suffer from alcohol-related conditions """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-AlcoholUseDisorder', requires=[ss.HIV, mi.AlcoholUseDisorder])
#         self.default_pars(rel_sus_hiv_alcoholusedisorder=1.4)
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.alcoholusedisorder.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_alcoholusedisorder
#         return


    
# class hiv_cervicalcancer(ss.Connector):
#     """ Connector for people with HIV more likely to acquire Cervical Cancer """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Cervical', requires=[ss.HIV, mi.CervicalCancer])
#         self.default_pars(
#             rel_sus_hiv_cervicalcancer=1.3  # People with HIV are 1.3x more likely to acquire Cervical Cancer
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.cervicalcancer.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_cervicalcancer
#         return

    
# class hiv_colorectalcancer(ss.Connector):
#     """ Connector for people with HIV more likely to acquire Colorectal Cancer """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Colorectal', requires=[ss.HIV, mi.ColorectalCancer])
#         self.default_pars(
#             rel_sus_hiv_colorectalcancer=1.3  # People with HIV are 1.3x more likely to acquire Colorectal Cancer
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.colorectalcancer.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_colorectalcancer
#         return


# class hiv_breastcancer(ss.Connector):
#     """ Connector for people with HIV more likely to acquire Breast Cancer """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Breast', requires=[ss.HIV, mi.BreastCancer])
#         self.default_pars(
#             rel_sus_hiv_breastcancer=1.3  # People with HIV are 1.3x more likely to acquire Breast Cancer
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.breastcancer.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_breastcancer
#         return


# class hiv_lungcancer(ss.Connector):
#     """ Connector for people with HIV more likely to acquire Lung Cancer """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Lung', requires=[ss.HIV, mi.LungCancer])
#         self.default_pars(
#             rel_sus_hiv_lungcancer=1.4  # People with HIV are 1.4x more likely to acquire Lung Cancer
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.lungcancer.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_lungcancer
#         return


# class hiv_prostatecancer(ss.Connector):
#     """ Connector for people with HIV more likely to acquire Prostate Cancer """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Prostate', requires=[ss.HIV, mi.ProstateCancer])
#         self.default_pars(
#             rel_sus_hiv_prostatecancer=1.3  # People with HIV are 1.3x more likely to acquire Prostate Cancer
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.prostatecancer.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_prostatecancer
#         return


# class hiv_othercancer(ss.Connector):
#     """ Connector for people with HIV more likely to acquire Other Cancers """
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(label='HIV-Other', requires=[ss.HIV, mi.OtherCancer])
#         self.default_pars(
#             rel_sus_hiv_othercancer=1.3  # People with HIV are 1.3x more likely to acquire Other Cancers
#         )
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         sim.diseases.othercancer.rel_sus[sim.people.hiv.infected] = self.pars.rel_sus_hiv_othercancer
#         return

    
# # Functions to read in datafiles
# def read_interactions(datafile=None):
#     """
#     Read in datafile with risk/condition interactions.
#     At some point, this can be adjusted to automatically create the Connectors.
#     """
#     if datafile is None:
#         datafile = sc.thispath() / '../mighti/data/rel_sus.csv'
#     df = pd.read_csv(datafile)

#     rel_sus = defaultdict(dict)
    

#     for cond in df.has_condition.unique():
#         conddf = df.loc[df.has_condition == cond]
#         conddf.reset_index(inplace=True, drop=True)
#         interacting_conds = conddf.columns[~conddf.isna().any()].tolist()
#         interacting_conds.remove('has_condition')
#         for interacting_cond in interacting_conds:
#             rel_sus[interacting_cond][cond] = conddf.loc[0, interacting_cond]

#     return rel_sus


# class GenericNCDConnector(ss.Connector):
#     """
#     A generic connector to model interactions between two diseases.
#     This class adjusts susceptibility based on relative risk from the interaction matrix.
#     """
#     def __init__(self, condition1, condition2, relative_risk, pars=None, **kwargs):
#         """
#         Initialize the connector with the two interacting conditions and their relative risk.
#         """
#         label = f'{condition1}-{condition2}'  # Create a unique label for each connector
#         # print(f"Creating connector with label: {label}")  # For debugging purposes
#         super().__init__(label=label, requires=[getattr(mi, condition1), getattr(mi, condition2)])
#         self.condition1 = condition1
#         self.condition2 = condition2
#         self.relative_risk = relative_risk
#         self.default_pars(rel_sus=relative_risk)  # Use the passed relative risk
#         self.update_pars(pars, **kwargs)
#         return

#     def update(self):
#         sim = self.sim
#         cond1_obj = getattr(sim.diseases, self.condition1.lower())  # Get the first disease object
        
#         # Apply the relative risk from condition1 to condition2's susceptible individuals
#         sim.diseases[self.condition2.lower()].rel_sus[cond1_obj.infected] = self.pars.rel_sus
#         return