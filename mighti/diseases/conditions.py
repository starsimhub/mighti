"""
The health conditions of MIGHTI: one class per condition, each named as in the parameter CSV file (see `base_disease`)
"""

import starsim as ss
from .base_disease import ChronicDisease, AcuteDisease, RemittingDisease, GenericSIS, GenericSIR, NonAcquiredDisease, StaticCondition

__all__ = ['AcuteHepatitis', 'AlcoholUseDisorder', 'AlzheimersDisease', 'AnxietyDisorder', 'Asthma', 'BipolarDisorder', 'BreastCancer', 'CardiovascularDiseases', 'CervicalCancer', 'ChromosomalAbnormalities', 'ChronicKidneyDisease', 'ChronicLiverDisease', 'ChronicPain', 'ColorectalCancer', 'CongenitalHeartAnomalies', 'CongenitalMusculoskeletal', 'COPD', 'COVID19', 'Dementia', 'DiarrhealDiseases', 'DigestiveCongenitalAnomalies', 'DownSyndrome', 'DrugUseDisorder', 'EsophagealCancer', 'HPV', 'Hyperlipidemia', 'Hypertension', 'Influenza', 'InterpersonalViolence', 'LowerRespiratoryInfections', 'LungCancer', 'MajorDepressiveDisorder', 'MaternalConditions', 'NeonatalEncephalopathy', 'NeonatalJaundice', 'NeonatalPretermBirth', 'NeonatalSepsis', 'NeuralTubeDefects', 'Obesity', 'OpioidUseDisorder', 'ParkinsonsDisease', 'ProstateCancer', 'ProteinEnergyMalnutrition', 'PTSD', 'RoadInjuries', 'SelfHarm', 'TobaccoUse', 'Tuberculosis', 'Type1Diabetes', 'Type2Diabetes']


# Chronic conditions (no remission)
class AlzheimersDisease(ChronicDisease):
    """ Alzheimer's disease """

class Asthma(ChronicDisease):
    """ Asthma """

class BreastCancer(ChronicDisease):
    """ Breast cancer """

class CardiovascularDiseases(ChronicDisease):
    """ Cardiovascular diseases """

class CervicalCancer(ChronicDisease):
    """ Cervical cancer """

class ChronicKidneyDisease(ChronicDisease):
    """ Chronic kidney disease """

class ChronicLiverDisease(ChronicDisease):
    """ Chronic liver disease """

class ChronicPain(ChronicDisease):
    """ Chronic pain """

class ColorectalCancer(ChronicDisease):
    """ Colorectal cancer """

class COPD(ChronicDisease):
    """ COPD """

class Dementia(ChronicDisease):
    """ Dementia """

class EsophagealCancer(ChronicDisease):
    """ Esophageal cancer """

class Hyperlipidemia(ChronicDisease):
    """ Hyperlipidemia """

class Hypertension(ChronicDisease):
    """ Hypertension """

class LungCancer(ChronicDisease):
    """ Lung cancer """

class Obesity(ChronicDisease):
    """ Obesity """

class ParkinsonsDisease(ChronicDisease):
    """ Parkinson's disease """

class ProstateCancer(ChronicDisease):
    """ Prostate cancer """

class Type1Diabetes(ChronicDisease):
    """ Type 1 diabetes """


# Acute conditions (in MIGHTI, the course of a chronic one)
class InterpersonalViolence(AcuteDisease):
    """ Interpersonal violence """

class MaternalConditions(AcuteDisease):
    """ Maternal conditions """

class ProteinEnergyMalnutrition(AcuteDisease):
    """ Protein energy malnutrition """

class RoadInjuries(AcuteDisease):
    """ Road injuries """

class SelfHarm(AcuteDisease):
    """ Self harm """


# Remitting conditions
class AlcoholUseDisorder(RemittingDisease):
    """ Alcohol use disorder """

class AnxietyDisorder(RemittingDisease):
    """ Anxiety disorder """

class BipolarDisorder(RemittingDisease):
    """ Bipolar disorder """

class DrugUseDisorder(RemittingDisease):
    """ Drug use disorder """

class OpioidUseDisorder(RemittingDisease):
    """ Opioid use disorder """

class PTSD(RemittingDisease):
    """ PTSD """

class TobaccoUse(RemittingDisease):
    """ Tobacco use """

class Type2Diabetes(RemittingDisease):
    """ Type 2 diabetes """


# Infections from which people recover to susceptible
class AcuteHepatitis(GenericSIS):
    """ Acute hepatitis """

class COVID19(GenericSIS):
    """ COVID-19 """

class HPV(GenericSIS):
    """ HPV """

class Influenza(GenericSIS):
    """ Influenza """

class LowerRespiratoryInfections(GenericSIS):
    """ Lower respiratory infections """


# Infections from which people recover to immunity
class DiarrhealDiseases(GenericSIR):
    """ Diarrheal diseases """

class Tuberculosis(GenericSIR):
    """ Tuberculosis """


# Conditions present from birth
class CongenitalHeartAnomalies(NonAcquiredDisease):
    """ Congenital heart anomalies """

class CongenitalMusculoskeletal(NonAcquiredDisease):
    """ Congenital musculoskeletal """

class DigestiveCongenitalAnomalies(NonAcquiredDisease):
    """ Digestive congenital anomalies """

class NeonatalEncephalopathy(NonAcquiredDisease):
    """ Neonatal encephalopathy """
    is_neonatal = True

class NeonatalJaundice(NonAcquiredDisease):
    """ Neonatal jaundice """
    is_neonatal = True

class NeonatalPretermBirth(NonAcquiredDisease):
    """ Neonatal preterm birth """
    is_neonatal = True

class NeonatalSepsis(NonAcquiredDisease):
    """ Neonatal sepsis """
    is_neonatal = True

class NeuralTubeDefects(NonAcquiredDisease):
    """ Neural tube defects """


# Lifelong conditions present from birth
class ChromosomalAbnormalities(StaticCondition):
    """ Chromosomal abnormalities """

class DownSyndrome(StaticCondition):
    """ Down syndrome """


class MajorDepressiveDisorder(RemittingDisease):
    """
    Major depressive disorder: a remitting condition whose remission is faster for people who are housed

    Remission is `remission_mult` times `remission_rate`, and `1 + housing_protect` times that for people with
    `neighbourhood_situation` (see `mi.sdoh.NeighbourhoodSituation`); without that module, nobody is housed.
    """
    remission_mult = ss.par(1.0, doc='Multiplier on remission, e.g. for care')
    housing_protect = ss.par(0.5, doc='Relative increase in remission for people who are housed')
    remission = ss.Transition(RemittingDisease.affected, RemittingDisease.susceptible, rate=remission_mult*RemittingDisease.remission_rate*(1 + housing_protect*ss.ppl.get('neighbourhood_situation', 0))) # Nobody is housed unless a module adds the property
