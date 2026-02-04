import os
import logging

import mighti  # ensures we know the installed package path


logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Define base path for data output — always points to active MIGHTI install
# -------------------------------------------------------------------------
MIGHTI_BASE = os.path.dirname(mighti.__file__)
DATA_DIR = os.path.join(MIGHTI_BASE, "data")


def ensure_data_dir() -> str:
    """Create and return the `mighti/data/` output directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def data_path(filename: str) -> str:
    """Return full path for a file inside `mighti/data/`."""
    ensure_data_dir()
    return os.path.join(DATA_DIR, filename)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
WPP_DATA = os.path.join(PROJECT_ROOT, "raw_data", "wpp_data")


def wpp_path(filename: str) -> str:
    """Return full path for files in `raw_data/wpp_data/`."""
    return os.path.join(WPP_DATA, filename)


DISEASE_DATA_DIR = os.path.join(RAW_DATA_DIR, "disease_data")


def disease_data_path(filename: str) -> str:
    """Return full path for files in `raw_data/disease_data/`."""
    return os.path.join(DISEASE_DATA_DIR, filename)


# Mapping from raw “cause” strings (GBD/WPP inputs) to MIGHTI condition names
cause_map = {
    "Diabetes mellitus type 1": "Type1Diabetes",
    "Diabetes mellitus type 2": "Type2Diabetes",
    "Hypertension": "Hypertension",
    "High body-mass index": "Obesity",
    "Cardiovascular diseases": "CardiovascularDiseases",
    "Chronic kidney disease": "ChronicKidneyDisease",
    "High LDL cholesterol": "Hyperlipidemia",
    "Cervical cancer": "CervicalCancer",
    "Colon and rectum cancer": "ColorectalCancer",
    "Breast cancer": "BreastCancer",
    "Tracheal, bronchus, and lung cancer": "LungCancer",
    "Prostate cancer": "ProstateCancer",
    "Alcohol use disorders": "AlcoholUseDisorder",
    "Tobacco use": "TobaccoUse",
    "Dementia": "Dementia",
    "Post-traumatic stress disorder": "PTSD",
    "Bipolar disorder": "BipolarDisorder",
    "Major depressive disorder": "MajorDepressiveDisorder",
    "Human papillomavirus infection": "HPV",
    "Influenza and pneumonia": "Flu",
    "Hepatitis B": "ViralHepatitis",
    "Hepatitis C": "ViralHepatitis",
    "Interpersonal violence": "InterpersonalViolence",
    "Self-harm": "SelfHarm",
    "Road injuries": "RoadInjuries",
    "Cirrhosis and other chronic liver diseases": "ChronicLiverDisease",
    "Asthma": "Asthma",
    "Chronic obstructive pulmonary disease": "COPD",
    "Alzheimer’s disease and other dementias": "AlzheimersDisease",
    "Parkinson’s disease": "ParkinsonsDisease",
    "Neonatal encephalopathy due to birth asphyxia and trauma": "NeonatalEncephalopathy",
    "Neonatal preterm birth": "NeonatalPretermBirth",
    "Neonatal sepsis and other neonatal infections": "NeonatalSepsis",
    "Neural tube defects": "NeuralTubeDefects",
    "Congenital heart anomalies": "CongenitalHeartAnomalies",
    "Congenital musculoskeletal and limb anomalies": "CongenitalMusculoskeletal",
    "Digestive congenital anomalies": "DigestiveCongenitalAnomalies",
    "Down syndrome": "DownSyndrome",
    "Other chromosomal abnormalities": "ChromosomalAbnormalities",
    "Diarrheal disease": "DiarrhealDisease",
    "Esophageal cancer": "EsophagealCancer",
    "Protein-energy malnutrition": "ProteinEnergyMalnutrition",
}

