import os
import logging


logger = logging.getLogger(__name__)

# Project root (repo root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# -------------------------------------------------------------------------
# Define base path for data output
#
# IMPORTANT: We intentionally do NOT import `mighti` here.
# Importing `mighti` pulls in `starsim` which may trigger heavy numba/matplotlib
# initialization and can fail in some environments. Instead, we write processed
# outputs to the repo-local `data/processed/` folder.
# -------------------------------------------------------------------------
_ENV_DATA_DIR = os.environ.get("MIGHTI_DATA_DIR")
if _ENV_DATA_DIR:
    DATA_DIR = os.path.abspath(os.path.expanduser(_ENV_DATA_DIR))
else:
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def ensure_data_dir():
    """Create and return the `data/processed/` output directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def data_path(filename):
    """Return full path for a file inside `data/processed/`."""
    ensure_data_dir()
    return os.path.join(DATA_DIR, filename)


RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data/raw")
WPP_DATA = os.path.join(PROJECT_ROOT, "data/raw", "wpp_data")


def wpp_path(filename):
    """Return full path for files in `data/raw/wpp_data/`."""
    return os.path.join(WPP_DATA, filename)


DISEASE_DATA_DIR = os.path.join(RAW_DATA_DIR, "disease_data")


def disease_data_path(filename):
    """Return full path for files in `data/raw/disease_data/`."""
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
    # Use AcuteHepatitis as the modeled hepatitis condition (collapse hepatitis-related causes)
    "Acute hepatitis": "AcuteHepatitis",
    "Hepatitis B": "AcuteHepatitis",
    "Hepatitis C": "AcuteHepatitis",
    "Interpersonal violence": "InterpersonalViolence",
    "Self-harm": "SelfHarm",
    "Road injuries": "RoadInjuries",
    "Cirrhosis and other chronic liver diseases": "ChronicLiverDisease",
    "Asthma": "Asthma",
    "Chronic obstructive pulmonary disease": "COPD",
    # IHME exports are inconsistent about curly vs straight apostrophes
    "Alzheimer’s disease and other dementias": "AlzheimersDisease",
    "Alzheimer's disease and other dementias": "AlzheimersDisease",
    "Parkinson’s disease": "ParkinsonsDisease",
    "Parkinson's disease": "ParkinsonsDisease",
    "Anxiety disorders": "AnxietyDisorder",
    "Drug use disorders": "DrugUseDisorder",
    # Collapse opioid use disorder into overall drug use disorder for modeled condition
    "Opioid use disorders": "DrugUseDisorder",
    # Sub-causes that we collapse into the modeled DrugUseDisorder category
    "Cocaine use disorders": "DrugUseDisorder",
    "Amphetamine use disorders": "DrugUseDisorder",
    "Neonatal encephalopathy due to birth asphyxia and trauma": "NeonatalEncephalopathy",
    "Neonatal preterm birth": "NeonatalPretermBirth",
    "Neonatal sepsis and other neonatal infections": "NeonatalSepsis",
    "Hemolytic disease and other neonatal jaundice": "NeonatalJaundice",
    "Neural tube defects": "NeuralTubeDefects",
    "Congenital heart anomalies": "CongenitalHeartAnomalies",
    "Congenital musculoskeletal and limb anomalies": "CongenitalMusculoskeletal",
    "Digestive congenital anomalies": "DigestiveCongenitalAnomalies",
    "Down syndrome": "DownSyndrome",
    "Other chromosomal abnormalities": "ChromosomalAbnormalities",
    "Diarrheal diseases": "DiarrhealDiseases",
    "Esophageal cancer": "EsophagealCancer",
    "Protein-energy malnutrition": "ProteinEnergyMalnutrition",
    # -------------------------------------------------------------
    # Additions for Eswatini COD + prevalence alignment (2026-02)
    # -------------------------------------------------------------
    "Tuberculosis": "Tuberculosis",
    "Lower respiratory infections": "LowerRespiratoryInfections",
    "COVID-19": "COVID19",
    "Maternal hemorrhage": "MaternalConditions",
    "Maternal sepsis and other maternal infections": "MaternalConditions",
    "Maternal hypertensive disorders": "MaternalConditions",
    "Maternal obstructed labor and uterine rupture": "MaternalConditions",
    "Maternal abortion and miscarriage": "MaternalConditions",
    "Maternal disorders": "MaternalConditions",
    "HIV/AIDS": "HIV",
    "Neonatal disorders": "NeonatalEncephalopathy",
}

