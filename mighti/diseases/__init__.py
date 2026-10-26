"""
MIGHTI's health conditions: the base classes (`base_disease`) and one class per condition (`conditions`)
"""
from .base_disease import (
    get_disease_parameters,
    ChronicDisease,
    AcuteDisease,
    RemittingDisease,
    GenericSIS,
    GenericSIR,
    AcuteSurgicalDisease,
    NonAcquiredDisease,
    StaticCondition,
)
from .conditions import *
