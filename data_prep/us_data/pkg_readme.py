"""
US/NYC data preparation utilities.

This folder mirrors the role of the raw WPP inputs (now under `raw_data/wpp_data/`) but is intended for US-specific sources
(e.g., AIDSVu, CDC WONDER/NCHS, BRFSS, ACS/Census, NYC DOHMH).

Network downloads are intentionally not implemented here; instead, we standardize
local "drop-in" raw CSV formats and produce MIGHTI-ready `mighti/data/{region}_*.csv`
files without overwriting any existing outputs unless explicitly requested.
"""

