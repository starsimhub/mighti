# MIGHTI Test Suite

This folder contains automated tests for the MIGHTI simulation framework. Tests are organized by module and purpose, and follow standard `pytest` conventions.

| File Name              | Purpose                                                       | Test Type      |
|------------------------|---------------------------------------------------------------|----------------|
| `test_sim.py`          | Verify basic simulation runs, including disease progression   | Integration    |
| `test_hiv.py`          | Validate HIV-specific logic (e.g., CD4 dynamics, ART timing)  | Unit / Scenario |
| `test_prevalence.py`   | Test loading, interpolation, and application of prevalence    | Unit           |
| `test_connector.py`    | Test disease-disease interactions (e.g., HIV-NCD connectors)  | Unit           |
| `test_life_expectancy.py` | Validate LE and mortality calculations from death outputs   | Unit           |
| `test_calibration.py`  | Ensure calibration machinery runs and produces valid outputs  | Integration    |

