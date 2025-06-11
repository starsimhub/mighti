# MIGHTI Test Suite

This folder contains automated tests for the MIGHTI simulation framework. The purpose of these tests is to ensure correctness, stability, and expected behavior of disease modules, simulation components, and analysis tools within MIGHTI.

Each test file serves one or more of the following purposes:
	•	Unit tests: Validate internal logic of specific components (e.g., disease state transitions, parameter handling).
	•	Integration tests: Ensure that multiple components (e.g., diseases, networks, interventions) work together correctly in simulation runs.
	•	Regression/behavioral tests: Detect changes in core behavior over time using predefined scenarios or known outputs.

This structure mirrors the Starsim and STISim frameworks, allowing for modular validation as the MIGHTI codebase expands. All tests are designed to run locally with minimal dependencies, and plots or outputs can be toggled via do_plot.

Test files use either pytest or standalone script style (if __name__ == '__main__') depending on the complexity and target behavior being tested.

| File Name              | Purpose                                                       | Test Type      |
|------------------------|---------------------------------------------------------------|----------------|
| `test_sim.py`          | Verify basic simulation runs, including disease progression   | Integration    |
| `test_disease.py`      | Validates disease state logic                                 | Unit / Scenario |
| `test_connector.py`  | Validates disease interactions  (e.g., HIV-AlcoholUseDisorder connectors)      | Unit / Scenario |
| `test_interventions.py` | Validate application, timing, and effect of interventions    | Unit / Scenario |
| `test_prevalence.py`   | Test loading, interpolation, and application of prevalence    | Unit           |
| `test_life_expectancy.py` | Validate LE and mortality calculations from death outputs   | Unit           |
| `test_calibration.py`  | Ensure calibration machinery runs and produces valid outputs  | Integration    |


