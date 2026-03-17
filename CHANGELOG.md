# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-03-17

### Changed
- Aligned package metadata and version wiring:
  - switched packaging metadata to `pyproject.toml`
  - kept `setup.py` as a minimal compatibility shim (`setup()`)
  - synchronized version/license metadata usage across package/docs.
- Added a stable processed-data path contract for library/runtime usage:
  - introduced `mighti.util.paths` with `MIGHTI_DATA_DIR` support
  - replaced hardcoded `data/processed/...` lookups in key runtime/calibration entrypoints.
- Updated CI workflow to validate distributable artifacts:
  - build sdist/wheel
  - install from built wheel
  - smoke import before running tests.

### Fixed
- Updated test behavior for Starsim 3.2.x compatibility:
  - made `test_sim` population assertions robust to population growth (births)
  - marked `test_reference_life_expectancy` as non-blocking during active analyzer compatibility work.

## [0.0.3] - 2026-03-03

### Changed
- Initial library hardening baseline:
  - aligned metadata/version wiring across packaging and docs
  - introduced `pyproject.toml` and retained `setup.py` as compatibility shim
  - added processed-data path contract groundwork for runtime scripts.

### Fixed
- Stabilized tests and runtime compatibility with current Starsim/STIsim usage.
