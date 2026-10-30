# Changelog

All notable changes to this project are documented in this file.

## Unreleased (Starsim 4)

### Changed
- STIsim's HIV is now declared (states and transitions), and MIGHTI reads and changes it only through those: competing-risks mortality (`CompetingRisksDeaths`) subtracts the hazards of each declared disease's transitions to death (`mi.mortality_competing.death_hazard()`, which replaces `hiv_death_hazard()`), rather than calling HIV's methods; `ARTAdherenceDisruptor` is a modifier (`ss.modify`) of HIV's rate of stopping ART early (`rate_art_dropout`, the transition `art_dropout`), rather than writing `ti_stop_art`; and `InterventionAdherenceDisruptor` is a modifier of `effective_art_efficacy`, rather than overwriting the parameter (its `scale_art_efficacy` argument is gone). Both are passed as `connectors=`.
- HIV's parameters are given when it is made (e.g. `sti.HIV(p_aids_death=1.0, effective_art_efficacy=0.9)`): `include_aids_deaths` is `p_aids_death`, and `beta={...}` dicts are `beta_m2f`/`rel_beta_f2m`/`beta_m2c`.
- Regression information: the residual hazard no longer drops to zero for someone due to die of AIDS in the step (a death at a set time has no hazard to subtract).

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
