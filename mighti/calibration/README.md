# Calibration diagnostics

`calibration_diseases.py` fits each condition's annual probability of acquisition by sex (and, with cause-specific death rates, its annual probability of death) with `sim.fit()` (Optuna), and saves Optuna diagnostics for each calibrated disease. `calibration_hiv.py` fits HIV's transmission parameters the same way. The parameters to fit are given bounds with `ss.par(value, bounds=[low, high])`; the goodness of fit is a function of the sim.

## Output location

For a run on date `YYYYMMDD`, diagnostics are saved under:

`mighti/calibration/results/calibration_<region>_<YYYYMMDD>/diagnostics/<DiseaseName>/`

Example:

`mighti/calibration/results/calibration_eswatini_20260309/diagnostics/Type1Diabetes/`

Typical files include:

- `plot_optimization_history.png`
- `plot_param_importances.png`
- `plot_slice.png`
- `plot_timeline.png`
- `plot_edf.png`

## How to interpret

- `plot_optimization_history`: objective value by trial; should trend downward for stable improvement.
- `plot_param_importances`: relative influence of each calibrated parameter on objective.
- `plot_slice`: objective vs parameter value; useful to see boundary-hitting or flat regions.
- `plot_timeline`: trial timing; helps spot slow/hanging trials.
- `plot_edf`: cumulative distribution of objective values; compares search efficiency.

## Practical checks

- If best values repeatedly sit at lower/upper bounds, treat as boundary-hitting and test bound sensitivity.
- If history is flat/erratic, increase `total_trials` or refine objective/parameter bounds.
- If diagnostics are missing for a disease, check logs for skip reasons (missing observed prevalence columns or no signal).
