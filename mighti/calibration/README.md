# Calibration Diagnostics

`calibration_diseases.py` now saves Optuna diagnostics for each calibrated disease.

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
