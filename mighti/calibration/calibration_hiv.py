"""
Calibrate HIV betas against observed prevalence (by sex and age bins).

This script used to be hard-coded for Eswatini. It now supports:
- changing region and input CSV paths via CLI flags
- changing sim settings (n_agents, years, seed, trials) without editing code

The betas to fit are given bounds in `make_sim()` (`ss.par(value, bounds=[low, high])`), and `sim.fit()`
fits them to the observed prevalence by sex and age with the goodness of fit of `make_gof()`.

Example
-------
python -m mighti.calibration.calibration_hiv --region eswatini --start 1990 --stop 2023 --n-agents 10000 --trials 200
"""


import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from mighti.util.paths import get_data_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibConfig:
    region: str
    data_dir: Path
    prevalence_hiv_csv: Path
    asfr_csv: Path
    mortality_csv: Path
    n_agents: int
    total_pop: int
    start: int
    stop: int
    dt: float
    init_prev: float
    trials: int
    sampler_seed: int
    keep_db: bool
    obs_units: str


def _resolve_default_paths(region, data_dir):
    """Resolve default input file paths for a region."""
    return {
        "prevalence_hiv_csv": data_dir / f"{region}_prevalence_hiv.csv",
        "asfr_csv": data_dir / f"{region}_asfr.csv",
        "mortality_csv": data_dir / f"{region}_mortality_rates.csv",
    }


def _require_exists(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}. "
            "Pass an explicit path via CLI or place the expected file under --data-dir."
        )
    return path


def make_sim(cfg):
    """ The sim to calibrate: HIV, whose betas have bounds (see `ss.par`), with the HIV prevalence analyzer """
    import pandas as pd
    import starsim as ss
    import stisim as sti
    import mighti as mi

    hiv = sti.HIV(
        beta_m2f = ss.par(0.03, bounds=[0.001, 0.10]), # HIV transmission parameter
        beta_m2c = ss.par(0.001, bounds=[0.0001, 0.1]), # Mother-to-child transmission
        init_prev = float(cfg.init_prev),
    )
    pregnancy = ss.Pregnancy(fertility_rate=pd.read_csv(cfg.asfr_csv))
    death = ss.Deaths(death_rate=pd.read_csv(cfg.mortality_csv), rate_units=1)
    prevalence_analyzer = mi.analyzers.PrevalenceAnalyzer_HIV(diseases=["HIV"]) # The results calibrated to

    sim = ss.Sim(
        dt=ss.years(cfg.dt),
        n_agents=int(cfg.n_agents),
        total_pop=int(cfg.total_pop),
        start=int(cfg.start),
        stop=int(cfg.stop),
        diseases=hiv,
        networks=[sti.StructuredSexual(), ss.MaternalNet()],
        demographics=[pregnancy, death],
        analyzers=[prevalence_analyzer],
        rand_seed=cfg.sampler_seed,
        label=f"HIV calibration ({cfg.region})",
    )
    return sim


def load_observed(cfg):
    """ The observed HIV prevalence by age and sex (columns Year, Age, HIV_female, HIV_male), as fractions """
    import pandas as pd
    obs = pd.read_csv(cfg.prevalence_hiv_csv)
    for col in ["Age", "Year", "HIV_male", "HIV_female"]:
        obs[col] = pd.to_numeric(obs[col], errors="coerce")
    obs = obs.dropna(subset=["Age", "Year"]).astype(dict(Age=int, Year=int))
    units = cfg.obs_units
    if units == "percent" or (units == "auto" and obs[["HIV_male", "HIV_female"]].max().max() > 1):
        obs[["HIV_male", "HIV_female"]] /= 100
    return obs


def make_gof(obs):
    """
    The goodness of fit of a sim: the mean absolute difference between the simulated and observed HIV prevalence,
    by sex, age group (the analyzer's), and year
    """
    import numpy as np
    import pandas as pd

    def gof(sim):
        an = sim.analyzers.prevalence_analyzer_hiv
        res = an.results
        years = np.floor(np.asarray(sim.t.yearvec)).astype(int)
        errors = []
        for i, (age_low, _) in enumerate(an.age_bins):
            sim_df = pd.DataFrame(dict(Year=years, sim_female=res[f'hiv_prev_female_{i}'][:], sim_male=res[f'hiv_prev_male_{i}'][:])).groupby('Year').last()
            merged = obs[obs['Age'] == age_low].merge(sim_df, left_on='Year', right_index=True)
            for sex in ['female', 'male']:
                errors.append(np.abs(merged[f'sim_{sex}'] - merged[f'HIV_{sex}']).dropna())
        errors = pd.concat(errors)
        if not len(errors):
            raise ValueError("No observed HIV prevalence overlaps the simulated years and age groups")
        return float(errors.mean())

    return gof


def run_calib(cfg):
    """ Fit HIV's betas to the observed HIV prevalence by age and sex (with Optuna, since the sim is stochastic); returns an `ss.fit.Fit` """
    sim = make_sim(cfg)
    gof = make_gof(load_observed(cfg))
    kwargs = dict(storage=f'calib_hiv_{cfg.region}.log') if cfg.keep_db else {}
    fit = sim.fit(gof=gof, method='optuna', n_trials=int(cfg.trials), rand_seed=int(cfg.sampler_seed), **kwargs)
    print(fit)
    return fit


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate HIV transmission betas against observed prevalence.")
    parser.add_argument("--region", default="eswatini", help="Region name (used to build default file names).")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing region CSV inputs (defaults to MIGHTI_DATA_DIR or repo data/processed).",
    )
    parser.add_argument("--prevalence-hiv-csv", default=None, help="Observed HIV prevalence CSV (Age/Year/HIV_female/HIV_male).")
    parser.add_argument("--asfr-csv", default=None, help="ASFR CSV for pregnancy module.")
    parser.add_argument("--mortality-csv", default=None, help="Mortality rates CSV for deaths module.")
    parser.add_argument("--n-agents", type=int, default=10_000, help="Number of agents.")
    parser.add_argument("--total-pop", type=int, default=9_980_999, help="Total population scaling (for analyzers).")
    parser.add_argument("--start", type=int, default=1990, help="Start year.")
    parser.add_argument("--stop", type=int, default=2023, help="Stop year.")
    parser.add_argument("--dt", type=float, default=1.0, help="Timestep size (years).")
    parser.add_argument("--init-prev", type=float, default=0.15, help="Initial HIV prevalence used by the HIV module (not the observed data).")
    parser.add_argument("--trials", type=int, default=100, help="Total Optuna trials.")
    parser.add_argument("--sampler-seed", type=int, default=12345, help="Seed for the Optuna sampler.")
    parser.add_argument("--keep-db", action="store_true", help="Keep the trials in a file (calib_hiv_<region>.log), so that running again resumes the fit.")
    parser.add_argument(
        "--obs-units",
        default="auto",
        choices=["auto", "fraction", "percent"],
        help="Units of observed HIV prevalence columns. Use 'percent' if values are percents (0-100 or 0-1 meaning percent).",
    )
    args = parser.parse_args(argv)

    data_dir = Path(get_data_dir(args.data_dir))
    defaults = _resolve_default_paths(args.region, data_dir)

    prevalence_hiv_csv = Path(args.prevalence_hiv_csv).expanduser() if args.prevalence_hiv_csv else defaults["prevalence_hiv_csv"]
    asfr_csv = Path(args.asfr_csv).expanduser() if args.asfr_csv else defaults["asfr_csv"]
    mortality_csv = Path(args.mortality_csv).expanduser() if args.mortality_csv else defaults["mortality_csv"]

    return CalibConfig(
        region=str(args.region),
        data_dir=data_dir,
        prevalence_hiv_csv=_require_exists(prevalence_hiv_csv, "prevalence-hiv-csv"),
        asfr_csv=_require_exists(asfr_csv, "asfr-csv"),
        mortality_csv=_require_exists(mortality_csv, "mortality-csv"),
        n_agents=int(args.n_agents),
        total_pop=int(args.total_pop),
        start=int(args.start),
        stop=int(args.stop),
        dt=float(args.dt),
        init_prev=float(args.init_prev),
        n_trials=int(args.trials),
        sampler_seed=int(args.sampler_seed),
        keep_db=bool(args.keep_db),
        obs_units=str(args.obs_units),
    )


#%% Run as a script
if __name__ == '__main__':

    import sciris as sc

    T = sc.tic()
    cfg = _parse_args()
    fit = run_calib(cfg=cfg)
    sc.toc(T)
    print('Done.')
