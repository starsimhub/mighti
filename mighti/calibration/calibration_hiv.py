"""
Calibrate HIV betas against observed prevalence (by sex and age bins).

This script used to be hard-coded for Eswatini. It now supports:
- changing region and input CSV paths via CLI flags
- changing sim settings (n_agents, years, seed, trials) without editing code

Example
-------
python mighti_calibration.py --region eswatini --start 1990 --stop 2023 --n-agents 10000 --trials 200
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
    use_stisim_calibration: bool


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
    """Build and initialize a simulation for HIV calibration."""
    import pandas as pd
    import starsim as ss
    import stisim as sti
    import mighti as mi

    hiv = sti.HIV(beta_m2f=0.05, beta_m2c=0.025, init_prev=float(cfg.init_prev))

    fertility_rate = {"fertility_rate": pd.read_csv(cfg.asfr_csv)}
    pregnancy = ss.Pregnancy(pars=fertility_rate)

    death_rates = {"death_rate": pd.read_csv(cfg.mortality_csv), "rate_units": 1}
    death = ss.Deaths(death_rates)

    sexual = sti.StructuredSexual()
    maternal = ss.MaternalNet()

    # Analyzer drives the results we calibrate to
    obs_df = pd.read_csv(cfg.prevalence_hiv_csv)
    prevalence_analyzer = mi.analyzers.PrevalenceAnalyzer_HIV(
        prevalence_data=obs_df,
        diseases=["HIV"],
        label="prevalence_analyzer",  # stable key in sim.results / sim.analyzers
    )

    sim = ss.Sim(
        dt=cfg.dt,
        n_agents=int(cfg.n_agents),
        total_pop=int(cfg.total_pop),
        start=int(cfg.start),
        stop=int(cfg.stop),
        diseases=hiv,
        networks=[sexual, maternal],
        demographics=[pregnancy, death],
        analyzers=[prevalence_analyzer],
        label=f"HIV calibration ({cfg.region})",
    )

    sim.init()

    # Compatibility aliases for STIsim calibration's dot-path result extraction.
    # Some workflows/dataframes may use prefixes like "prevalence.*" or
    # "prevalence_analyzer.*" even though our analyzer key is
    # "prevalence_analyzer_hiv".
    try:
        if hasattr(sim, "results") and "prevalence_analyzer_hiv" in sim.results:
            base = sim.results["prevalence_analyzer_hiv"]
            if "prevalence" not in sim.results:
                sim.results["prevalence"] = base
            if "prevalence_analyzer" not in sim.results:
                sim.results["prevalence_analyzer"] = base
    except Exception:
        pass

    return sim

def build_sim(sim, calib_pars):
    # Local import so `--help` works without Starsim installed/importable
    hiv = sim.diseases.hiv
    nw = sim.networks.structuredsexual

    # Apply the calibration parameters
    for k, pars in calib_pars.items():  # Loop over the calibration parameters
        if k == 'rand_seed':
            sim.pars.rand_seed = pars
            continue

        v = pars['value']
        if 'hiv_' in k:  # HIV parameters
            k = k.replace('hiv_', '')  # Strip off identifying part of parameter name
            hiv.pars[k] = v
        elif 'nw_' in k:  # Network parameters
            k = k.replace('nw_', '')  # As above
            if 'pair_form' in k:
                nw.pars[k].set(v)
            else:
                nw.pars[k] = v
        else:
            raise NotImplementedError(f'Parameter {k} not recognized')

    return sim


def _infer_obs_scale(obs_df, *, units):
    """
    Return a multiplier to convert observed HIV prevalence into model units (fractions 0-1).

    Notes on units
    --------------
    - If your observed values are in 0–1 (e.g. 0.52 meaning 52%), use `fraction`.
    - If your observed values are in 0–100 (e.g. 52 meaning 52%), use `percent`.
    - `auto` assumes:
        * max > 1  => percent
        * max <= 1 => fraction
    """
    units = str(units).strip().lower()
    if units not in {"auto", "fraction", "percent"}:
        raise ValueError(f"Invalid obs units {units!r}; use auto|fraction|percent")

    import pandas as pd

    vals = pd.to_numeric(obs_df[["HIV_male", "HIV_female"]].stack(), errors="coerce")
    vmax = float(vals.max()) if len(vals) else 0.0

    if units == "fraction":
        return 1.0
    if units == "percent":
        return 0.01

    # auto
    if vmax > 1.0:
        return 0.01
    return 1.0


def build_stisim_style_hiv_data(
    cfg,
    *,
    results_key,
    analyzer_age_bins,
):
    """
    Build a DataFrame compatible with `stisim.calibration.Calibration(data=...)`.

    Important: STIsim's `make_df()` does **not** use dot paths; it parses result
    names by splitting on underscores and treating the first chunk as the key in
    `sim.results`. Therefore column names here must follow:
        <results_key>_hiv_prev_male_{i}
        <results_key>_hiv_prev_female_{i}

    In practice we set `results_key="prevalence"` and provide a compatibility
    alias `sim.results["prevalence"] = sim.results["prevalence_analyzer_hiv"]`.
    """
    import numpy as np
    import pandas as pd

    obs = pd.read_csv(cfg.prevalence_hiv_csv)
    obs["Age"] = pd.to_numeric(obs.get("Age"), errors="coerce")
    obs["Year"] = pd.to_numeric(obs.get("Year"), errors="coerce")
    obs["HIV_male"] = pd.to_numeric(obs.get("HIV_male"), errors="coerce")
    obs["HIV_female"] = pd.to_numeric(obs.get("HIV_female"), errors="coerce")
    obs = obs.dropna(subset=["Age", "Year"])
    obs["Age"] = obs["Age"].astype(int)
    obs["Year"] = obs["Year"].astype(int)

    scale = _infer_obs_scale(obs, units=cfg.obs_units)
    obs["HIV_male"] = obs["HIV_male"] * scale
    obs["HIV_female"] = obs["HIV_female"] * scale

    years = np.arange(int(cfg.start), int(cfg.stop) + 1, dtype=int)
    # STIsim's calibration.make_df() returns timestep indices as `time` for
    # analyzer results, so keep observed data on the same 0-based axis while
    # still using calendar years to look up observed values.
    out = pd.DataFrame({"time": np.arange(len(years), dtype=float)})
    out_year = pd.Series(years, index=out.index)

    # Map analyzer bins to observed (lower-edge) age values
    for i, (a0, _a1) in enumerate(analyzer_age_bins):
        a0i = int(a0)
        sub = obs[obs["Age"] == a0i]
        if sub.empty:
            # keep NaNs
            out[f"{results_key}_hiv_prev_male_{i}"] = np.nan
            out[f"{results_key}_hiv_prev_female_{i}"] = np.nan
            continue

        male_series = sub.set_index("Year")["HIV_male"]
        fem_series = sub.set_index("Year")["HIV_female"]
        out[f"{results_key}_hiv_prev_male_{i}"] = out_year.map(male_series)
        out[f"{results_key}_hiv_prev_female_{i}"] = out_year.map(fem_series)

    # Drop any series that have no observed points in the selected time range
    keep_cols = ["time"]
    for c in out.columns:
        if c == "time":
            continue
        if out[c].notna().any():
            keep_cols.append(c)
    out = out[keep_cols]

    return out


def safe_stisim_eval_fn(sim, data=None, sim_result_list=None, weights=None, df_res_list=None):
    """
    STIsim-style evaluator that tolerates empty observed/model overlaps.

    Some age-sex HIV prevalence bins can have observed data but no simulated
    denominator after dropping NaNs, especially in sparse older age bins.
    STIsim's default evaluator passes those empty arrays to compute_gof(), which
    raises on max(empty). We skip those series and score the usable overlaps.
    """
    import pandas as pd
    import stisim.calibration as scali

    df_res = scali.make_df(sim, df_res_list=df_res_list)
    sim.df_res = df_res

    fit = 0.0
    n_scored = 0
    skipped = []
    for skey in sim_result_list:
        data_df = data[skey].reset_index()
        model_df = df_res[["time", skey]]
        combined = pd.merge(data_df, model_df, how="left", on="time").dropna(subset=[f"{skey}_x", f"{skey}_y"])
        if combined.empty:
            skipped.append(skey)
            continue

        losses = scali.compute_gof(combined[f"{skey}_x"], combined[f"{skey}_y"])
        if weights and (skey in weights.keys()) and (weights[skey] != 1):
            losses *= weights[skey]
        fit += losses.sum()
        n_scored += 1

    if n_scored == 0:
        raise ValueError(
            "No HIV calibration series had overlapping non-null observed and simulated values. "
            f"Available data columns: {list(sim_result_list or [])}"
        )
    if skipped:
        logger.debug("Skipped %d empty HIV calibration series: %s", len(skipped), skipped)
    return float(fit)


def run_calib(cfg, calib_pars=None):
    """
    Run the calibration simulation with the given parameters.

    Args:
        calib_pars (dict): Dictionary of calibration parameters.
    """
    sim = make_sim(cfg)
    if cfg.use_stisim_calibration:
        import optuna
        import stisim.calibration as scali
        import mighti as mi

        # Use the analyzer's age bins to build matching observed columns
        prev_analyzer = None
        analyzers = getattr(sim, "analyzers", None)
        if isinstance(analyzers, dict):
            for a in analyzers.values():
                if isinstance(a, mi.analyzers.PrevalenceAnalyzer_HIV):
                    prev_analyzer = a
                    break
        if prev_analyzer is None and hasattr(analyzers, "__iter__"):
            for a in analyzers:
                if isinstance(a, mi.analyzers.PrevalenceAnalyzer_HIV):
                    prev_analyzer = a
                    break
        if prev_analyzer is None:
            # Fallback: take first analyzer if present
            try:
                prev_analyzer = next(iter(analyzers.values())) if isinstance(analyzers, dict) else next(iter(analyzers))
            except Exception:
                prev_analyzer = None
        if prev_analyzer is None:
            raise RuntimeError("Could not find PrevalenceAnalyzer_HIV on sim; cannot build STIsim-style calibration data.")

        # STIsim calibration extracts results via underscore parsing:
        #   modname = sres.split('_')[0]
        # then looks up `sim.results[modname][resname]`.
        # Therefore, we must use a **modname without underscores**, i.e. "prevalence".
        if hasattr(sim, "results") and "prevalence" not in sim.results:
            # Create/repair alias to whatever prevalence analyzer key exists
            base_key = None
            if "prevalence_analyzer_hiv" in sim.results:
                base_key = "prevalence_analyzer_hiv"
            else:
                for k in sim.results.keys():
                    if "prevalence_analyzer" in str(k):
                        base_key = k
                        break
            if base_key is not None:
                sim.results["prevalence"] = sim.results[base_key]
        results_key = "prevalence"

        data_df = build_stisim_style_hiv_data(
            cfg,
            results_key=str(results_key),
            analyzer_age_bins=list(getattr(prev_analyzer, "age_bins", [])),
        )
        # Quick sanity check: ensure parsed modname exists in sim.results
        # (STIsim uses modname = sres.split('_')[0])
        bad_prefixes = sorted({c.split("_")[0] for c in data_df.columns if c != "time"} - set(sim.results.keys()))
        if bad_prefixes:
            logger.warning(
                "Calibration data contains prefixes not in sim.results: %s. sim.results keys=%s",
                bad_prefixes,
                list(sim.results.keys()),
            )
        logger.info("Using STIsim-style calibration modname: %s", results_key)

        calib = scali.Calibration(
            sim=sim,
            calib_pars=calib_pars,
            build_fn=build_sim,
            data=data_df,
            save_results=True,
            total_trials=int(cfg.trials),
            n_workers=1,
            keep_db=bool(cfg.keep_db),
            die=True,
            reseed=False,
            sampler=optuna.samplers.TPESampler(seed=int(cfg.sampler_seed)),
        )
        result_cols = [c for c in data_df.columns if c != "time"]
        calib.eval_fn = safe_stisim_eval_fn
        calib.eval_kw = dict(
            data=calib.data,
            sim_result_list=result_cols,
            weights=None,
            df_res_list=result_cols,
        )
    else:
        import optuna
        import pandas as pd
        import starsim as ss

        data = pd.read_csv(cfg.prevalence_hiv_csv)
        calib = ss.Calibration(
            sim=sim,
            calib_pars=calib_pars,
            build_fn=build_sim,
            eval_fn=eval_fn,
            eval_kw={'data': data},
            total_trials=int(cfg.trials),
            n_workers=1,
            keep_db=bool(cfg.keep_db),
            die=True,
            reseed=False,
            sampler=optuna.samplers.TPESampler(seed=int(cfg.sampler_seed)),
        )

    calib.calibrate()
    calib.check_fit()

    # Return the results for further analysisz
    return calib


def eval_fn(sim, data=None, sim_result_list=None, weights=None, df_res_list=None):
    """
    Custom evaluation function for HIV calibration
    """
    import pandas as pd
    import starsim as ss
    import mighti as mi
    if isinstance(sim, ss.MultiSim):
        sim = sim.sims[0]

    def _find_col(df, candidates):
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in df.columns:
                return cand
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        return None

    if data is None:
        raise ValueError("eval_fn requires observed HIV prevalence data via eval_kw={'data': ...}.")

    hiv_female_col = _find_col(data, ["HIV_female", "hiv_female"])
    hiv_male_col = _find_col(data, ["HIV_male", "hiv_male"])
    if hiv_female_col is None or hiv_male_col is None:
        hiv_like = [c for c in data.columns if "hiv" in c.lower()]
        raise ValueError(
            "Observed prevalence data is missing required columns for HIV by sex. "
            f"Expected columns like 'HIV_female' and 'HIV_male'. "
            f"Found HIV-like columns: {hiv_like}. "
            "Please provide an HIV prevalence CSV with Age/Year and sex-stratified HIV prevalence."
        )

    # Normalize observed data (if in %)
    if pd.to_numeric(data[hiv_female_col], errors="coerce").max() > 1:
        data[[hiv_female_col, hiv_male_col]] = data[[hiv_female_col, hiv_male_col]] / 100.0

    fit = 0
    prev_analyzer = sim.analyzers.get('prevalence_analyzer') if hasattr(sim.analyzers, "get") else getattr(sim.analyzers, "prevalence_analyzer", None)
    prev_results = sim.results.get('prevalence_analyzer') if hasattr(sim, "results") else None
    if prev_analyzer is None or prev_results is None:
        # More robust: search analyzer dict for the prevalence analyzer and use that key for results
        prev_label = None
        prev_analyzer = None
        for label, analyzer in getattr(sim, "analyzers", {}).items():
            if isinstance(analyzer, mi.analyzers.PrevalenceAnalyzer_HIV):
                prev_label = label
                prev_analyzer = analyzer
                break
        if prev_label is None or prev_analyzer is None:
            raise ValueError(
                "PrevalenceAnalyzer_HIV not found on sim; ensure analyzers include "
                "mi.analyzers.PrevalenceAnalyzer_HIV()."
            )
        prev_results = sim.results.get(prev_label)
        if prev_results is None:
            raise ValueError(f"PrevalenceAnalyzer_HIV results not found under key {prev_label!r}.")

    for index, (age_low, age_high) in enumerate(prev_analyzer.age_bins):
        prev_observed_data = data[data['Age'] == age_low][['Year', 'Age', hiv_female_col, hiv_male_col]].copy()
        prev_observed_data['Year'] = prev_observed_data['Year'].astype(int)

        # Normalize analyzer time vector to int years
        sim_years = [t.year if hasattr(t, 'year') else int(t) for t in prev_analyzer.timevec]

        prev_sim_data = pd.DataFrame({
            'Year': sim_years,
            'Age': age_low,
            'sim_HIV_female': prev_results[f'hiv_prev_female_{index}'],
            'sim_HIV_male':   prev_results[f'hiv_prev_male_{index}'],
        })

        merged = pd.merge(prev_observed_data, prev_sim_data, on=['Year', 'Age'], how='inner')
        merged['error'] = abs(merged['sim_HIV_female'] - merged[hiv_female_col]) + \
                        abs(merged['sim_HIV_male'] - merged[hiv_male_col])
        fit += merged['error'].sum()

    n_obs = len(data['Age'].unique()) * 2
    return fit / n_obs


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
    parser.add_argument("--keep-db", action="store_true", help="Keep calibration DB so you can resume later.")
    parser.add_argument(
        "--obs-units",
        default="auto",
        choices=["auto", "fraction", "percent"],
        help="Units of observed HIV prevalence columns. Use 'percent' if values are percents (0-100 or 0-1 meaning percent).",
    )
    parser.add_argument(
        "--use-stisim-calibration",
        action="store_true",
        help="Use STIsim's calibration wrapper (compute_gof + eval-from-data). Recommended.",
    )
    parser.add_argument(
        "--no-use-stisim-calibration",
        dest="use_stisim_calibration",
        action="store_false",
        help="Use the legacy custom eval function instead of STIsim calibration.",
    )
    parser.set_defaults(use_stisim_calibration=True)
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
        trials=int(args.trials),
        sampler_seed=int(args.sampler_seed),
        keep_db=bool(args.keep_db),
        obs_units=str(args.obs_units),
        use_stisim_calibration=bool(args.use_stisim_calibration),
    )


#%% Run as a script
if __name__ == '__main__':

    import sciris as sc

    T = sc.tic()

    cfg = _parse_args()

    # Define the calibration parameters. These are parsed in build_sim() as: {hiv/nw}_{parameter_name}
    # where hiv is for STIsim HIV parameters and nw is for StructuredSexual network parameters.
    calib_pars = dict(
        hiv_beta_m2f = dict(low=0.001, high=0.10, guess=0.03), # HIV transmission parameter
        hiv_beta_m2c = dict(low=0.0001, high=0.1, guess=0.001), # Network females in risk group 1 concurrent partners
    )

    calib = run_calib(cfg=cfg, calib_pars=calib_pars)

    sc.toc(T)
    print('Done.')
    