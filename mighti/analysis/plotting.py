"""
Provides utility functions for generating plots from simulation outputs
"""


import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import Counter


def _safe_get_result(analyzer, key, sim):
    """
    Return analyzer result as a numpy array aligned to sim.timevec length.
    Works for both Starsim ≤2.x and ≥3.x result structures.
    """
    val = analyzer.results.get(key)
    if val is None:
        return np.zeros(len(sim.timevec))
    if hasattr(val, "values"):  # e.g. pandas Series
        val = val.values
    val = np.array(val)
    if len(val) != len(sim.timevec):
        val = np.pad(val, (0, max(0, len(sim.timevec) - len(val))), mode="edge")[: len(sim.timevec)]
    return val

logger = logging.getLogger(__name__)

def _is_interactive_backend():
    """Return True if current Matplotlib backend can show GUI windows."""
    try:
        import matplotlib
        from matplotlib import rcsetup

        backend = str(matplotlib.get_backend()).lower()
        return backend in {b.lower() for b in rcsetup.interactive_bk}
    except Exception:
        return False


def _finalize_figure(fig, *, show=True, savepath=None, dpi=200):
    """
    Common end-of-plot behavior:
    - Save if `savepath` is provided
    - Only call plt.show() if requested AND backend is interactive
    """
    if savepath:
        try:
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        except Exception as e:
            logger.warning("Failed to save figure to %s: %s", savepath, e)

    if show and _is_interactive_backend():
        plt.show()
    return fig


def _coerce_sim_years(sim):
    """
    Return a numeric year vector aligned to sim.timevec.
    Handles numeric years, datetime-like, or pandas Timestamp.
    """
    tv = np.array(getattr(sim, "timevec", []))
    if tv.size == 0:
        return tv.astype(float)
    # If timevec is already numeric years, use directly
    if tv.dtype.kind in ("i", "u", "f"):
        years = tv.astype(float)
        # Heuristic: treat as years if values look like calendar years
        finite = years[np.isfinite(years)]
        if finite.size and (finite.min() >= 1800) and (finite.max() <= 2500):
            return years

    # Try datetime conversion
    try:
        dt = pd.to_datetime(tv)
        years = dt.year.to_numpy(dtype=float)
        # Guard: if conversion produced all-NaT, fallback
        if np.all(pd.isna(dt)):
            raise ValueError("timevec not datetime-like")
        return years
    except Exception:
        pass
    # Fallback: numeric
    return np.asarray(tv, dtype=float)


def plot_hiv_prevalence_vs_observed(
    sim,
    prevalence_analyzer,
    observed_hiv_df,
    *,
    age_starts=None,
    start_year=None,
    end_year=None,
    ncols=3,
    figsize=(14, 8),
    title=None,
    show=True,
    savepath=None,
):
    """
    Plot simulated HIV prevalence vs observed, stratified by age bin and sex.

    This is designed to work with `mi.analyzers.PrevalenceAnalyzer_HIV`, which
    stores per-age-bin results as:
      - `hiv_prev_male_{i}`, `hiv_prev_female_{i}` (proportions 0-1)

    Observed data is expected in *wide* format with columns:
      - `Age` (lower bound of age bin, e.g. 15, 20, 25...)
      - `Year`
      - `HIV_male`, `HIV_female` (either proportions 0-1 or %)
    """
    if observed_hiv_df is None or len(observed_hiv_df) == 0:
        raise ValueError("observed_hiv_df is empty; nothing to plot")

    # Default bins: match common reporting bins and keep the figure readable
    if age_starts is None:
        age_starts = [15, 20, 25, 30, 35, 40, 45, 50]

    # Build mapping from age_start -> analyzer index
    age_bins = getattr(prevalence_analyzer, "age_bins", None)
    if not age_bins:
        raise ValueError("prevalence_analyzer has no `age_bins`; expected PrevalenceAnalyzer_HIV-like analyzer")
    age_to_i = {int(a0): i for i, (a0, _a1) in enumerate(age_bins)}

    chosen = [a for a in age_starts if int(a) in age_to_i]
    if not chosen:
        raise ValueError(f"No requested age bins found in analyzer.age_bins. Requested={age_starts}, available={[int(a0) for a0,_ in age_bins]}")

    # Prefer pulling time series from sim.results (more reliable than analyzer.results)
    results_store = None
    try:
        if hasattr(sim, "analyzers") and hasattr(sim, "results"):
            # Starsim often stores analyzers as dict label->analyzer
            if isinstance(sim.analyzers, dict):
                for label, analyzer in sim.analyzers.items():
                    if analyzer is prevalence_analyzer:
                        results_store = sim.results.get(label)
                        break
                # Fallback: match by analyzer name if object identity differs
                if results_store is None:
                    want_name = getattr(prevalence_analyzer, "name", None)
                    for label, analyzer in sim.analyzers.items():
                        if want_name and getattr(analyzer, "name", None) == want_name:
                            results_store = sim.results.get(label)
                            break
    except Exception:
        results_store = None

    def _get_series(key):
        if isinstance(results_store, dict) and key in results_store:
            return np.asarray(results_store[key], dtype=float)
        return _safe_get_result(prevalence_analyzer, key, sim).astype(float)

    # Sim years and mask
    years = _coerce_sim_years(sim)
    if years.size == 0:
        raise ValueError("sim.timevec is empty")
    y0 = int(np.nanmin(years)) if start_year is None else int(start_year)
    y1 = int(np.nanmax(years)) if end_year is None else int(end_year)
    sim_mask = (years >= y0) & (years <= y1)

    # Clean observed
    df = observed_hiv_df.copy()
    # Support mixed case column names
    colmap = {c.lower(): c for c in df.columns}
    for req in ("age", "year", "hiv_male", "hiv_female"):
        if req not in colmap:
            raise ValueError(f"observed_hiv_df missing required column {req!r}. Found columns={list(df.columns)}")
    age_col = colmap["age"]
    year_col = colmap["year"]
    m_col = colmap["hiv_male"]
    f_col = colmap["hiv_female"]

    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[m_col] = pd.to_numeric(df[m_col], errors="coerce")
    df[f_col] = pd.to_numeric(df[f_col], errors="coerce")
    df = df.dropna(subset=[age_col, year_col])
    df[age_col] = df[age_col].astype(int)
    df[year_col] = df[year_col].astype(int)
    df = df[(df[year_col] >= y0) & (df[year_col] <= y1)]

    # Determine observed units (0-1 vs %)
    obs_max = float(np.nanmax([df[m_col].max(skipna=True), df[f_col].max(skipna=True)])) if len(df) else 0.0
    obs_scale = 100.0 if obs_max <= 1.5 else 1.0  # treat <=1.5 as proportion

    # Layout
    n = len(chosen)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax_idx, a0 in enumerate(chosen):
        ax = axes[ax_idx]
        i = age_to_i[int(a0)]

        sim_m = _get_series(f"hiv_prev_male_{i}") * 100.0
        sim_f = _get_series(f"hiv_prev_female_{i}") * 100.0

        ax.plot(years[sim_mask], sim_m[sim_mask], color="blue", lw=2.5, label="Sim male" if ax_idx == 0 else None)
        ax.plot(years[sim_mask], sim_f[sim_mask], color="red", lw=2.5, label="Sim female" if ax_idx == 0 else None)

        obs = df[df[age_col] == int(a0)]
        if len(obs):
            ax.scatter(obs[year_col].astype(float), obs[m_col] * obs_scale, color="blue", edgecolor="black", s=35, zorder=5, label="Obs male" if ax_idx == 0 else None)
            ax.scatter(obs[year_col].astype(float), obs[f_col] * obs_scale, color="red", edgecolor="black", s=35, zorder=5, label="Obs female" if ax_idx == 0 else None)

        a1 = age_bins[i][1]
        ax.set_title(f"Age {int(a0)}–{int(a1)-1}" if np.isfinite(a1) else f"Age {int(a0)}+", fontsize=11)
        ax.grid(True, alpha=0.25)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # Shared labels
    fig.suptitle(title or "HIV prevalence: simulated vs observed", fontsize=14)
    for ax in axes[:n]:
        ax.set_xlim(y0, y1)
        ax.set_ylim(bottom=0)
    fig.supxlabel("Year")
    fig.supylabel("Prevalence (%)")

    # One legend (from first axis)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _finalize_figure(fig, show=show, savepath=savepath)
    return fig, axes[:n]

def plot_life_expectancy_timeseries(
    le_df,
    *,
    sex="Both",
    scenarios=None,
    highlight_years=None,
    title=None,
    ylabel="Life expectancy at birth (e₀)",
    xlabel="Year",
    figsize=(10, 5),
    show=True,
):
    """
    Plot a time series of life expectancy at birth (e₀).

    Expects a tidy DataFrame with columns:
      - year (int)
      - scenario (str)
      - sex (str): 'Male', 'Female', or 'Both'
      - e0 (float)
    """
    if le_df is None or len(le_df) == 0:
        raise ValueError("le_df is empty; nothing to plot")

    df = le_df.copy()
    required = {"year", "scenario", "sex", "e0"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"le_df missing required columns: {sorted(missing)}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["sex"] = df["sex"].astype(str)
    df["scenario"] = df["scenario"].astype(str)
    df["e0"] = pd.to_numeric(df["e0"], errors="coerce")
    df = df.dropna(subset=["e0"])

    # Filter
    df = df[df["sex"].str.lower() == str(sex).lower()]
    if scenarios is not None:
        keep = {s.lower() for s in scenarios}
        df = df[df["scenario"].str.lower().isin(keep)]

    if df.empty:
        raise ValueError("No rows left after filtering; check `sex`/`scenarios`")

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    for scen, sub in df.groupby("scenario", sort=True):
        sub = sub.sort_values("year")
        ax.plot(sub["year"], sub["e0"], linewidth=2, label=scen)

    # Optional highlights (e.g., 1990 HIV era inflection)
    if highlight_years:
        for y in highlight_years:
            ax.axvline(int(y), color="k", linestyle="--", linewidth=1, alpha=0.35)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title(title or f"Life expectancy at birth over time ({sex})")
    plt.tight_layout()

    if show:
        plt.show()
    return fig, ax

def plot_mean_prevalence_plhiv(sim, prevalence_analyzer, disease, *, show=True, savepath=None):
    """
    Plot mean prevalence over time for a given disease and both sexes.
    """
    disease = disease.lower() 

    def extract_results(key_pattern):
        matching_keys = [k for k in prevalence_analyzer.results.keys()
                         if k.startswith(f'{disease}_{key_pattern}_')]
        matching_keys = sorted(matching_keys, key=lambda x: int(x.split('_')[-1]))
        if not matching_keys:
            logger.debug("No keys found for pattern %s_%s_", disease, key_pattern)
        return [_safe_get_result(prevalence_analyzer, k, sim) for k in matching_keys]

    male_num_with_HIV = np.sum(extract_results('num_with_HIV_male'), axis=0)
    female_num_with_HIV = np.sum(extract_results('num_with_HIV_female'), axis=0)
    male_den_with_HIV = np.sum(extract_results('den_with_HIV_male'), axis=0)
    female_den_with_HIV = np.sum(extract_results('den_with_HIV_female'), axis=0)
    male_num_without_HIV = np.sum(extract_results('num_without_HIV_male'), axis=0)
    female_num_without_HIV = np.sum(extract_results('num_without_HIV_female'), axis=0)
    male_den_without_HIV = np.sum(extract_results('den_without_HIV_male'), axis=0)
    female_den_without_HIV = np.sum(extract_results('den_without_HIV_female'), axis=0)

    male_den_with_HIV[male_den_with_HIV == 0] = 1
    female_den_with_HIV[female_den_with_HIV == 0] = 1
    male_den_without_HIV[male_den_without_HIV == 0] = 1
    female_den_without_HIV[female_den_without_HIV == 0] = 1

    mean_prevalence_male_with_HIV = np.nan_to_num(male_num_with_HIV / male_den_with_HIV) * 100
    mean_prevalence_female_with_HIV = np.nan_to_num(female_num_with_HIV / female_den_with_HIV) * 100
    mean_prevalence_male_without_HIV = np.nan_to_num(male_num_without_HIV / male_den_without_HIV) * 100
    mean_prevalence_female_without_HIV = np.nan_to_num(female_num_without_HIV / female_den_without_HIV) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sim.timevec, mean_prevalence_male_with_HIV, label=f'Male {disease} (HIV+)', color='blue', lw=2)
    ax.plot(sim.timevec, mean_prevalence_female_with_HIV, label=f'Female {disease} (HIV+)', color='red', lw=2)
    ax.plot(sim.timevec, mean_prevalence_male_without_HIV, '--', color='blue', lw=2, label=f'Male {disease} (HIV–)')
    ax.plot(sim.timevec, mean_prevalence_female_without_HIV, '--', color='red', lw=2, label=f'Female {disease} (HIV–)')
    ax.set_xlabel('Year', fontsize=16)
    ax.set_ylabel(f'{disease} Prevalence (%)', fontsize=16)
    ax.legend()
    ax.grid(True)
    _finalize_figure(fig, show=show, savepath=savepath)
    return fig, ax
    
    
def plot_mean_prevalence(sim, prevalence_analyzer, disease, prevalence_data_df, init_year, end_year, *, show=True, savepath=None):
    """
    Plot mean prevalence over time for a given disease and both sexes, including observed data points.
    Ensures x-axis uses numeric years, not dates.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import units, dates

    disease = disease.lower()

    # --- Safe getter ---
    def _safe_get_result(analyzer, key, sim):
        val = analyzer.results.get(key)
        if val is None:
            return np.zeros(len(sim.timevec))
        if hasattr(val, "values"):
            val = val.values
        val = np.array(val, dtype=float)
        if len(val) != len(sim.timevec):
            val = np.pad(val, (0, max(0, len(sim.timevec) - len(val))), mode="edge")[:len(sim.timevec)]
        return val

    # --- Extract results ---
    def extract_results(key_pattern):
        matching_keys = [k for k in prevalence_analyzer.results.keys()
                         if k.startswith(f"{disease}_{key_pattern}_")]
        matching_keys = sorted(matching_keys, key=lambda x: int(x.split('_')[-1]))
        if not matching_keys:
            logger.debug("No keys found for pattern %s_%s_", disease, key_pattern)
        return [_safe_get_result(prevalence_analyzer, k, sim) for k in matching_keys]

    n_t = len(sim.timevec)

    def _to_len_n(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if x.size == 1:
            return np.full(n_t, float(x.flat[0]))
        return x

    male_num = _to_len_n(np.sum(extract_results("num_male"), axis=0))
    female_num = _to_len_n(np.sum(extract_results("num_female"), axis=0))
    male_den = _to_len_n(np.sum(extract_results("den_male"), axis=0))
    female_den = _to_len_n(np.sum(extract_results("den_female"), axis=0))

    # Fallback: if no per-sex keys (e.g. disease not in analyzer list), use total keys
    if (np.asarray(male_den) == 0).all() or (np.asarray(female_den) == 0).all():
        num_total_key = f"{disease}_num_total"
        den_total_key = f"{disease}_den_total"
        if num_total_key in prevalence_analyzer.results and den_total_key in prevalence_analyzer.results:
            total_num = _safe_get_result(prevalence_analyzer, num_total_key, sim)
            total_den = _safe_get_result(prevalence_analyzer, den_total_key, sim)
            total_den = np.atleast_1d(np.asarray(total_den, dtype=float))
            if total_den.size == 1:
                total_den = np.broadcast_to(total_den, (n_t,)).copy()
            total_den[total_den == 0] = 1
            # Use same total for both sexes so at least one line is non-zero
            male_num = female_num = total_num.astype(float) / 2.0
            male_den = female_den = total_den.astype(float) / 2.0
        else:
            logger.warning(
                "No prevalence results for disease '%s'. Is it in the analyzer's disease list?",
                disease,
            )

    male_den[male_den == 0] = 1
    female_den[female_den == 0] = 1

    total_male_prev = np.nan_to_num(male_num / male_den) * 100
    total_female_prev = np.nan_to_num(female_num / female_den) * 100

    mask = (sim.timevec >= init_year) & (sim.timevec <= end_year)
    time_years = np.array(sim.timevec[mask], dtype=float)

    # --- Plot simulated curves ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_years, total_male_prev[mask], label="Male Simulated", color="blue", lw=3)
    ax.plot(time_years, total_female_prev[mask], label="Female Simulated", color="red", lw=3)

    # --- Clean observed data ---
    df = prevalence_data_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "year" not in df.columns:
        raise ValueError("prevalence_data_df must include a 'year' column")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    most_recent = df["year"].max()
    df = df[(df["year"] >= init_year) & (df["year"] <= min(end_year, most_recent))]

    disease_col = disease.lower()

    if disease_col in df.columns and "sex" in df.columns:
        for sex, color in [("male", "blue"), ("female", "red")]:
            obs = df[df["sex"].str.lower() == sex]
            if not obs.empty:
                ax.scatter(
                    obs["year"].astype(float), obs[disease_col] * 100,
                    label=f"{sex.capitalize()} Observed",
                    color=color, edgecolor="black", s=80, zorder=5
                )

    # --- Ensure numeric (not datetime) x-axis ---
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    units.registry[np.ndarray] = None   # disable automatic datetime conversion

    # --- Style ---
    ax.set_title(f"{disease.capitalize()} Prevalence Over Time (All Ages)", fontsize=16)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Prevalence (%)", fontsize=14)
    ax.set_xlim(init_year - 1, end_year + 1)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=12)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _finalize_figure(fig, show=show, savepath=savepath)

    logger.info("Observed data restricted to %s–%s (%s rows used).", init_year, min(end_year, most_recent), len(df))
    return total_male_prev, total_female_prev   
    
    
# ---------------------------------------------------------------------
# Demographics related plot functions
# ---------------------------------------------------------------------
def plot_mx_comparison(sim_mx_df, observed_mx_csv, year, age_interval=5, figsize=(14, 10)):
    """
    Plot simulated and observed mx (mortality rate) by age group for a given year,
    using the output of calculate_mortality_rates.

    Args:
        sim_mx_df: DataFrame from calculate_mortality_rates with ['year', 'age', 'sex', 'mx']
        observed_mx_csv: Path to observed mx CSV with columns ['Time', 'Sex', 'Age', 'mx']
        year: Year to plot (should be present in both data sources)
        age_interval: Plot in this age grouping (default 5)
        figsize: Figure size
    """
    observed = observed_mx_csv
    sexes = ['Male', 'Female']
    colors = {'Male': 'blue', 'Female': 'red'}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes = [ax1, ax2]
    
    for i, sex in enumerate(sexes):
        color = colors[sex]
        sim_df = sim_mx_df[(sim_mx_df['sex'] == sex) & (sim_mx_df['year'] == year)][['age', 'mx']].copy()
        sim_df['age_group'] = (sim_df['age'] // age_interval) * age_interval
        sim_mx_grouped = sim_df.groupby('age_group')['mx'].mean().reset_index()

        obs_df = observed[(observed['Sex'] == sex) & (observed['Time'] == year)][['Age', 'mx']].copy()
        obs_df['age_group'] = (obs_df['Age'] // age_interval) * age_interval
        obs_mx_grouped = obs_df.groupby('age_group')['mx'].mean().reset_index()

        ax = axes[i]
        ax.set_yscale('log')  # ← log scale here

        ax.plot(sim_mx_grouped['age_group'], sim_mx_grouped['mx'],
                linestyle='-', linewidth=8, alpha=0.4, color=color, label='Simulated')
        ax.plot(obs_mx_grouped['age_group'], obs_mx_grouped['mx'],
                marker='s', linestyle='--', linewidth=2, markersize=5,
                color='black', label='Observed')
        ax.set_title(f"{sex} Mortality Rate (mx) Comparison, {year}", fontsize=24)
        ax.set_ylabel('m(x)', fontsize=24)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel('Age Group', fontsize=24)
    plt.tight_layout()
    plt.show()    
    

def plot_life_expectancy(life_table, observed_data, year, max_age=100, figsize=(14, 10), title=None):
    """
    Plot simulated and observed life expectancy by age and sex for a given year.

    Args:
        life_table: DataFrame from simulation with ['Age', 'e(x)', 'sex', 'year']
        observed_data: Long-format DataFrame with ['Age', 'Sex', 'Time', 'ex']
        year: Year to filter
        max_age: Maximum age to plot
        figsize: Size of the figure
        title: Optional plot title
    """
    male_sim = life_table[life_table['sex'] == 'Male']
    female_sim = life_table[life_table['sex'] == 'Female']
    male_obs = observed_data[observed_data['Sex'] == 'Male']
    female_obs = observed_data[observed_data['Sex'] == 'Female']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax1.plot(male_sim['Age'], male_sim['e(x)'], '-', linewidth=12, alpha=0.4, color='blue', label='Simulated')
    ax1.plot(male_obs['Age'], male_obs['ex'], 's--', linewidth=2, markersize=5, color='black', label='Observed')
    ax1.set_title('Male', fontsize=28)
    ax1.set_ylabel('Life Expectancy (years)', fontsize=24)
    ax1.tick_params(labelsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=20)

    ax2.plot(female_sim['Age'], female_sim['e(x)'], '-', linewidth=12, alpha=0.4, color='red', label='Simulated')
    ax2.plot(female_obs['Age'], female_obs['ex'], 's--', linewidth=2, markersize=5, color='black', label='Observed')
    ax2.set_title('Female', fontsize=28)
    ax2.set_xlabel('Age', fontsize=24)
    ax2.set_ylabel('Life Expectancy (years)', fontsize=24)
    ax2.tick_params(labelsize=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=20)

    if title:
        plt.suptitle(title, fontsize=24, y=0.98)
    else:
        plt.suptitle(f'Life Expectancy by Age in {year}', fontsize=24, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(bottom=0.1)
    plt.show()
    return fig, (ax1, ax2)


def plot_cost_effectiveness_plane(results, wtp_thresholds=[100, 500, 1000],
                                  figsize=(8, 6),savepath=None,show=True):
    
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each result as a scatter point
    for res in results:
        ax.scatter(res['delta_daly'], res['delta_cost'], label=res['label'], s=80)
        ax.annotate(res['label'], (res['delta_daly'], res['delta_cost']), fontsize=10,
                    xytext=(5, 5), textcoords='offset points')

    # Plot WTP threshold lines
    x_vals = np.linspace(0, max([r['delta_daly'] for r in results]) * 1.1, 100)
    for wtp in wtp_thresholds:
        ax.plot(x_vals, wtp * x_vals, linestyle='--', label=f'${wtp}/DALY')

    # Axis labels and formatting
    ax.set_xlabel('Incremental DALYs averted', fontsize=12)
    ax.set_ylabel('Incremental cost ($)', fontsize=12)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend()
    ax.set_title('Cost-Effectiveness Plane', fontsize=14)
    ax.grid(True)

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_adherence_by_condition(sim, label):
    results = sim.results[label]
    t = np.array(results["time"])
    with_cond = np.array(results["on_with_condition"])
    without_cond = np.array(results["on_without_condition"])

    plt.plot(t, with_cond, label="With condition")
    plt.plot(t, without_cond, label="Without condition")
    plt.xlabel("Year")
    plt.ylabel("Proportion on ART")
    plt.title(f"Adherence by {label}")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()
    