"""
Calculates and analyzes mortality rates and life expectancy from simulation data
"""


import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple


import numpy as np
import pandas as pd


def calculate_mortality_rates(sim, deaths_module, year=None, max_age=100, radix=100000):
    """
    Compute age-specific mortality rates (m(x)) using simulated death tracking and survivorship.

    Args:
        sim (ss.Sim): The simulation object.
        deaths_module: Module tracking male/female deaths by age (e.g., mi.DeathsByAgeSexAnalyzer).
        year (int, optional): Simulation year for labeling output. If None, uses current sim year.
        max_age (int): Maximum age to include in calculations.
        radix (int): Reference population size used for initial survivorship (typically 100000).

    Returns:
        pd.DataFrame: A table with columns ['year', 'age', 'sex', 'mx'].
    """
    surv_an = sim.analyzers.survivorship_analyzer
    # SurvivorshipAnalyzer stores l(x) as a *fraction* of the initial sex-specific cohort (≈1.0 at birth).
    # To compute m(x) we must convert to person-years in a radix cohort.
    lx_male = np.asarray(surv_an.results.lx_male, dtype=float)
    lx_female = np.asarray(surv_an.results.lx_female, dtype=float)

    ppl = sim.people
    female = np.asarray(ppl.female, dtype=bool)
    n0_male = max(int((~female).sum()), 1)
    n0_female = max(int(female.sum()), 1)

    deaths_by_age = {'Male': np.zeros(max_age + 1), 'Female': np.zeros(max_age + 1)}
    person_years = {'Male': np.zeros(max_age + 1), 'Female': np.zeros(max_age + 1)}

    for age in range(max_age + 1):
        deaths_by_age['Male'][age] = (
            deaths_module.results.male_deaths_by_age[age]
            if age < len(deaths_module.results.male_deaths_by_age) else 0
        )
        deaths_by_age['Female'][age] = (
            deaths_module.results.female_deaths_by_age[age]
            if age < len(deaths_module.results.female_deaths_by_age) else 0
        )

    mortality_rates = []
    current_year = year if year is not None else int(sim.t.yearvec[sim.t.ti])

    for age in range(max_age):
        for sex, lx_frac in [('Male', lx_male), ('Female', lx_female)]:
            deaths = float(deaths_by_age[sex][age])
            # Scale deaths to the requested radix cohort size for this sex
            sex_scale = float(radix) / (n0_male if sex == 'Male' else n0_female)
            deaths_scaled = deaths * sex_scale

            # Person-years between x and x+1 in a radix cohort for this sex
            Lx = 0.5 * (lx_frac[age] + lx_frac[age + 1]) * float(radix)
            person_years[sex][age] = Lx
            mx = deaths_scaled / Lx if Lx > 0 else 0.0

            mortality_rates.append({
                'year': current_year,
                'age': age,
                'sex': sex,
                'mx': mx
            })

    # Handle open interval (max_age)
    for sex, lx_frac in [('Male', lx_male), ('Female', lx_female)]:
        deaths = float(deaths_by_age[sex][max_age])
        sex_scale = float(radix) / (n0_male if sex == 'Male' else n0_female)
        deaths_scaled = deaths * sex_scale

        # crude open-interval approximation using the last closed interval survivors
        Lx = 0.5 * (lx_frac[max_age - 1] + lx_frac[max_age]) * float(radix)
        mx = deaths_scaled / Lx if Lx > 0 else 0.0
        mortality_rates.append({'year': current_year, 'age': max_age, 'sex': sex, 'mx': mx})

    return pd.DataFrame(mortality_rates)


def calculate_life_table_from_mx(sim, df_mx_male, df_mx_female, max_age=100):
    """
    Compute life tables for males and females using m(x) and simulated l(0) from survivorship analyzer.

    Args:
        sim: Simulation object with a SurvivorshipAnalyzer.
        df_mx_male, df_mx_female: DataFrames with columns ['age', 'mx'].
        max_age: Maximum age to compute.

    Returns:
        pd.DataFrame with columns ['sex', 'Age', 'l(x)', 'd(x)', 'q(x)', 'm(x)', 'L(x)', 'T(x)', 'e(x)'].
    """
    surv_an = sim.analyzers.survivorship_analyzer
    l0_male = surv_an.results.lx_male[0]
    l0_female = surv_an.results.lx_female[0]

    def compute_life_table(sex, l0, m_x):
        l_x = [l0]
        for age in range(max_age):
            l_next = l_x[-1] * np.exp(-m_x[age])
            l_x.append(l_next)
        l_x = np.array(l_x)

        d_x = l_x[:-1] - l_x[1:]
        d_x = np.append(d_x, l_x[-1])  # all die at terminal age

        q_x = 1 - np.exp(-m_x)

        L_x = 0.5 * (l_x[:-1] + l_x[1:])
        L_x = np.append(L_x, l_x[-1] / m_x[-1] if m_x[-1] > 0 else 0)

        # Compute T(x) and e(x)
        T_x = np.zeros_like(L_x)
        T_accum = 0
        for i in reversed(range(max_age + 1)):
            T_accum += L_x[i]
            T_x[i] = T_accum

        # Avoid divide-by-zero warnings when l_x reaches 0 at extreme ages
        e_x = np.divide(T_x, l_x, out=np.zeros_like(T_x), where=(l_x > 0))

        return pd.DataFrame({
            'sex': sex,
            'Age': np.arange(max_age + 1),
            'l(x)': l_x,
            'd(x)': d_x,
            'q(x)': q_x,
            'm(x)': m_x,
            'L(x)': L_x,
            'T(x)': T_x,
            'e(x)': e_x
        })

    # Align and extract m(x)
    m_x_male = df_mx_male.set_index('age').reindex(range(max_age + 1)).fillna(0)['mx'].values
    m_x_female = df_mx_female.set_index('age').reindex(range(max_age + 1)).fillna(0)['mx'].values

    lt_male = compute_life_table('Male', l0_male, m_x_male)
    lt_female = compute_life_table('Female', l0_female, m_x_female)

    return pd.concat([lt_male, lt_female], ignore_index=True)


def load_un_mx_from_wide(mx_csv_path: str, year: int, max_age: int = 100) -> pd.DataFrame:
    """
    Load UN/WPP nMx from wide file with columns: Age, Sex, 1986.0, ..., 2023.0
    Return tidy ['age','sex','mx'] for the requested year and ages 0..max_age.
    """
    df = pd.read_csv(mx_csv_path)
    df.columns = [c.strip() for c in df.columns]
    # Identify ID cols
    age_col = next((c for c in df.columns if c.lower() == 'age'), None)
    sex_col = next((c for c in df.columns if c.lower() == 'sex'), None)
    if age_col is None or sex_col is None:
        raise ValueError("UN mx file must have 'Age' and 'Sex' columns.")
    id_vars = [age_col, sex_col]
    val_cols = [c for c in df.columns if c not in id_vars]

    long = df.melt(id_vars=id_vars, value_vars=val_cols,
                   var_name='year', value_name='mx')
    long['year'] = pd.to_numeric(long['year'], errors='coerce').astype('Int64')
    long = long.dropna(subset=['year'])
    long['year'] = long['year'].astype(int)

    out = long[long['year'] == year].copy()
    out = out.rename(columns={age_col: 'age', sex_col: 'sex'})
    out['age'] = pd.to_numeric(out['age'], errors='coerce').fillna(0).astype(int)
    out['sex'] = out['sex'].astype(str).str.strip().str.title()  # Male/Female
    out['mx']  = pd.to_numeric(out['mx'], errors='coerce').fillna(0.0)

    # Grid and clean per sex
    grids = []
    for s in ['Male', 'Female']:
        tmp = out[out['sex'] == s].set_index('age').reindex(range(max_age+1))
        if 'mx' not in tmp.columns:
            tmp['mx'] = np.nan
        tmp['mx'] = tmp['mx'].interpolate().bfill().fillna(0.0)
        tmp = tmp.reset_index().rename(columns={'index': 'age'})
        tmp['sex'] = s
        grids.append(tmp[['age', 'sex', 'mx']])
    return pd.concat(grids, ignore_index=True)


def load_un_ex_from_wide(ex_csv_path: str, year: int, *, age: int = 0) -> pd.DataFrame:
    """
    Load UN/WPP e(x) (life expectancy) from a wide file with columns:
      Age, Sex, 1986.0, ..., 2023.0

    Returns tidy DataFrame with columns: ['age', 'sex', 'ex'] for the requested year.
    """
    df = pd.read_csv(ex_csv_path)
    df.columns = [c.strip() for c in df.columns]

    age_col = next((c for c in df.columns if c.lower() == "age"), None)
    sex_col = next((c for c in df.columns if c.lower() == "sex"), None)
    if age_col is None or sex_col is None:
        raise ValueError("UN ex file must have 'Age' and 'Sex' columns.")

    id_vars = [age_col, sex_col]
    val_cols = [c for c in df.columns if c not in id_vars]
    long = df.melt(id_vars=id_vars, value_vars=val_cols, var_name="year", value_name="ex")
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long = long.dropna(subset=["year"])
    long["year"] = long["year"].astype(int)

    out = long[long["year"] == int(year)].copy()
    out = out.rename(columns={age_col: "age", sex_col: "sex"})
    out["age"] = pd.to_numeric(out["age"], errors="coerce").fillna(0).astype(int)
    out["sex"] = out["sex"].astype(str).str.strip().str.title()
    out["ex"] = pd.to_numeric(out["ex"], errors="coerce").fillna(0.0)

    if age is not None:
        out = out[out["age"] == int(age)]
    return out[["age", "sex", "ex"]].reset_index(drop=True)


def observed_e0_from_un_ex(ex_csv_path: str, year: int) -> dict:
    """Convenience: return {'Male','Female','Both'} life expectancy at birth from UN ex."""
    df = load_un_ex_from_wide(ex_csv_path, year=year, age=0)
    out = {row["sex"]: float(row["ex"]) for _, row in df.iterrows()}
    male = out.get("Male", np.nan)
    female = out.get("Female", np.nan)
    both = np.nanmean([male, female])
    return {"Male": male, "Female": female, "Both": both}


def calculate_life_expectancy(sim, year=None, max_age=100, radix=100_000):
    """
    Compute life expectancy (e₀) at birth from a completed MIGHTI simulation.

    This function wraps the mortality-rate and life-table calculations to
    provide quick access to male, female, and combined e₀ values.

    Args:
        sim: Completed simulation object.
        year (int, optional): Target year for labeling. Defaults to sim's final year.
        max_age (int): Maximum age to include.
        radix (int): Initial population base (default 100,000).

    Returns:
        dict: {'Male': e0_male, 'Female': e0_female, 'Both': e0_both}
    """
    # Find the relevant analyzer
    deaths_analyzer = None
    for a in sim.analyzers.values() if isinstance(sim.analyzers, dict) else sim.analyzers:
        if isinstance(a, type) or "death" in getattr(a, "name", "").lower():
            deaths_analyzer = a
            break
    if deaths_analyzer is None:
        raise ValueError("DeathsByAgeSexAnalyzer not found in simulation analyzers.")

    # Step 1: Compute mortality rates
    df_mx = calculate_mortality_rates(sim, deaths_analyzer, year=year, max_age=max_age, radix=radix)

    # Step 2: Compute life table
    lt = calculate_life_table_from_mx(
        sim,
        df_mx_male=df_mx[df_mx['sex'] == 'Male'],
        df_mx_female=df_mx[df_mx['sex'] == 'Female'],
        max_age=max_age,
    )

    # Step 3: Extract life expectancy at birth
    e0_male = lt.loc[(lt['sex'] == 'Male') & (lt['Age'] == 0), 'e(x)'].values[0]
    e0_female = lt.loc[(lt['sex'] == 'Female') & (lt['Age'] == 0), 'e(x)'].values[0]

    # Weighted mean for both sexes (by initial survivors)
    l0_male = lt.loc[(lt['sex'] == 'Male') & (lt['Age'] == 0), 'l(x)'].values[0]
    l0_female = lt.loc[(lt['sex'] == 'Female') & (lt['Age'] == 0), 'l(x)'].values[0]
    e0_both = (e0_male * l0_male + e0_female * l0_female) / (l0_male + l0_female)

    return {'Male': e0_male, 'Female': e0_female, 'Both': e0_both}


def calculate_life_table(sim, year=None, max_age=100, radix=100_000) -> pd.DataFrame:
    """
    Compute a full life table from a completed simulation.

    Returns a DataFrame with columns:
      ['sex', 'Age', 'l(x)', 'd(x)', 'q(x)', 'm(x)', 'L(x)', 'T(x)', 'e(x)']
    """
    deaths_analyzer = None
    for a in sim.analyzers.values() if isinstance(sim.analyzers, dict) else sim.analyzers:
        if isinstance(a, type) or "death" in getattr(a, "name", "").lower():
            deaths_analyzer = a
            break
    if deaths_analyzer is None:
        raise ValueError("DeathsByAgeSexAnalyzer not found in simulation analyzers.")

    df_mx = calculate_mortality_rates(sim, deaths_analyzer, year=year, max_age=max_age, radix=radix)
    lt = calculate_life_table_from_mx(
        sim,
        df_mx_male=df_mx[df_mx["sex"] == "Male"],
        df_mx_female=df_mx[df_mx["sex"] == "Female"],
        max_age=max_age,
    )
    return lt


# -----------------------------------------------------------------------------
# Reference life expectancy utilities (Stevens-style YLL support)
# -----------------------------------------------------------------------------
def life_table_from_mx(
    mx: np.ndarray,
    *,
    max_age: int = 100,
    radix: int = 100_000,
) -> pd.DataFrame:
    """
    Construct a standard period life table from age-specific mortality rates m(x).

    Notes
    -----
    - This is intentionally independent of `sim`/survivorship analyzers so it can be
      used to build *reference* (idealized) e(x) tables from any m(x) source.
    - Uses a simple constant-force-of-mortality approximation within each 1-year
      age interval: q(x) = 1 - exp(-m(x)).
    """
    mx = np.asarray(mx, dtype=float)
    if mx.ndim != 1:
        raise ValueError("mx must be a 1D array of length max_age+1")
    if len(mx) < (max_age + 1):
        raise ValueError(f"mx length {len(mx)} is < max_age+1 ({max_age+1})")

    mx = np.clip(mx[: max_age + 1], 0.0, np.inf)
    ages = np.arange(max_age + 1)

    # Survivorship
    lx = np.empty(max_age + 1, dtype=float)
    lx[0] = float(radix)
    for a in range(max_age):
        lx[a + 1] = lx[a] * np.exp(-mx[a])

    # Deaths and probabilities
    dx = np.empty(max_age + 1, dtype=float)
    dx[:-1] = lx[:-1] - lx[1:]
    dx[-1] = lx[-1]  # terminal open interval
    qx = 1.0 - np.exp(-mx)

    # Person-years
    Lx = np.empty(max_age + 1, dtype=float)
    Lx[:-1] = 0.5 * (lx[:-1] + lx[1:])
    # crude open-interval approximation
    Lx[-1] = (lx[-1] / mx[-1]) if mx[-1] > 0 else 0.0

    # Total person-years above age x, then remaining life expectancy
    Tx = np.zeros(max_age + 1, dtype=float)
    accum = 0.0
    for i in reversed(range(max_age + 1)):
        accum += float(Lx[i])
        Tx[i] = accum
    ex = np.divide(Tx, lx, out=np.zeros_like(Tx), where=lx > 0)

    return pd.DataFrame(
        {
            "Age": ages,
            "l(x)": lx,
            "d(x)": dx,
            "q(x)": qx,
            "m(x)": mx,
            "L(x)": Lx,
            "T(x)": Tx,
            "e(x)": ex,
        }
    )


def reference_ex_from_mx_df(
    df_mx: pd.DataFrame,
    *,
    max_age: int = 100,
    radix: int = 100_000,
) -> pd.DataFrame:
    """
    Build an age/sex-specific reference remaining life expectancy table e(x).

    Parameters
    ----------
    df_mx:
        Tidy DataFrame with at least: ['age','sex','mx'] (case-insensitive accepted).

    Returns
    -------
    DataFrame with columns: ['age','sex','ex'] for ages 0..max_age and sex in {Male,Female}.
    """
    if df_mx is None or len(df_mx) == 0:
        raise ValueError("df_mx is empty")

    cols = {c.lower(): c for c in df_mx.columns}
    for req in ("age", "sex", "mx"):
        if req not in cols:
            raise ValueError(f"df_mx must include column {req!r}")

    d = df_mx.rename(columns={cols["age"]: "age", cols["sex"]: "sex", cols["mx"]: "mx"}).copy()
    d["age"] = pd.to_numeric(d["age"], errors="coerce").fillna(0).astype(int)
    d["sex"] = d["sex"].astype(str).str.strip().str.title()
    d["mx"] = pd.to_numeric(d["mx"], errors="coerce").fillna(0.0).astype(float)

    out = []
    for sex in ("Male", "Female"):
        tmp = d[d["sex"] == sex].set_index("age").reindex(range(max_age + 1))
        mx = tmp["mx"].to_numpy(dtype=float)
        mx = np.nan_to_num(mx, nan=0.0)
        lt = life_table_from_mx(mx, max_age=max_age, radix=radix)
        out.append(pd.DataFrame({"age": lt["Age"].astype(int), "sex": sex, "ex": lt["e(x)"].astype(float)}))

    return pd.concat(out, ignore_index=True)


def make_ex_lookup(df_ex: pd.DataFrame) -> Callable[[str, float], float]:
    """
    Create a fast (sex, age)->remaining life expectancy lookup from a tidy ex table.

    Expected columns (case-insensitive): ['age','sex','ex'].
    """
    if df_ex is None or len(df_ex) == 0:
        raise ValueError("df_ex is empty")

    cols = {c.lower(): c for c in df_ex.columns}
    for req in ("age", "sex", "ex"):
        if req not in cols:
            raise ValueError(f"df_ex must include column {req!r}")

    d = df_ex.rename(columns={cols["age"]: "age", cols["sex"]: "sex", cols["ex"]: "ex"}).copy()
    d["age"] = pd.to_numeric(d["age"], errors="coerce").fillna(0).astype(int)
    d["sex"] = d["sex"].astype(str).str.strip().str.title()
    d["ex"] = pd.to_numeric(d["ex"], errors="coerce").fillna(0.0).astype(float)

    max_age = int(d["age"].max()) if len(d) else 0
    table: Dict[Tuple[str, int], float] = {(r.sex, int(r.age)): float(r.ex) for r in d.itertuples(index=False)}

    def lookup(sex: str, age: float) -> float:
        s = str(sex).strip().title()
        a = int(np.floor(float(age)))
        if a < 0:
            a = 0
        if a > max_age:
            return 0.0
        return float(table.get((s, a), 0.0))

    return lookup


def calculate_life_expectancy_from_mx_df(
    df_mx: pd.DataFrame,
    *,
    max_age: int = 100,
    radix: int = 100_000,
) -> dict:
    """
    Compute e0 from a tidy age/sex m(x) table.

    Parameters
    ----------
    df_mx:
        DataFrame with columns: ['age','sex','mx'].

    Returns
    -------
    dict: {'Male': e0_male, 'Female': e0_female, 'Both': e0_both}
    """
    df_ex = reference_ex_from_mx_df(df_mx, max_age=max_age, radix=radix)
    e0 = df_ex[df_ex["age"] == 0].set_index("sex")["ex"].to_dict()
    male = float(e0.get("Male", np.nan))
    female = float(e0.get("Female", np.nan))
    both = float(np.nanmean([male, female]))
    return {"Male": male, "Female": female, "Both": both}


def calculate_life_expectancy_from_age_sex_mx_analyzer(
    sim,
    *,
    year: int | None = None,
    max_age: int = 100,
    radix: int = 100_000,
) -> dict:
    """
    Compute e0 using realized m(x) from `AgeSexMxAnalyzer`.

    This avoids relying on `SurvivorshipAnalyzer` in long simulations with births
    and population turnover, and is the preferred method for calibration/forecast
    demos.
    """
    mx_an = None
    for a in sim.analyzers.values() if isinstance(sim.analyzers, dict) else sim.analyzers:
        if getattr(a, "name", "") == "age_sex_mx_analyzer":
            mx_an = a
            break
    if mx_an is None or not hasattr(mx_an, "to_mx_df"):
        raise ValueError("AgeSexMxAnalyzer not found in sim analyzers.")

    df_mx = mx_an.to_mx_df(year=year)
    # df_mx includes deaths/exposure columns; keep mx rows
    df_mx = df_mx[["age", "sex", "mx"]].copy()
    df_mx["mx"] = pd.to_numeric(df_mx["mx"], errors="coerce").fillna(0.0)
    return calculate_life_expectancy_from_mx_df(df_mx, max_age=max_age, radix=radix)
