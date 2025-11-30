"""
Analyzers for demographic outcomes such as age-specific deaths and survivorship.
"""

import numpy as np
import pandas as pd
import starsim as ss


__all__ = ["DeathsByAgeSexAnalyzer", "SurvivorshipAnalyzer", "ConditionAtDeathAnalyzer", "InfantDeathsAnalyzer"]

class DeathsByAgeSexAnalyzer(ss.Analyzer):
    """Tracks infant deaths and deaths by age/sex, Starsim-3.0.3 compatible."""

    def __init__(self, max_age=100, **kwargs):
        super().__init__(**kwargs)
        self.max_age = max_age
        self._dead_prev = None  # snapshot of who was dead in previous step

    def init_pre(self, sim):
        super().init_pre(sim)
        self._dead_prev = np.zeros(len(sim.people), dtype=bool)

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result("infant_deaths", label="Cumulative infant deaths", dtype=int),
            ss.Result("male_deaths_by_age",   label="Male deaths by age",   dtype=int, shape=self.max_age + 1),
            ss.Result("female_deaths_by_age", label="Female deaths by age", dtype=int, shape=self.max_age + 1),
        )

    def _ensure_size(self):
        """Resize internal arrays if population size changes (future-proof)."""
        n_now = len(self.sim.people)
        if self._dead_prev is None:
            self._dead_prev = np.zeros(n_now, dtype=bool)
        elif self._dead_prev.size != n_now:
            new = np.zeros(n_now, dtype=bool)
            n_copy = min(self._dead_prev.size, n_now)
            new[:n_copy] = self._dead_prev[:n_copy]
            self._dead_prev = new

    def step(self):
        ppl = self.sim.people
        ti = self.sim.ti
        self._ensure_size()

        # New deaths this step
        new_deaths_mask = ppl.dead & ~self._dead_prev
        new_deaths = new_deaths_mask.uids

        # Cumulative infant deaths
        self.results.infant_deaths[ti] = int(np.count_nonzero(ppl.dead[ppl.age < 1]))

        # Tally new deaths by age/sex
        if len(new_deaths):
            ages = np.clip(np.floor(ppl.age[new_deaths]).astype(int), 0, self.max_age)
            fem = ppl.female[new_deaths]

            if np.any(fem):
                idx, cnt = np.unique(ages[fem], return_counts=True)
                self.results.female_deaths_by_age[idx] += cnt
            if np.any(~fem):
                idx, cnt = np.unique(ages[~fem], return_counts=True)
                self.results.male_deaths_by_age[idx] += cnt

        # Update snapshot
        self._dead_prev = np.array(ppl.dead, dtype=bool)

    def finalize(self):
        super().finalize() 

    def finalize_results(self):
        super().finalize_results()

    def to_df(self):
        ages = np.arange(self.max_age + 1)
        return pd.concat([
            pd.DataFrame({
                "age": ages, "sex": "Male",
                "deaths": self.results.male_deaths_by_age[:],
            }),
            pd.DataFrame({
                "age": ages, "sex": "Female",
                "deaths": self.results.female_deaths_by_age[:],
            }),
        ], ignore_index=True)

class SurvivorshipAnalyzer(ss.Analyzer):
    """
    Computes survivorship l(x) by age and sex for life table construction.
    Fully compliant with Starsim analyzer lifecycle.
    """

    def __init__(self, max_age=100, **kwargs):
        super().__init__(**kwargs)
        self.name = "survivorship_analyzer"
        self.max_age = max_age
        self.survivorship_data = {sex: np.zeros(max_age + 1) for sex in ["Male", "Female"]}
        self._yearvec = None

    def init_results(self):
        """Initialize result containers."""
        super().init_results()  # <-- this line fixes the crash
        self.define_results(
            ss.Result("lx_male", label="Male survivorship l(x)", shape=self.max_age + 1, dtype=float),
            ss.Result("lx_female", label="Female survivorship l(x)", shape=self.max_age + 1, dtype=float)
        )
        self._yearvec = getattr(self.sim.t, "yearvec", None)

    def step(self):
        """Accumulate survivorship by age and sex at each step."""
        ppl = self.sim.people
        for sex in ["Male", "Female"]:
            mask = ppl.female if sex == "Female" else ~ppl.female
            alive = mask & ~ppl.dead
            ages = ppl.age[alive]
            for a in range(self.max_age):
                self.survivorship_data[sex][a] += np.sum((ages >= a) & (ages < a + 1))

    def finalize(self):
        """Normalize to l(0)=1 and copy into self.results arrays."""
        for sex in ["Male", "Female"]:
            lx = self.survivorship_data[sex].copy()
            if lx[0] > 0:
                lx /= lx[0]
            if sex == "Male":
                self.results.lx_male[:] = lx
            else:
                self.results.lx_female[:] = lx

    def finalize_results(self):
        """Required placeholder for Starsim loop."""
        pass

    def to_df(self):
        """Convert stored survivorship to tidy DataFrame."""
        records = []
        year = self.sim.t.yearvec[-1] if self._yearvec is not None else None
        for sex, key in zip(["Male", "Female"], ["lx_male", "lx_female"]):
            lx = getattr(self.results, key, np.zeros(self.max_age + 1))
            for age, val in enumerate(lx):
                records.append({
                    "year": year,
                    "age": age,
                    "sex": sex,
                    "survival": float(val)
                })
        return pd.DataFrame(records)

class ConditionAtDeathAnalyzer(ss.Analyzer):
    """
    Simple analyzer that records who died, their age/sex/YLL,
    and which non-HIV conditions they had or died from.

    Notes:
    - HIV deaths are excluded (use the HIV module directly).
    - Analyzer still records HIV-positive status at death.
    - No cause grouping logic here (handled in main scripts).
    """

    def __init__(self, conditions=None, ex_life_expectancy=80.0, **kwargs):
        super().__init__(**kwargs)
        # Store non-HIV conditions only (HIV tracked separately)
        self.conditions = [c.lower() for c in (conditions or []) if c.lower() != "hiv"]
        self.ex_life_expectancy = ex_life_expectancy
        self.records = []
        self.name = "condition_at_death_analyzer"

    def init_results(self):
        """Initialize analyzer results."""
        super().init_results()
        self.results = dict()
        self.results['n_deaths'] = 0
        self.results['by_cause'] = {}
        self.results['by_sex'] = {}
        self.results['by_age'] = {}

    def _had_condition(self, disease, uid):
        """Return True if agent had this condition at death."""
        if disease is None:
            return False
        for attr in ("infected", "affected", "active"):
            if hasattr(disease, attr):
                arr = getattr(disease, attr)
                try:
                    return bool(arr[uid])
                except Exception:
                    return False
        return False

    def _died_of_condition(self, disease, uid, ti):
        """Return True if this module has ti_dead and matches current step."""
        if disease is None or not hasattr(disease, "ti_dead"):
            return False
        ti_dead = getattr(disease, "ti_dead")
        try:
            return bool((ti_dead == ti)[uid])
        except Exception:
            raw = getattr(ti_dead, "raw", None)
            if raw is None:
                return False
            return bool(np.isfinite(raw[uid]) and raw[uid] == ti)

    def step(self):
        ppl = self.sim.people
        ti = self.sim.t.ti
        year = float(self.sim.t.yearvec[ti])
        died_now = ppl.dead.uids
        if not len(died_now):
            return

        # Reference HIV module to check HIV infection at death
        hiv_mod = getattr(self.sim.diseases, "hiv", None)

        for uid in died_now:
            age = float(ppl.age[uid])
            sex = "Female" if ppl.female[uid] else "Male"

            # Simple YLL estimate (replace later with life-table lookup if needed)
            le = 75 if sex == "Female" else 70
            if isinstance(self.ex_life_expectancy, (int, float)):
                le = float(self.ex_life_expectancy)
            yll = max(0.0, le - age)

            rec = dict(uid=int(uid), year=year, age=age, sex=sex, yll=yll)

            # Record presence and cause flags for each disease
            for cond in self.conditions:
                disease = getattr(self.sim.diseases, cond, None)
                rec[f"died_{cond}"] = self._had_condition(disease, uid)
                rec[f"cause_{cond}"] = self._died_of_condition(disease, uid, ti)

            # Record HIV infection status (not death cause)
            rec["hiv_positive"] = False
            if hiv_mod is not None and hasattr(hiv_mod, "infected"):
                try:
                    rec["hiv_positive"] = bool(hiv_mod.infected[uid])
                except Exception:
                    pass

            # Determine primary cause (non-HIV only)
            causes = [c for c in self.conditions if rec.get(f"cause_{c}", False)]
            if len(causes) == 1:
                rec["primary_cause"] = causes[0]
            elif len(causes) > 1:
                rec["primary_cause"] = "multiple"
            else:
                rec["primary_cause"] = None

            self.records.append(rec)

    def to_df(self):
        """Return DataFrame of recorded deaths and conditions."""
        return pd.DataFrame(self.records)


class InfantDeathsAnalyzer(ss.Analyzer):
    """
    Analyzer specifically for tracking neonatal and infant deaths.
    
    Tracks:
    - Neonatal deaths (age < 28 days)
    - Infant deaths (age < 1 year)
    - Deaths by cause (which neonatal/congenital disease)
    - Person-years at risk for infants
    
    This analyzer is separate from SurvivorshipAnalyzer to avoid interfering
    with the main survivorship calculation while still tracking early deaths.
    """
    
    def __init__(self, max_age=100, **kwargs):
        super().__init__(**kwargs)
        self.name = "infant_deaths_analyzer"
        self.max_age = max_age
        self._dead_prev = None
        self._neonatal_diseases = []  # Will be populated in init_pre
        
    def init_pre(self, sim):
        super().init_pre(sim)
        self._dead_prev = np.zeros(len(sim.people), dtype=bool)
        
        # Identify neonatal/congenital diseases
        self._neonatal_diseases = []
        if hasattr(sim, 'diseases'):
            from mighti.diseases.base_disease import NonAcquiredDisease, StaticCondition
            for name, disease in sim.diseases.items():
                # Check if it's a NonAcquiredDisease or StaticCondition
                # StaticCondition is a subclass of NonAcquiredDisease, but we check both explicitly
                is_non_acquired = isinstance(disease, NonAcquiredDisease)
                is_static = isinstance(disease, StaticCondition)
                
                if is_non_acquired or is_static:
                    if name not in self._neonatal_diseases:
                        self._neonatal_diseases.append(name)
                # Also check is_neonatal flag for explicit marking
                elif hasattr(disease, 'is_neonatal') and disease.is_neonatal:
                    if name not in self._neonatal_diseases:
                        self._neonatal_diseases.append(name)
    
    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result("neonatal_deaths_male", dtype=int, label="Neonatal deaths (age < 28 days), male"),
            ss.Result("neonatal_deaths_female", dtype=int, label="Neonatal deaths (age < 28 days), female"),
            ss.Result("infant_deaths_male", dtype=int, label="Infant deaths (age < 1 year), male"),
            ss.Result("infant_deaths_female", dtype=int, label="Infant deaths (age < 1 year), female"),
            ss.Result("infant_person_years_male", dtype=float, label="Person-years at age < 1, male"),
            ss.Result("infant_person_years_female", dtype=float, label="Person-years at age < 1, female"),
        )
        # Initialize deaths by cause dictionary (not a Result, just a regular attribute)
        self._neonatal_deaths_by_cause = {}
    
    def _ensure_size(self):
        """Resize internal arrays if population size changes."""
        n_now = len(self.sim.people)
        if self._dead_prev is None:
            self._dead_prev = np.zeros(n_now, dtype=bool)
        elif self._dead_prev.size != n_now:
            new = np.zeros(n_now, dtype=bool)
            n_copy = min(self._dead_prev.size, n_now)
            new[:n_copy] = self._dead_prev[:n_copy]
            self._dead_prev = new
    
    def step(self):
        """Track neonatal and infant deaths at each step."""
        ppl = self.sim.people
        ti = self.sim.ti
        self._ensure_size()
        
        # New deaths this step
        new_deaths_mask = ppl.dead & ~self._dead_prev
        new_deaths = new_deaths_mask.uids
        
        if len(new_deaths) == 0:
            self._dead_prev = np.array(ppl.dead, dtype=bool)
            return
        
        # Get ages and sex for new deaths
        ages = ppl.age[new_deaths]
        female = ppl.female[new_deaths]
        
        # Neonatal deaths (age < 28 days = 28/365 years)
        neonatal_threshold = 28 / 365.0
        neonatal_mask = ages < neonatal_threshold
        
        if np.any(neonatal_mask):
            neonatal_deaths = new_deaths[neonatal_mask]
            neonatal_female = female[neonatal_mask]
            
            # Count by sex
            self.results.neonatal_deaths_male[ti] = int(np.sum(~neonatal_female))
            self.results.neonatal_deaths_female[ti] = int(np.sum(neonatal_female))
            
            # Track by cause (which neonatal disease caused the death)
            for disease_name in self._neonatal_diseases:
                disease = getattr(self.sim.diseases, disease_name, None)
                if disease is None:
                    continue
                
                # Check if death was caused by this disease
                # StaticCondition and NonAcquiredDisease both have ti_dead attribute
                if hasattr(disease, 'ti_dead'):
                    ti_dead = getattr(disease, 'ti_dead', None)
                    if ti_dead is not None:
                        # Check if any of the neonatal deaths have ti_dead matching this disease
                        try:
                            # Try to access .raw first (for FloatArr states)
                            if hasattr(ti_dead, 'raw'):
                                died_from_disease = np.array([
                                    np.isfinite(ti_dead.raw[uid]) and ti_dead.raw[uid] == ti 
                                    for uid in neonatal_deaths
                                ], dtype=bool)
                            else:
                                # Fallback: direct access
                                died_from_disease = np.array([
                                    np.isfinite(ti_dead[uid]) and ti_dead[uid] == ti 
                                    for uid in neonatal_deaths
                                ], dtype=bool)
                            
                            if np.any(died_from_disease):
                                if disease_name not in self._neonatal_deaths_by_cause:
                                    self._neonatal_deaths_by_cause[disease_name] = 0
                                self._neonatal_deaths_by_cause[disease_name] += int(np.sum(died_from_disease))
                        except Exception as e:
                            # Debug: print error if needed
                            pass
        
        # Infant deaths (age < 1 year)
        infant_mask = ages < 1.0
        if np.any(infant_mask):
            infant_deaths = new_deaths[infant_mask]
            infant_female = female[infant_mask]
            
            self.results.infant_deaths_male[ti] = int(np.sum(~infant_female))
            self.results.infant_deaths_female[ti] = int(np.sum(infant_female))
        
        # Accumulate person-years for infants (alive people at age < 1)
        alive = ~ppl.dead
        infant_alive = ppl.age[alive] < 1.0
        if np.any(infant_alive):
            infant_alive_uids = np.where(alive)[0][infant_alive]
            infant_female_alive = ppl.female[infant_alive_uids]
            
            # Person-years = count of people (assuming 1 timestep = some fraction of a year)
            # We'll accumulate this over all timesteps
            dt = self.sim.t.dt if hasattr(self.sim.t, 'dt') else 1.0 / 365.0  # Default to daily
            self.results.infant_person_years_male[ti] = float(np.sum(~infant_female_alive) * dt)
            self.results.infant_person_years_female[ti] = float(np.sum(infant_female_alive) * dt)
        
        # Update snapshot
        self._dead_prev = np.array(ppl.dead, dtype=bool)
    
    def finalize(self):
        """Compute cumulative totals."""
        super().finalize()
        # Results are already cumulative from step(), but we can add summary here if needed
        pass
    
    def finalize_results(self):
        super().finalize_results()
    
    def to_df(self):
        """Convert results to DataFrame."""
        records = []
        
        # Get cumulative totals
        total_neonatal_m = int(np.sum(self.results.neonatal_deaths_male[:]))
        total_neonatal_f = int(np.sum(self.results.neonatal_deaths_female[:]))
        total_infant_m = int(np.sum(self.results.infant_deaths_male[:]))
        total_infant_f = int(np.sum(self.results.infant_deaths_female[:]))
        total_py_m = float(np.sum(self.results.infant_person_years_male[:]))
        total_py_f = float(np.sum(self.results.infant_person_years_female[:]))
        
        records.append({
            'category': 'neonatal',
            'sex': 'Male',
            'deaths': total_neonatal_m,
            'person_years': total_py_m if total_py_m > 0 else 0.0,
            'mortality_rate': total_neonatal_m / total_py_m if total_py_m > 0 else 0.0
        })
        records.append({
            'category': 'neonatal',
            'sex': 'Female',
            'deaths': total_neonatal_f,
            'person_years': total_py_f if total_py_f > 0 else 0.0,
            'mortality_rate': total_neonatal_f / total_py_f if total_py_f > 0 else 0.0
        })
        records.append({
            'category': 'infant',
            'sex': 'Male',
            'deaths': total_infant_m,
            'person_years': total_py_m if total_py_m > 0 else 0.0,
            'mortality_rate': total_infant_m / total_py_m if total_py_m > 0 else 0.0
        })
        records.append({
            'category': 'infant',
            'sex': 'Female',
            'deaths': total_infant_f,
            'person_years': total_py_f if total_py_f > 0 else 0.0,
            'mortality_rate': total_infant_f / total_py_f if total_py_f > 0 else 0.0
        })
        
        # Add deaths by cause
        for disease_name, count in self._neonatal_deaths_by_cause.items():
            records.append({
                'category': 'neonatal_by_cause',
                'sex': 'Both',
                'disease': disease_name,
                'deaths': count,
                'person_years': 0.0,
                'mortality_rate': 0.0
            })
        
        return pd.DataFrame(records)
    