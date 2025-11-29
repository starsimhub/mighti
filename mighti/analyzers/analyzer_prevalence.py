"""
Analyzer for tracking and recording disease prevalence over time.

This analyzer computes stratified disease prevalence by:
- Age group (in 5-year bins up to 80+)
- Sex (male/female)
- HIV status (with/without HIV)

It stores both counts and prevalence estimates for:
- Total population
- Population stratified by sex and HIV status
- Each age-sex-HIV-status group
"""


import starsim as ss
import numpy as np
import sciris as sc

__all__ = ["PrevalenceAnalyzer", "PrevalenceAnalyzer_HIV", "PrevalenceAnalyzer_SDoH", "OnARTByConditionAnalyzer"]



# ---------------------------------------------------------------------
# Generic Prevalence Analyzer
# ---------------------------------------------------------------------
class PrevalenceAnalyzer(ss.Analyzer):
    """
    General prevalence analyzer that tracks prevalence by sex and age for all diseases.
    Now fully compatible with Starsim 3.0+.
    """

    def __init__(self, prevalence_data=None, diseases=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "prevalence_analyzer"
        self.prevalence_data = prevalence_data
        self.diseases = diseases or []
        self.age_bins = [
            (0,15),(15,20),(20,25),(25,30),(30,35),(35,40),
            (40,45),(45,50),(50,55),(55,60),(60,65),
            (65,70),(70,75),(75,80),(80,float('inf'))
        ]
        self.results_defined = False

    def init_results(self):
        """Initialize the analyzer's results storage."""
        super().init_results()
        if not hasattr(self, "results"):
            self.results = sc.odict()
        if self.results_defined:
            return

        existing = set(self.results.keys())
        new_results = []
        self.diseases = [d.lower() for d in self.diseases]

        for disease in self.diseases:
            # Age-sex bins
            for i, _ in enumerate(self.age_bins):
                for sex in ["male", "female"]:
                    for suffix in ["num", "den", "prev"]:
                        key = f"{disease}_{suffix}_{sex}_{i}"
                        if key not in existing:
                            dtype = float if suffix == "prev" else int
                            new_results.append(ss.Result(key, dtype=dtype, scale=False))
                            existing.add(key)

            # Total-level results
            for suffix in ["num_total", "den_total", "prev_total"]:
                key = f"{disease}_{suffix}"
                if key not in existing:
                    dtype = float if "prev" in key else int
                    new_results.append(ss.Result(key, dtype=dtype, scale=False))
                    existing.add(key)

        if new_results:
            self.define_results(*new_results)
        self.results_defined = True

    def step(self):
        """Compute disease prevalence for each age-sex bin."""
        sim = self.sim
        ti  = self.ti
        ppl = sim.people
        self.diseases = [d.lower() for d in self.diseases]

        for disease in self.diseases:
            dis = getattr(sim.diseases, disease, None)
            if dis is None:
                continue

            # Dynamically detect which attribute to use (infected vs affected)
            if hasattr(dis, "infected"):
                status_attr = "infected"
            elif hasattr(dis, "affected"):
                status_attr = "affected"
            else:
                # Fallback: try common names
                status_attr = "infected" if disease in ["hiv","hpv","flu","viralhepatitis","tb","lowerrespiratoryinfections"] else "affected"
            has_disease = getattr(dis, status_attr)

            # Total prevalence
            num_total = np.sum(has_disease)
            den_total = len(ppl)
            self.results[f"{disease}_num_total"][ti] = num_total
            self.results[f"{disease}_den_total"][ti] = den_total
            self.results[f"{disease}_prev_total"][ti] = sc.safedivide(num_total, den_total)

            # Age-sex stratified prevalence
            for i, (a0, a1) in enumerate(self.age_bins):
                age_group = (ppl.age >= a0) & (ppl.age < a1)
                for sex, mask_sex in zip(["male", "female"], [ppl.male, ppl.female]):
                    mask = age_group & mask_sex
                    num = np.sum(mask & has_disease)
                    den = np.sum(mask)
                    self.results[f"{disease}_num_{sex}_{i}"][ti]  = num
                    self.results[f"{disease}_den_{sex}_{i}"][ti]  = max(den, 1)
                    self.results[f"{disease}_prev_{sex}_{i}"][ti] = sc.safedivide(num, den)


# ---------------------------------------------------------------------
# Shared utility for defining summary-level results
# ---------------------------------------------------------------------
def _add_summary_results(analyzer, disease, existing, new_results, suffixes):
    """
    Register summary-level prevalence and count results safely.
    `suffixes` should be a list of name endings, e.g.,
        ['prev_no_hiv', 'prev_has_hiv', 'num_total', 'den_total'].
    """
    for suffix in suffixes:
        key = f"{disease}_{suffix}"
        if key not in existing:
            dtype = float if "prev" in suffix else int
            new_results.append(ss.Result(key, dtype=dtype, scale=False))
            existing.add(key)
    return existing, new_results


# ---------------------------------------------------------------------
# HIV-Stratified Prevalence Analyzer
# ---------------------------------------------------------------------
class PrevalenceAnalyzer_HIV(ss.Analyzer):
    """Tracks prevalence stratified by HIV status, sex, and age."""

    @staticmethod
    def cond_prob(numerator, denominator):
        """Return conditional probability for sciris People-like masks."""
        numer = np.sum(numerator & denominator)
        denom = np.sum(denominator)
        return sc.safedivide(numer, denom)

    def __init__(self, prevalence_data=None, diseases=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "prevalence_analyzer_hiv"
        self.prevalence_data = prevalence_data
        self.diseases = [d.lower() for d in (diseases or [])]
        self.age_bins = [(0,15),(15,20),(20,25),(25,30),(30,35),(35,40),(40,45),
                         (45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80),(80,float("inf"))]
        self.results_defined = False

    def init_results(self):
        """Initialize all result arrays for each disease and HIV stratum."""
        super().init_results()
        if not hasattr(self, "results"):
            self.results = sc.odict()
        if self.results_defined:
            return

        existing = set(self.results.keys())
        new_results = []

        for disease in self.diseases:
            # Per-age-sex bins
            for i, _ in enumerate(self.age_bins):
                for sex in ["male", "female"]:
                    for key in [
                        f"{disease}_num_{sex}_{i}", f"{disease}_den_{sex}_{i}", f"{disease}_prev_{sex}_{i}",
                        f"{disease}_num_with_HIV_{sex}_{i}", f"{disease}_den_with_HIV_{sex}_{i}", f"{disease}_prev_with_HIV_{sex}_{i}",
                        f"{disease}_num_without_HIV_{sex}_{i}", f"{disease}_den_without_HIV_{sex}_{i}", f"{disease}_prev_without_HIV_{sex}_{i}",
                    ]:
                        if key not in existing:
                            dtype = float if "prev" in key else int
                            new_results.append(ss.Result(key, dtype=dtype, scale=False))
                            existing.add(key)

            # Add summary-level results (fix for Fig2)
            summary_suffixes = [
                "prev_no_hiv", "prev_has_hiv",
                "prev_no_hiv_f", "prev_has_hiv_f",
                "prev_no_hiv_m", "prev_has_hiv_m",
                "num_total", "den_total"
            ]
            existing, new_results = _add_summary_results(self, disease, existing, new_results, summary_suffixes)

        if new_results:
            self.define_results(*new_results)
        self.results_defined = True

    def step(self):
        sim = self.sim
        ti = self.ti
        ppl = sim.people
        hiv = sim.diseases.hiv

        has_hiv = hiv.infected
        no_hiv = hiv.susceptible
        male, female = ppl.male, ppl.female

        for disease in self.diseases:
            dis = getattr(sim.diseases, disease)
            # Dynamically detect which attribute to use (infected vs affected)
            if hasattr(dis, "infected"):
                status_attr = "infected"
            elif hasattr(dis, "affected"):
                status_attr = "affected"
            else:
                # Fallback: try common names
                status_attr = "infected" if disease in ["hiv", "hpv", "flu", "viralhepatitis", "tb", "diarrhealdisease", "lowerrespiratoryinfections"] else "affected"
            has_disease = getattr(dis, status_attr)

            # Track stratified prevalences
            total_num_with_HIV = total_den_with_HIV = 0

            for i, (a0, a1) in enumerate(self.age_bins):
                age_group = (ppl.age >= a0) & (ppl.age < a1)
                for sex, mask_sex in zip(["male", "female"], [male, female]):
                    mask = age_group & mask_sex
                    num = np.sum(mask & has_disease)
                    den = np.sum(mask)
                    self.results[f"{disease}_num_{sex}_{i}"][ti]  = num
                    self.results[f"{disease}_den_{sex}_{i}"][ti]  = max(den, 1)
                    self.results[f"{disease}_prev_{sex}_{i}"][ti] = sc.safedivide(num, den)

                    for status, status_mask in [("with_HIV", has_hiv), ("without_HIV", no_hiv)]:
                        submask = mask & status_mask
                        num_s = np.sum(submask & has_disease)
                        den_s = np.sum(submask)
                        self.results[f"{disease}_num_{status}_{sex}_{i}"][ti]  = num_s
                        self.results[f"{disease}_den_{status}_{sex}_{i}"][ti]  = max(den_s, 1)
                        self.results[f"{disease}_prev_{status}_{sex}_{i}"][ti] = sc.safedivide(num_s, den_s)
                        if status == "with_HIV":
                            total_num_with_HIV += num_s
                            total_den_with_HIV += den_s

            # Safe conditional probabilities
            self.results[f"{disease}_prev_no_hiv"][ti]   = self.cond_prob(has_disease, no_hiv)
            self.results[f"{disease}_prev_has_hiv"][ti]  = self.cond_prob(has_disease, has_hiv)
            self.results[f"{disease}_prev_no_hiv_f"][ti] = self.cond_prob(has_disease & female, no_hiv & female)
            self.results[f"{disease}_prev_has_hiv_f"][ti]= self.cond_prob(has_disease & female, has_hiv & female)
            self.results[f"{disease}_prev_no_hiv_m"][ti] = self.cond_prob(has_disease & male, no_hiv & male)
            self.results[f"{disease}_prev_has_hiv_m"][ti]= self.cond_prob(has_disease & male, has_hiv & male)
            self.results[f"{disease}_num_total"][ti] = total_num_with_HIV
            self.results[f"{disease}_den_total"][ti] = max(total_den_with_HIV, 1)


# ---------------------------------------------------------------------
# SDoH-Stratified Prevalence Analyzer (updated)
# ---------------------------------------------------------------------
class PrevalenceAnalyzer_SDoH(ss.Analyzer):
    """Stratifies prevalence by a binary SDoH attribute (e.g., housed/unhoused)."""

    @staticmethod
    def cond_prob_bool(numerator, denominator):
        numer = np.sum(numerator & denominator)
        denom = np.sum(denominator)
        return sc.safedivide(numer, denom)

    def __init__(self, prevalence_data=None, diseases=None, sdoh_attr="neighbourhood_situation", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "prevalence_analyzer_sdoh"
        self.prevalence_data = prevalence_data
        self.diseases = [d.lower() for d in (diseases or [])]
        self.sdoh_attr = sdoh_attr
        self.age_bins = [(0,15),(15,20),(20,25),(25,30),(30,35),(35,40),(40,45),
                         (45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80),(80,float("inf"))]
        self.results_defined = False

    def init_results(self):
        super().init_results()
        if not hasattr(self, "results"):
            self.results = sc.odict()
        if self.results_defined:
            return

        existing = set(self.results.keys())
        new_results = []

        if "sdoh_prop_adults" not in existing:
            new_results.append(ss.Result("sdoh_prop_adults", dtype=float, scale=False))

        for disease in self.diseases:
            for i, _ in enumerate(self.age_bins):
                for sex in ["male", "female"]:
                    for key in [
                        f"{disease}_num_{sex}_{i}", f"{disease}_den_{sex}_{i}", f"{disease}_prev_{sex}_{i}",
                        f"{disease}_num_with_SDoH_{sex}_{i}", f"{disease}_den_with_SDoH_{sex}_{i}", f"{disease}_prev_with_SDoH_{sex}_{i}",
                        f"{disease}_num_without_SDoH_{sex}_{i}", f"{disease}_den_without_SDoH_{sex}_{i}", f"{disease}_prev_without_SDoH_{sex}_{i}",
                    ]:
                        if key not in existing:
                            dtype = float if "prev" in key else int
                            new_results.append(ss.Result(key, dtype=dtype, scale=False))
                            existing.add(key)

            # Add summary-level results (same idea as HIV analyzer)
            summary_suffixes = [
                "prev_no_sdoh", "prev_has_sdoh",
                "prev_no_sdoh_f", "prev_has_sdoh_f",
                "prev_no_sdoh_m", "prev_has_sdoh_m",
                "num_total", "den_total"
            ]
            existing, new_results = _add_summary_results(self, disease, existing, new_results, summary_suffixes)

        if new_results:
            self.define_results(*new_results)
        self.results_defined = True



class OnARTByConditionAnalyzer(ss.Analyzer):
    """Tracks ART coverage among HIV+ individuals, stratified by condition (e.g., depression)."""

    @staticmethod
    def cond_prob(num, den):
        return sc.safedivide(np.sum(num & den), np.sum(den))

    def __init__(self, condition_key="majordepressivedisorder.affected", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"onart_{condition_key.replace('.', '_')}"
        self.condition_key = condition_key
        self.results_defined = False

    def init_results(self):
        super().init_results()
        if self.results_defined:
            return
        results = [
            ss.Result("onart_with_condition", dtype=float),
            ss.Result("onart_without_condition", dtype=float),
        ]
        self.define_results(*results)
        self.results_defined = True

    def step(self):
        ppl = self.sim.people
        cond = np.asarray(ppl.states.get(self.condition_key), dtype=bool)
        hiv  = np.asarray(ppl.states.get("hiv.infected"), dtype=bool)
        art  = np.asarray(ppl.states.get("hiv.on_art"), dtype=bool)
        ti = self.ti
        self.results["onart_with_condition"][ti]    = self.cond_prob(art, hiv & cond)
        self.results["onart_without_condition"][ti] = self.cond_prob(art, hiv & ~cond)
        