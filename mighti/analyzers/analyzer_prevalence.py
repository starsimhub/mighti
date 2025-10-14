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

__all__ = ["PrevalenceAnalyzer", "PrevalenceAnalyzer_HIV", "PrevalenceAnalyzer_SDoH"]

class PrevalenceAnalyzer(ss.Analyzer):
    """Simple prevalence analyzer that tracks prevalence by sex and age for all diseases.
    Compatible with Starsim 3.0+ (prevents duplicate result key errors).
    """

    def __init__(self, prevalence_data=None, diseases=None, *args, **kwargs):
        super().__init__(*args)
        self.name = 'prevalence_analyzer_general'
        self.prevalence_data = prevalence_data
        self.diseases = diseases or []
        self.age_bins = [
            (0,15),(15,20),(20,25),(25,30),(30,35),(35,40),
            (40,45),(45,50),(50,55),(55,60),(60,65),
            (65,70),(70,75),(75,80),(80,float('inf'))
        ]
        self.results_defined = False

    def init_results(self):
        """Define results safely, skipping already existing ones."""
        if self.results_defined:
            return

        existing = set(self.results.keys())
        new_results = []

        for disease in self.diseases:
            for i, _ in enumerate(self.age_bins):
                for sex in ['male', 'female']:
                    base_keys = [
                        f'{disease}_num_{sex}_{i}',
                        f'{disease}_den_{sex}_{i}',
                        f'{disease}_prev_{sex}_{i}',
                    ]
                    for key in base_keys:
                        if key not in existing:
                            dtype = float if "prev" in key else int
                            new_results.append(ss.Result(key, dtype=dtype, scale=False))
                            existing.add(key)

            # Add total results (non-stratified)
            for key in [
                f'{disease}_num_total',
                f'{disease}_den_total',
                f'{disease}_prev_total',
            ]:
                if key not in existing:
                    dtype = float if "prev" in key else int
                    new_results.append(ss.Result(key, dtype=dtype, scale=False))
                    existing.add(key)

        if new_results:
            self.define_results(*new_results)
        self.results_defined = True

    def step(self):
        """Compute prevalence by age and sex for each disease."""
        sim = self.sim
        ti = self.ti
        ppl = sim.people

        for disease in self.diseases:
            # Determine disease attribute: 'infected' for communicable, 'affected' for chronic/NCD
            dis = getattr(sim.diseases, disease.lower())
            status_attr = 'infected' if disease.lower() in ['hiv','hpv','flu','viralhepatitis','tb'] else 'affected'
            has_disease = getattr(dis, status_attr)

            # --- Total prevalence ---
            num_total = np.sum(has_disease)
            den_total = len(ppl)
            self.results[f'{disease}_num_total'][ti] = num_total
            self.results[f'{disease}_den_total'][ti] = den_total
            self.results[f'{disease}_prev_total'][ti] = sc.safedivide(num_total, den_total)

            # --- Age-sex stratified prevalence ---
            for i, (a0, a1) in enumerate(self.age_bins):
                age_group = (ppl.age >= a0) & (ppl.age < a1)
                for sex, sexmask in zip(['male', 'female'], [ppl.male, ppl.female]):
                    mask = age_group & sexmask
                    num = np.sum(mask & has_disease)
                    den = np.sum(mask)
                    self.results[f'{disease}_num_{sex}_{i}'][ti] = num
                    self.results[f'{disease}_den_{sex}_{i}'][ti] = max(den, 1)
                    self.results[f'{disease}_prev_{sex}_{i}'][ti] = sc.safedivide(num, den)
                    

class PrevalenceAnalyzer_HIV(ss.Analyzer):
    """
    Analyzer for disease prevalence stratified by HIV status, sex, and age.
    Compatible with Starsim 3.0+ (prevents duplicate result key errors).
    """

    @staticmethod
    def cond_prob(numerator, denominator):
        numer = len((denominator & numerator).uids)
        denom = len(denominator.uids)
        return sc.safedivide(numer, denom)

    def __init__(self, prevalence_data, diseases=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'prevalence_analyzer_hiv'
        self.prevalence_data = prevalence_data
        self.diseases = diseases or []

        # Define age bins
        self.age_bins = [
            (0, 15), (15, 20), (20, 25), (25, 30), (30, 35),
            (35, 40), (40, 45), (45, 50), (50, 55), (55, 60),
            (60, 65), (65, 70), (70, 75), (75, 80), (80, float('inf'))
        ]

        self.results_defined = False

    def init_results(self):
        """Define all results safely, skipping duplicates if already defined."""
        if self.results_defined:
            return

        existing = set(self.results.keys())  # existing result keys
        new_results = []

        for disease in self.diseases:
            for i, (age_start, age_end) in enumerate(self.age_bins):
                for sex in ['male', 'female']:
                    base_keys = [
                        f'{disease}_num_{sex}_{i}',
                        f'{disease}_den_{sex}_{i}',
                        f'{disease}_prev_{sex}_{i}',
                        f'{disease}_num_with_HIV_{sex}_{i}',
                        f'{disease}_den_with_HIV_{sex}_{i}',
                        f'{disease}_prev_with_HIV_{sex}_{i}',
                        f'{disease}_num_without_HIV_{sex}_{i}',
                        f'{disease}_den_without_HIV_{sex}_{i}',
                        f'{disease}_prev_without_HIV_{sex}_{i}',
                    ]
                    for key in base_keys:
                        if key not in existing:
                            new_results.append(ss.Result(key, dtype=float if "prev" in key else int, scale=False))
                            existing.add(key)

            # Add total and summary prevalence results
            summary_keys = [
                f'{disease}_prev_no_hiv', f'{disease}_prev_has_hiv',
                f'{disease}_prev_no_hiv_f', f'{disease}_prev_has_hiv_f',
                f'{disease}_prev_no_hiv_m', f'{disease}_prev_has_hiv_m',
                f'{disease}_num_total', f'{disease}_den_total',
            ]
            for key in summary_keys:
                if key not in existing:
                    new_results.append(ss.Result(key, dtype=float if "prev" in key else int, scale=False))
                    existing.add(key)

        if new_results:
            self.define_results(*new_results)

        self.results_defined = True

    def init_pre(self, sim):
        """Initialize any pre-simulation storage."""
        super().init_pre(sim)
        npts = len(sim.t)
        # Age distribution (0–100 years, yearly resolution)
        self.results['population_age_distribution'] = np.zeros((npts, 101))
        return

    def step(self):
        """Compute stratified prevalence at each timestep."""
        sim = self.sim
        ti = self.ti
        ppl = sim.people
        hiv = sim.diseases.hiv

        denom = (ppl.age >= 0)
        has_hiv = denom & hiv.infected
        no_hiv = denom & hiv.susceptible

        for disease in self.diseases:
            dis = getattr(sim.diseases, disease.lower())
            status_attr = 'infected' if disease.lower() in ['hiv', 'hpv', 'flu', 'viralhepatitis', 'tb'] else 'affected'
            has_disease = denom & getattr(dis, status_attr)

            has_disease_f = has_disease & ppl.female
            has_disease_m = has_disease & ppl.male
            has_hiv_f = has_hiv & ppl.female
            has_hiv_m = has_hiv & ppl.male
            no_hiv_f = no_hiv & ppl.female
            no_hiv_m = no_hiv & ppl.male

            total_num_with_HIV = 0
            total_den_with_HIV = 0

            for i, (a0, a1) in enumerate(self.age_bins):
                age_group = (ppl.age >= a0) & (ppl.age < a1)

                num_male = np.sum(age_group & has_disease_m)
                den_male = np.sum(age_group & ppl.male)
                num_female = np.sum(age_group & has_disease_f)
                den_female = np.sum(age_group & ppl.female)
                num_with_HIV_male = np.sum(age_group & has_disease_m & has_hiv)
                den_with_HIV_male = np.sum(age_group & ppl.male & has_hiv)
                num_with_HIV_female = np.sum(age_group & has_disease_f & has_hiv)
                den_with_HIV_female = np.sum(age_group & ppl.female & has_hiv)
                num_without_HIV_male = np.sum(age_group & has_disease_m & no_hiv)
                den_without_HIV_male = np.sum(age_group & ppl.male & no_hiv)
                num_without_HIV_female = np.sum(age_group & has_disease_f & no_hiv)
                den_without_HIV_female = np.sum(age_group & ppl.female & no_hiv)

                total_num_with_HIV += num_with_HIV_male + num_with_HIV_female
                total_den_with_HIV += den_with_HIV_male + den_with_HIV_female

                for sex, num, den, num_w, den_w, num_wo, den_wo in zip(
                    ['male', 'female'],
                    [num_male, num_female],
                    [den_male, den_female],
                    [num_with_HIV_male, num_with_HIV_female],
                    [den_with_HIV_male, den_with_HIV_female],
                    [num_without_HIV_male, num_without_HIV_female],
                    [den_without_HIV_male, den_without_HIV_female],
                ):
                    self.results[f'{disease}_num_{sex}_{i}'][ti] = num
                    self.results[f'{disease}_den_{sex}_{i}'][ti] = den
                    self.results[f'{disease}_prev_{sex}_{i}'][ti] = sc.safedivide(num, den)
                    self.results[f'{disease}_num_with_HIV_{sex}_{i}'][ti] = num_w
                    self.results[f'{disease}_den_with_HIV_{sex}_{i}'][ti] = den_w
                    self.results[f'{disease}_prev_with_HIV_{sex}_{i}'][ti] = sc.safedivide(num_w, den_w)
                    self.results[f'{disease}_num_without_HIV_{sex}_{i}'][ti] = num_wo
                    self.results[f'{disease}_den_without_HIV_{sex}_{i}'][ti] = den_wo
                    self.results[f'{disease}_prev_without_HIV_{sex}_{i}'][ti] = sc.safedivide(num_wo, den_wo)

            # Summary prevalences (conditional probabilities)
            self.results[f'{disease}_prev_no_hiv'][ti] = self.cond_prob(has_disease, no_hiv)
            self.results[f'{disease}_prev_has_hiv'][ti] = self.cond_prob(has_disease, has_hiv)
            self.results[f'{disease}_prev_no_hiv_f'][ti] = self.cond_prob(has_disease_f, no_hiv_f)
            self.results[f'{disease}_prev_has_hiv_f'][ti] = self.cond_prob(has_disease_f, has_hiv_f)
            self.results[f'{disease}_prev_no_hiv_m'][ti] = self.cond_prob(has_disease_m, no_hiv_m)
            self.results[f'{disease}_prev_has_hiv_m'][ti] = self.cond_prob(has_disease_m, has_hiv_m)

            self.results[f'{disease}_num_total'][ti] = total_num_with_HIV
            self.results[f'{disease}_den_total'][ti] = total_den_with_HIV

class PrevalenceAnalyzer_SDoH(ss.Analyzer):
    """
    Stratifies disease prevalence by an SDoH boolean (e.g., housing), sex, age.
    """

    def initialize(self, sim):
        super().initialize(sim)
        self.sim = sim
        self.init_results()

    @staticmethod
    def cond_prob_bool(numerator_bool, denominator_bool):
        # Works on NumPy bool arrays or any array-like that casts to boolean
        num = int(np.sum(numerator_bool & denominator_bool))
        den = int(np.sum(denominator_bool))
        return sc.safedivide(num, den)

    def __init__(self, prevalence_data, diseases=None, sdoh_attr="neighbourhood_situation", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "prevalence_analyzer_sdoh"
        self.prevalence_data = prevalence_data
        self.diseases = diseases or []
        self.sdoh_attr = sdoh_attr
        self.age_bins = [
            (0,15),(15,20),(20,25),(25,30),(30,35),
            (35,40),(40,45),(45,50),(50,55),(55,60),
            (60,65),(65,70),(70,75),(75,80),(80,float("inf"))
        ]
        self.results_defined = False

    def init_results(self):
        if self.results_defined:
            return
        
        if "sdoh_prop_adults" not in self.results:
            self.define_results(ss.Result("sdoh_prop_adults", dtype=float, scale=False))

        existing = set(self.results.keys())
        new = []

        for disease in self.diseases:
            for i, _ in enumerate(self.age_bins):
                for sex in ("male","female"):
                    for key in (
                        f"{disease}_num_{sex}_{i}",
                        f"{disease}_den_{sex}_{i}",
                        f"{disease}_prev_{sex}_{i}",
                        f"{disease}_num_with_SDoH_{sex}_{i}",
                        f"{disease}_den_with_SDoH_{sex}_{i}",
                        f"{disease}_prev_with_SDoH_{sex}_{i}",
                        f"{disease}_num_without_SDoH_{sex}_{i}",
                        f"{disease}_den_without_SDoH_{sex}_{i}",
                        f"{disease}_prev_without_SDoH_{sex}_{i}",
                    ):
                        if key not in existing:
                            new.append(ss.Result(key, dtype=float if "prev" in key else int, scale=False))
                            existing.add(key)

            for key in (
                f"{disease}_prev_no_sdoh", f"{disease}_prev_has_sdoh",
                f"{disease}_prev_no_sdoh_f", f"{disease}_prev_has_sdoh_f",
                f"{disease}_prev_no_sdoh_m", f"{disease}_prev_has_sdoh_m",
                f"{disease}_num_total", f"{disease}_den_total",
            ):
                if key not in existing:
                    new.append(ss.Result(key, dtype=float if "prev" in key else int, scale=False))
                    existing.add(key)

        if new:
            self.define_results(*new)
        self.results_defined = True

    def init_pre(self, sim):
        super().init_pre(sim)
        npts = len(sim.t)
        self.results["population_age_distribution"] = np.zeros((npts, 101))

    def step(self):
        sim = self.sim
        ti  = self.ti
        ppl = sim.people

        # Make sure SDoH is a *boolean array*
        if hasattr(ppl, self.sdoh_attr):
            sdoh = np.asarray(getattr(ppl, self.sdoh_attr), dtype=bool)
        else:
            sdoh = np.zeros(len(ppl), dtype=bool)

        no_sdoh = ~sdoh
        alive   = np.asarray(ppl.alive, dtype=bool)
        male    = np.asarray(ppl.male,  dtype=bool)
        female  = np.asarray(ppl.female,dtype=bool)

        for disease in self.diseases:
            dis = getattr(sim.diseases, disease.lower())
            status_attr = "infected" if disease.lower() in ["hiv","hpv","flu","viralhepatitis","tb"] else "affected"
            has_disease = alive & np.asarray(getattr(dis, status_attr), dtype=bool)

            has_disease_f = has_disease & female
            has_disease_m = has_disease & male
            sdoh_f        = sdoh & female
            sdoh_m        = sdoh & male
            nosdoh_f      = no_sdoh & female
            nosdoh_m      = no_sdoh & male

            total_num_with = 0
            total_den_with = 0

            for i,(a0,a1) in enumerate(self.age_bins):
                age_group = (ppl.age >= a0) & (ppl.age < a1)

                num_m   = int(np.sum(age_group & has_disease_m))
                den_m   = int(np.sum(age_group & male))
                num_f   = int(np.sum(age_group & has_disease_f))
                den_f   = int(np.sum(age_group & female))

                num_w_m = int(np.sum(age_group & has_disease_m & sdoh))
                den_w_m = int(np.sum(age_group & male & sdoh))
                num_w_f = int(np.sum(age_group & has_disease_f & sdoh))
                den_w_f = int(np.sum(age_group & female & sdoh))

                num_wo_m = int(np.sum(age_group & has_disease_m & no_sdoh))
                den_wo_m = int(np.sum(age_group & male & no_sdoh))
                num_wo_f = int(np.sum(age_group & has_disease_f & no_sdoh))
                den_wo_f = int(np.sum(age_group & female & no_sdoh))

                total_num_with += (num_w_m + num_w_f)
                total_den_with += (den_w_m + den_w_f)

                for sex, num, den, num_w, den_w, num_wo, den_wo in zip(
                    ("male","female"),
                    (num_m, num_f),
                    (den_m, den_f),
                    (num_w_m, num_w_f),
                    (den_w_m, den_w_f),
                    (num_wo_m, num_wo_f),
                    (den_wo_m, den_wo_f),
                ):
                    self.results[f"{disease}_num_{sex}_{i}"][ti]              = num
                    self.results[f"{disease}_den_{sex}_{i}"][ti]              = max(den,1)
                    self.results[f"{disease}_prev_{sex}_{i}"][ti]             = sc.safedivide(num, den)
                    self.results[f"{disease}_num_with_SDoH_{sex}_{i}"][ti]    = num_w
                    self.results[f"{disease}_den_with_SDoH_{sex}_{i}"][ti]    = max(den_w,1)
                    self.results[f"{disease}_prev_with_SDoH_{sex}_{i}"][ti]   = sc.safedivide(num_w, den_w)
                    self.results[f"{disease}_num_without_SDoH_{sex}_{i}"][ti] = num_wo
                    self.results[f"{disease}_den_without_SDoH_{sex}_{i}"][ti] = max(den_wo,1)
                    self.results[f"{disease}_prev_without_SDoH_{sex}_{i}"][ti]= sc.safedivide(num_wo, den_wo)

            # Summary (now using boolean-array compatible function)
            self.results[f"{disease}_prev_no_sdoh"][ti]   = self.cond_prob_bool(has_disease, no_sdoh)
            self.results[f"{disease}_prev_has_sdoh"][ti]  = self.cond_prob_bool(has_disease, sdoh)
            self.results[f"{disease}_prev_no_sdoh_f"][ti] = self.cond_prob_bool(has_disease_f, nosdoh_f)
            self.results[f"{disease}_prev_has_sdoh_f"][ti]= self.cond_prob_bool(has_disease_f, sdoh_f)
            self.results[f"{disease}_prev_no_sdoh_m"][ti] = self.cond_prob_bool(has_disease_m, nosdoh_m)
            self.results[f"{disease}_prev_has_sdoh_m"][ti]= self.cond_prob_bool(has_disease_m, sdoh_m)

            self.results[f"{disease}_num_total"][ti] = total_num_with
            self.results[f"{disease}_den_total"][ti] = total_den_with

        # Overall adult SDoH proportion (Fig 5A)
        adults = np.asarray(ppl.age >= 15, dtype=bool) & alive
        self.results["sdoh_prop_adults"][ti] = sc.safedivide(np.sum(sdoh & adults), np.sum(adults))
        