import starsim as ss
import numpy as np
import sciris as sc

class PrevalenceAnalyzer(ss.Analyzer):
    @staticmethod
    def cond_prob(numerator, denominator):
        numer = len((denominator & numerator).uids)
        denom = len(denominator.uids)
        out = sc.safedivide(numer, denom)
        # print(f"Conditional Probability: Numerator={numer}, Denominator={denom}, Result={out}")
        return out

    def __init__(self, prevalence_data, diseases=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'prevalence_analyzer'
        self.prevalence_data = prevalence_data
        self.diseases = diseases

        # Define age bins
        self.age_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45),
                         (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, float('inf'))]

        self.results_defined = False
        # print(f"Initialized PrevalenceAnalyzer with diseases: {self.diseases}")
        return

    def init_results(self):
        if self.results_defined:
            return
        results = sc.autolist()
        for disease in self.diseases:
            for i, (age_start, age_end) in enumerate(self.age_bins):
                results += [
                    ss.Result(f'{disease}_num_male_{i}', dtype=int),
                    ss.Result(f'{disease}_den_male_{i}', dtype=int),
                    ss.Result(f'{disease}_num_female_{i}', dtype=int),
                    ss.Result(f'{disease}_den_female_{i}', dtype=int),
                    ss.Result(f'{disease}_num_with_HIV_male_{i}', dtype=int),
                    ss.Result(f'{disease}_den_with_HIV_male_{i}', dtype=int),
                    ss.Result(f'{disease}_num_with_HIV_female_{i}', dtype=int),
                    ss.Result(f'{disease}_den_with_HIV_female_{i}', dtype=int),
                    ss.Result(f'{disease}_num_without_HIV_male_{i}', dtype=int),
                    ss.Result(f'{disease}_den_without_HIV_male_{i}', dtype=int),
                    ss.Result(f'{disease}_num_without_HIV_female_{i}', dtype=int),
                    ss.Result(f'{disease}_den_without_HIV_female_{i}', dtype=int),
                ]
            results += [
                ss.Result(f'{disease}_prev_no_hiv', dtype=float, scale=False),
                ss.Result(f'{disease}_prev_has_hiv', dtype=float, scale=False),
                ss.Result(f'{disease}_prev_no_hiv_f', dtype=float, scale=False),
                ss.Result(f'{disease}_prev_has_hiv_f', dtype=float, scale=False),
                ss.Result(f'{disease}_prev_no_hiv_m', dtype=float, scale=False),
                ss.Result(f'{disease}_prev_has_hiv_m', dtype=float, scale=False),
                ss.Result(f'{disease}_num_total', dtype=int),  # Total numerator without sex
                ss.Result(f'{disease}_den_total', dtype=int),  # Total denominator without sex
            ]
        self.define_results(*results)
        self.results_defined = True
        print(f"Results initialized for diseases: {self.diseases}")
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        npts = len(sim.t)  # Number of time points in the simulation

        # Initialize array to store population age distribution for each year (single-age resolution)
        self.results['population_age_distribution'] = np.zeros((npts, 101))  # 0 to 100 years (single-year resolution)
        print(f"Initialized population age distribution array with shape: {self.results['population_age_distribution'].shape}")
        return

    def step(self):
        sim = self.sim
        ti = self.ti
        ppl = sim.people
        hiv = sim.diseases.hiv

        denom = (ppl.age >= 0)  # All individuals
        has_hiv = denom & hiv.infected  # Individuals with HIV
        no_hiv = denom & hiv.susceptible  # Individuals without HIV

        # print(f"Step {ti}: Processing diseases for time step {ti}")

        for disease in self.diseases:
            dis = getattr(sim.diseases, disease.lower())
            status_attr = 'infected' if disease in ['HIV', 'HPV', 'Flu'] else 'affected'
            has_disease = denom & getattr(dis, status_attr)

            has_disease_f = has_disease & ppl.female  # Women with disease
            has_disease_m = has_disease & ppl.male  # Men with disease
            has_hiv_f = has_hiv & ppl.female  # Women with HIV
            has_hiv_m = has_hiv & ppl.male  # Men with HIV
            no_hiv_f = no_hiv & ppl.female  # Women without HIV
            no_hiv_m = no_hiv & ppl.male  # Men without HIV

            total_num_with_HIV = 0
            total_den_with_HIV = 0

            for i, (age_start, age_end) in enumerate(self.age_bins):
                age_group = (ppl.age >= age_start) & (ppl.age < age_end)
                num_male = np.sum(age_group & has_disease_m)
                den_male = np.sum(age_group & ppl.male)
                num_female = np.sum(age_group & has_disease_f)
                den_female = np.sum(age_group & ppl.female)
                num_with_HIV_male = np.sum(age_group & has_disease_m & has_hiv)
                den_with_HIV_male = np.sum(age_group & has_hiv & ppl.male)
                num_with_HIV_female = np.sum(age_group & has_disease_f & has_hiv)
                den_with_HIV_female = np.sum(age_group & has_hiv & ppl.female)
                num_without_HIV_male = np.sum(age_group & has_disease_m & no_hiv)
                den_without_HIV_male = np.sum(age_group & no_hiv & ppl.male)
                num_without_HIV_female = np.sum(age_group & has_disease_f & no_hiv)
                den_without_HIV_female = np.sum(age_group & no_hiv & ppl.female)

                total_num_with_HIV += num_with_HIV_male + num_with_HIV_female
                total_den_with_HIV += den_with_HIV_male + den_with_HIV_female

                # print(f"Age group {age_start}-{age_end}: num_male={num_male}, den_male={den_male}, num_female={num_female}, den_female={den_female}")

                self.results[f'{disease}_num_male_{i}'][ti] = num_male
                self.results[f'{disease}_den_male_{i}'][ti] = den_male
                self.results[f'{disease}_num_female_{i}'][ti] = num_female
                self.results[f'{disease}_den_female_{i}'][ti] = den_female
                self.results[f'{disease}_num_with_HIV_male_{i}'][ti] = num_with_HIV_male
                self.results[f'{disease}_den_with_HIV_male_{i}'][ti] = den_with_HIV_male
                self.results[f'{disease}_num_with_HIV_female_{i}'][ti] = num_with_HIV_female
                self.results[f'{disease}_den_with_HIV_female_{i}'][ti] = den_with_HIV_female
                self.results[f'{disease}_num_without_HIV_male_{i}'][ti] = num_without_HIV_male
                self.results[f'{disease}_den_without_HIV_male_{i}'][ti] = den_without_HIV_male
                self.results[f'{disease}_num_without_HIV_female_{i}'][ti] = num_without_HIV_female
                self.results[f'{disease}_den_without_HIV_female_{i}'][ti] = den_without_HIV_female

            self.results[f'{disease}_prev_no_hiv'][ti] = self.cond_prob(has_disease, no_hiv)
            self.results[f'{disease}_prev_has_hiv'][ti] = self.cond_prob(has_disease, has_hiv)
            self.results[f'{disease}_prev_no_hiv_f'][ti] = self.cond_prob(has_disease_f, no_hiv_f)
            self.results[f'{disease}_prev_has_hiv_f'][ti] = self.cond_prob(has_disease_f, has_hiv_f)
            self.results[f'{disease}_prev_no_hiv_m'][ti] = self.cond_prob(has_disease_m, no_hiv_m)
            self.results[f'{disease}_prev_has_hiv_m'][ti] = self.cond_prob(has_disease_m, has_hiv_m)
            self.results[f'{disease}_num_total'][ti] = total_num_with_HIV
            self.results[f'{disease}_den_total'][ti] = total_den_with_HIV

            # print(f"Total with HIV: num={total_num_with_HIV}, den={total_den_with_HIV}")

        return