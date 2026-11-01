# MIGHTI on Starsim v4: port notes (October 26)

The port of MIGHTI (`v4` branch, `6835ca6`) to Starsim v4 (the `ssv4` working tree) and STIsim 1.7.1 (ported to v4 the same day), made in place and left uncommitted. The v4-side findings are in `ssv4/v4_design/oct26/port_proposals.md`, section MIGHTI (MG1–MG12).

## What was done

The health conditions are now declared diseases, the interactions between them are `ss.modify()`, deaths have one cause each (v4's `request_death(cause=)`), the calibrations use `sim.fit()`, and the copies, shims, and debugging scaffolding that v4 makes unnecessary are gone. Package code (`mighti/`): 1,594 lines added (with the new `mighti/diseases/conditions.py`), 9,381 removed, net −7,787 (14,163 → 6,397 lines). Tests: 233 added, 652 removed with `mighti_main.py`, the docs, and the requirements.

- **Migration script** (`python -m starsim.interop.migrate mighti/ tests/ docs/ mighti_main.py data_prep/`): 34 changes in 19 files (`copy_inputs=` dropped, `ss.NCD` → `ssl.NCD`, `ss.SIS`/`ss.SIR` → `ssl.AgentSIS`/`ssl.AgentSIR`, `ss.standardize_data`, `preg.pregnant`), and 64 items to do by hand (analyzers' first step, `n_` names, the beta pairs on STIsim's network, `ss.Tx`, `ss.Calibration`, `find_children`). Most of what it changed was then rewritten.
- **Conditions declared** (`mighti/diseases/base_disease.py`, 1,581 → 330 lines; the 50 one-class files → `conditions.py`, 191 lines). Each base class is a few lines of states and transitions: `onset = susceptible >> affected` at `p_acquire_multiplier*ss.by(sex=dict(female=p_acquire_female, male=p_acquire_male))` (annual probabilities, `ss.probperyear`, so Starsim converts them to the step: MIGHTI's own conversion, `_apply_dt_parameter_conversion()` and `convert_condition_parameters_to_dt()`, is gone), `remission`, and `death = ss.death(affected, rate=p_death)`. `ChronicDisease`, `AcuteDisease` (the same course in MIGHTI), `RemittingDisease`, `GenericSIS`, `GenericSIR`, and `AcuteSurgicalDisease` are declared; `NonAcquiredDisease` and `StaticCondition` (present from birth) stay written as code, with a callback on `ss.Pregnancy`'s birth. The parameters still come from the CSV file, by class name; `pars=dict(init_prev=...)` with MIGHTI's age-and-sex prevalence functions works as before; results are v4's (`affected`, `susceptible`, the flows `onset`, `remission`, `death`, and `prevalence`). `duration` (years since onset, for YLDs) reads the time of entering `affected` (`ti_affected`, kept by `ss.State(keep_time=True)`). `MajorDepressiveDisorder`'s faster remission for people who are housed is in its rate: `remission_mult*remission_rate*(1 + housing_protect*ss.ppl.get('neighbourhood_situation', 0))`.
- **Interactions** (`mighti/interactions/connectors.py`, 347 → 90 lines): `create_dynamic_connector(c1, c2, rr)` is `ss.modify('c2.rel_sus', multiply=rr, where=ss.state('c1.affected'))`, and `NCDHIVConnector(rel_sus)` is a list of them, one per condition, `where=hiv.infected`. The per-step reset of `rel_sus` to a stored baseline (MIGHTI's workaround so effects stop after remission and stack) is what modifiers do by construction; it also reset HIV's own `rel_sus`, wiping STIsim's PrEP and circumcision effects for anyone in an interaction (a MIGHTI bug, now gone). Both helpers take `diseases=` to keep the conditions in the sim (a modifier of a missing module is an error in v4; MIGHTI's connectors skipped them silently).
- **Mortality** (`mighti/demography/`, 805 → 160 lines): MIGHTI's "death pressure" protocol (each disease reporting per-agent probabilities to a central allocator, which drew deaths and stored a cause map on the sim) is replaced by v4's own: each module kills at its own rate with itself as the cause, and `people.cause_of_death` has one cause per death. `AdditiveHazardDeaths` is an `ss.Deaths` with a background multiplier from an age (25 lines); `CompetingRisksDeaths` is an `ss.Deaths` that kills at the all-cause hazard less the modeled causes' hazards (from each condition's `death_hazard()`, and STIsim HIV's), so total mortality stays at the observed all-cause rate. `HIVCompeting` (a 177-line copy of STIsim 1.4's `HIV.step_state()`) is gone.
- **Analyzers**: deaths by age and sex, m(x), cause of death, and condition at death read `people.dying()` and `people.cause_of_death` instead of diffing snapshots of who is dead (`analyzer_core.py`, 573 → 290 lines); `CauseDeathRateAnalyzer` reads `people.died_of(disease)`. `PrevalenceAnalyzer_SDoH` had no `step()` (its results were never filled); it has one now. `AdherenceAnalyzer` used Starsim 2's `initialize()`/`apply()` and is ported.
- **Interventions**: `T2D_ReduceMortalityTx` and `DepressionCare` (subclasses of `ss.treat_num`, removed in v4) are gone; the test now uses `ss.treat(ss.Treatment(efficacy=dict(death=0.5)), where=t2d.affected)`. `ART` (an empty subclass of `sti.ART`), `ARTNoAutoAdjust` (a 500-line copy of STIsim 1.4's `ART.step()`, unused), `ImproveHospitalDischarge` (set a parameter no disease has), `AdherenceFromDepression` ("legacy"), and the STIsim version-sniffing helpers are gone. The three housing interventions share a base class (`HousingSupport`, `start=` instead of `start_day=`/`start_year=`). The adherence modules (805 → 190 lines) keep their science (CASM and SDoH multipliers, the HRMM odds model, dropout by adherence and AUD, ART efficacy scaled by mean adherence) without the debugging, unit guessing (`ti_art` "in years, months, or timesteps"), and workarounds for STIsim 1.4's `post_art_decline` crashes.
- **SDoH** (`mighti/sdoh/core.py`): inheritance from the mother at birth is a callback on `ss.Pregnancy`'s birth (was `birth_mother_baby_pairs()`, which read `ti_delivery` and `find_children()`), draws use Starsim's distributions (common random numbers) instead of MIGHTI's own seeded NumPy generators (`mighti/util/rng.py`, gone), and the share with the property is a result (was a dict stuck on the sim).
- **Calibration**: `calibration_diseases.py` (the logic of `calibration_diseases_ver2.py`, the version the frozen results came from, with its joint fit of `p_death`) and `calibration_hiv.py` use `sim.fit(gof=..., method='optuna')`, with the parameters given bounds as `ss.par(ss.probperyear(0.001), bounds=[1e-6, 0.1])`; the build functions, `ss.Calibration`/`sti.Calibration` wiring, result-key aliasing for STIsim's `make_df()`, and the 1,221-line copy of the disease classes with `p_acquire=1` (`diseases_for_calibration.py`) are gone. The Optuna diagnostics are kept (from `fit.study`). Both ran end to end (smoke runs).
- **Package**: the `mi.<Anything>` attribute shim in `mighti/__init__.py` is gone (use the namespaces, as the README says); `mighti/util/utils.py` (unit conversions and `birth_mother_baby_pairs`) and `mighti/util/rng.py` are gone; `people_extend.py` lost an unused `People` subclass and a `step_die` replacement.
- **STIsim API** (1.4 → 1.7.1, not only the v4 port): `art_efficacy` is `effective_art_efficacy` (and `nonsupp_art_efficacy`), `include_care` is gone, and `p_hiv_death` is set by STIsim each step, so MIGHTI's `hiv.pars.art_efficacy = 0.9`, `include_care = ...`, and `p_hiv_death = ...` in `mighti_main.py` and the tests did nothing on STIsim ≥ 1.5, silently (MG7). They are removed, or renamed to `effective_art_efficacy` (the adherence scaling, `InterventionAdherenceDisruptor`).
- **Entry points**: `mighti_main.py` runs (100,000 agents, monthly, 2007–2024: 30 s; reads the analyzers from the sim, `dt=ss.years(dt_years)`, its observed-HIV path no longer depends on the working directory, and its plot of prevalence by HIV status is for the conditions in the run rather than the hard-coded AcuteHepatitis, which crashed on Starsim 3 too). `data_prep/` uses no Starsim and is unchanged.
- **Requirements**: `starsim>=4.0.0` and `stisim>=1.7.1` in `pyproject.toml`, `requirements.txt`, and `docs/requirements.txt` (`setup.py` has none); Python ≥ 3.11.

`v4port/` does not exist for MIGHTI (a new port); `manuscript/` was not touched.

## Decisions for Cliff

1. **Conditions declared, not kept as code.** A condition written as code (as v3's) cannot take `ss.modify()` of its `rel_sus` (v4 refuses: "nothing reads its rel_sus"), so the interactions would have stayed hand-rolled; declared, they are 5–10 lines each and the interactions, treatments, causes of death, and results come from v4 (MG1). The CSV interface, class names, `affected`/`infected` states, `pars=dict(init_prev=...)`, and `prevalence` result are unchanged.
2. **Dropped parameters**: `dur_condition` (a SciPy lognormal nothing read), `max_disease_duration` (never read), `rel_sus_hiv` (read through `sim.people.hiv[uids]`, which always failed silently, so the HIV effect came only from `NCDHIVConnector`), `p_acquire` (now only the default of `p_acquire_female`/`p_acquire_male`), and the sex-specific acquisition multipliers (the calibration fits the sex-specific probabilities directly). `remission_prevalence` is gone (the `reversed` state was emptied in the step it was filled, so it was always 0).
3. **Removed classes** (unused by tests, `mighti_main.py`, docs, or `data_prep/`; listed above): `ARTNoAutoAdjust`, `ART`, `ImproveHospitalDischarge`, `AdherenceFromDepression`, `DepressionCare`, `T2D_ReduceMortalityTx`, `HIVCompeting`, `PeopleCustom`, `fixed_step_die`. `DepressionCare` in v4 would be `care = ss.treat(..., where=mdd.affected)` plus `ss.modify(mdd.remission, multiply=1.5, where=care.ever)`.
4. **Calibration files**: `calibration_diseases.py` now holds `calibration_diseases_ver2.py`'s science (a superset: prevalence only when there are no death data), and `ver2` and `diseases_for_calibration.py` are deleted; the frozen results in `mighti/calibration/results/` are untouched.
5. **Annual rates where MIGHTI used per-step probabilities**: the SDoH loss and gain (0.01 and 0.10 "yearly" in the comments, per step in the code) and ART dropout (per month: MIGHTI's comments assume monthly steps) are now annual and monthly probabilities, so results no longer change with `dt`. At MIGHTI's default monthly step, ART dropout is unchanged; SDoH loss and gain are 12 times slower (the share housed at equilibrium is the same, 0.909).
6. **`CompetingRisksDeaths` no longer caps a modeled cause at the all-cause rate** (MIGHTI's capped AIDS deaths at the all-cause mortality of the person's age, which made a CD4 count of zero rarely fatal); the residual is 0 there. Its modeled hazards are each condition's death transitions' current hazards, with their modifiers (e.g. a treatment): `dis.death.hazard(uids)`.
7. **`NCDHIVConnector` returns a list of modifiers** (was a connector object); `connectors=mi.interactions.NCDHIVConnector(rel_sus) + [...]`.
8. **MIGHTI's own bugs not fixed** (behavior kept): the SDoH CSV is matched on `condition_name` ('NeighbourhoodSituation') while the file's rows are 'neighbourhood' etc., so the file's values are never used (98% housed, not 70%); `HRHAnalyzer` and `economics/budget_constraint.py` use Starsim 2's API and an import that cannot work (`from economics.utils import ...`), and nothing uses them; `tests/test_neonatal_disease` has no `.py` extension (never collected) and reads files that do not exist.

## MIGHTI's own bugs fixed or found

- `tests/test_adherence.py` set `hiv.pars.init_art`, which STIsim's HIV doesn't have, so "everyone HIV-positive starts ART at t=0" never happened; v4 now raises for a parameter set but never defined (MG7), so the line is removed (the test passes without it, as it did before).
- Congenital and neonatal conditions were never given to newborns on Starsim 3.1+ (`birth_mother_baby_pairs()` looked for mothers with `ti_delivery == sim.ti` after `ss.Pregnancy` had cleared it), so their prevalence decayed as the population turned over (Down syndrome 1% → 0.49% by 2020, against 0.74% now, where the newborns have it at 1%).
- `MajorDepressiveDisorder`'s remission was applied twice a step (in `step_state()`, and again from its `step()`). No effect with the Eswatini data, whose MDD `remission_rate` is blank (0).
- `T2D_ReduceMortalityTx` halved `rel_death` every step each person stayed eligible (so mortality went to 0 within a few steps), rather than once.
- The interaction connectors reset HIV's `rel_sus` to its value at the start every step, erasing STIsim's PrEP and circumcision effects for people with an interacting condition.
- `PrevalenceAnalyzer_SDoH` never recorded anything (no `step()`); `AdherenceAnalyzer` never ran (Starsim 2's `apply()`).
- `mighti_main.py` crashed at its last plot (AcuteHepatitis is not in the default run) and read the observed HIV file relative to the working directory.
- The STIsim parameters MIGHTI set (`art_efficacy`, `include_care`, `p_hiv_death`) had no effect on STIsim ≥ 1.5, which MIGHTI's `pyproject.toml` already required.

## For STIsim

- `sti.HIV` requests its deaths with no cause (`request_death(uids)`, `hiv.py:559` and `564`), so in v4 they have no cause: MIGHTI's cause-of-death analyzers report them as 'unknown'. Passing `cause=self` would attribute them.
- The renames above (`art_efficacy`, `include_care`) are in STIsim's changelog for 1.5, not in its port notes; worth a line in the v4 port notes' table, since setting a removed parameter is silent (MG7).

## Validation against Starsim 3

Starsim 3.7.2 with STIsim 1.7.1 (`git archive`; MIGHTI's own code needs Starsim ≥ 3.1 for `find_children()`, so its pinned 3.0.3 cannot run `mighti_main.py`), against v4 with this port; 12 seeds each, 5,000 agents, Eswatini data, 2007–2020, monthly steps; mean ± SD; z is the difference over its standard error. Scripts: `val/validate.py` and `val/summarize.py` in the session scratchpad (not kept). Two scenarios: six conditions acquired over life (Type 2 diabetes, AUD, hypertension, TB as a `GenericSIR`, CVD, CKD) with their interactions and Down syndrome, with pregnancy and background deaths; and HIV with Type 2 diabetes, AUD, MDD, Down syndrome, `NCDHIVConnector`, housing, HIV testing, and ART.

| Measure | Starsim 3 | v4 | Difference |
|---|---|---|---|
| T2D prevalence 2020 | 0.378 ± 0.010 | 0.380 ± 0.009 | +0.4% (z=0.4) |
| T2D deaths | 133 ± 12 | 128 ± 12 | −3.7% (z=−1.0) |
| AUD prevalence 2020 | 0.0094 ± 0.0016 | 0.0092 ± 0.0022 | −2.2% (z=−0.3) |
| Hypertension prevalence 2020 | 0.290 ± 0.008 | 0.287 ± 0.011 | −1.0% (z=−0.8) |
| CVD prevalence 2020 | 0.061 ± 0.003 | 0.064 ± 0.002 | +5.2% (z=2.9) |
| CKD prevalence 2020 | 0.073 ± 0.004 | 0.073 ± 0.005 | −0.4% (z=−0.2) |
| TB prevalence 2020 | 0.387 ± 0.011 | 0.378 ± 0.007 | −2.3% (z=−2.3) |
| TB deaths | 3,471 ± 43 | 3,415 ± 39 | −1.6% (z=−3.4) |
| Down syndrome prevalence 2020 | 0.0049 ± 0.0016 | 0.0074 ± 0.0017 | +52% (z=3.7) |
| Alive 2020 | 2,087 ± 45 | 2,063 ± 45 | −1.2% (z=−1.3) |
| Births | 1,222 ± 45 | 1,203 ± 45 | −1.5% (z=−1.0) |
| Background deaths | 536 ± 20 | 547 ± 18 | +2.1% (z=1.5) |
| With HIV: HIV prevalence 2020 | 0.209 ± 0.009 | 0.202 ± 0.007 | −3.3% (z=−2.0) |
| With HIV: on ART 2020 | 996 ± 46 | 959 ± 30 | −3.8% (z=−2.3) |
| With HIV: T2D prevalence 2020 | 0.413 ± 0.007 | 0.412 ± 0.006 | −0.3% (z=−0.5) |
| With HIV: MDD prevalence 2020 | 0.043 ± 0.004 | 0.041 ± 0.002 | −3.1% (z=−1.0) |
| With HIV: share housed 2020 | 0.899 ± 0.004 | 0.901 ± 0.005 | +0.3% (z=1.3) |
| With HIV: alive 2020 | 5,863 ± 69 | 5,714 ± 70 | −2.5% (z=−5.2) |

- **Within noise**: every acquired condition but TB and CVD at its end point (CVD's prevalence midway agrees: −0.9%, z −0.3), births, background deaths, AUD and MDD with HIV, and the share housed.
- **Down syndrome +52%**: MIGHTI's bug on Starsim 3 (above); v4 is right.
- **TB −2%**: TB's rates are high (annual probabilities of 0.096 of acquiring and 0.196 of dying), and on Starsim 3 someone could acquire TB and die of it in the same step; v4 takes one transition per person per step. An error of order `dt`.
- **Alive −2.5% with HIV, HIV −3%**: STIsim's port found the same (its ST2 and its fetuses counted as alive on Starsim 3); not MIGHTI's.
- Against MIGHTI's own versions (Starsim 3.2.2 with STIsim 1.4.0), the NCD scenario agrees as above, and HIV prevalence is −15% (0.237 against 0.202), from STIsim's changes between 1.4 and 1.7 (CD4-stratified mortality, suppressive and non-suppressive ART), not from Starsim.

## Timing

Each run is the validation's scenario at 20,000 agents, monthly, 2007–2020; 3 seeds after a discarded run that compiles the Numba kernels; run one at a time (load under 1). Build is `sim.init()`; run is `sim.run()` after it.

| Scenario | Starsim 3.2.2, STIsim 1.4.0 | Starsim 3.7.2, STIsim 1.7.1 | v4 | v4 against 3.7.2 |
|---|---|---|---|---|
| HIV with 4 conditions: build | 0.32 s | 0.16 s | 0.14 s | 1.2× |
| HIV with 4 conditions: run | 9.02 s | 5.75 s | 3.95 s | 1.46× faster |
| 7 conditions, no HIV: build | | 0.11 s | 0.15 s | 0.7× |
| 7 conditions, no HIV: run | | 3.71 s | 2.07 s | 1.8× faster |

Against MIGHTI's own versions (Starsim 3.2.2, STIsim 1.4.0), the HIV scenario runs 2.3 times faster. The declared conditions take longer to build (one engine each) and run faster than the hand-written per-agent loops they replace.

## Tests

Starsim 3 (MIGHTI's 3.2.2 with STIsim 1.4.0, and 3.0.3): 33 passed, 1 xpassed (`test_reference_life_expectancy`, marked xfail "under active development"). v4: 35 passed, 0 xfailed, 0 failed (one more test than before: `test_dt_scaling.py` checks the conversion as well as the behavior). Every test file has a v4 counterpart; none were deleted. Changes: `test_intervention.py` uses `ss.treat`; `test_dt_scaling.py` checks that the share acquiring a condition in 5 years is the same at yearly and monthly steps (was: that MIGHTI's own conversion function set the per-step parameters), and that Starsim's conversion of an annual probability is MIGHTI's; `test_mortality_additive.py` uses a disease that kills with itself as the cause (was: the death-pressure protocol), and checks that each death has one cause; the connector tests read `rel_sus.effective` (the modified value) rather than calling a connector's `step()`; `test_reference_life_expectancy.py` is no longer xfail, and checks that the cause of every death is `ss.Deaths`; tests read modules and results from the sim rather than from the objects given (which a v4 sim copies).

## Docs

The docs are Sphinx (`docs/`, API reference by autodoc). Sphinx is not installed in the `v4` env, so they were not built; every module named in `docs/api/*.rst` imports in the `v4` env. Updated: `docs/api/mighti.diseases.rst` (the 50 condition modules → `base_disease` and `conditions`), `docs/api/mighti.rst` (no `mighti.util.utils`; the mortality, SDoH, and adherence modules added), `docs/index.rst` (its toctree named `api/index`, which does not exist; now `api/modules`), `docs/requirements.txt`, `README.rst` (requirements, the interactions example, no attribute shim), and `mighti/calibration/README.md`.

## After Cliff's review (October 26)

The v4 features the MIGHTI proposals asked for are built, and the hacks they stood in for are replaced:

- **MG2** (`mighti/diseases/base_disease.py`): the `onset_time = ss.attr(affected.time)` lines (one per base class, there only so that the time of onset was kept) are gone; the state of having the condition is declared `ss.State(keep_time=True)` (`affected`, or `infected` for `GenericSIS`/`GenericSIR`), and `duration` reads `ti_affected` (or `ti_infected`) directly, without the cast and `nan_to_num()`.
- **MG3** (`mighti/sdoh/core.py`, `mighti/interventions/adherence.py`, `mighti/diseases/conditions.py`): `add_people_state()` (four lines of People internals) is gone. The SDoH modules add their property with `sim.people.add_state(ss.BoolArr(...))` and the adherence module with `sim.people.add_state(ss.FloatArr('adherence', default=1.0))`; `MajorDepressiveDisorder` reads `ss.ppl.get('neighbourhood_situation', 0)` in its remission rate, so it no longer needs an `init_pre()` adding the property for when no SDoH module is in the sim.
- **MG6** (`mighti/calibration/calibration_diseases.py`): the in-memory Optuna storage and `optuna.load_study(study_name='calibrate', ...)` are gone; the diagnostic plots read `fit.study`. A smoke run (`--smoke`, Type 2 diabetes, 5 trials) wrote every plot. Both calibration scripts use `n_trials=`.
- **MG8(b)** (`mighti/diseases/base_disease.py`): `Condition.death_hazard()` is now the sum of the current hazards of the condition's death transitions (`tr.hazard(uids)`, for those in its source state), so `CompetingRisksDeaths` sees modifiers such as a treatment's efficacy on `death`; `AcuteSurgicalDisease`'s own override (relative mortality before and after surgery, computed from the parameters) is gone, since its two death transitions carry it. `hazard()` gives the rate per unit of the sim's time, which is months in a sim with STIsim's HIV, so `death_hazard()` converts it to per year. Without modifiers it is identical to the old values (checked per agent for four conditions, including a surgical one).
- **MG8(a)**, not used: hazard-weighted choice of cause needs a hazard from every claim on a person, and MIGHTI's deaths come from declared transitions (whose engine requests them without `hazard=`), `ss.Deaths`, and STIsim's HIV. Passing `hazard=` from `NonAcquiredDisease` alone would change nothing.

Results: the NCD scenario (with `ss.Deaths` and with `CompetingRisksDeaths`) and the HIV scenario of the validation above, 8 seeds of 5,000 agents each, are identical before and after (every prevalence, death count, births, background deaths, HIV prevalence, on ART, share housed, and the sum of `duration`, which differs at the seventh digit since the kept time is float32). No treatment modifies a condition's deaths in those scenarios, so the change in `CompetingRisksDeaths` does not show there. Tests: 35 passed before and after.
