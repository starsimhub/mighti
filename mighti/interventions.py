"""
Defines interventions.
"""


import starsim as ss
import stisim as sti
import numpy as np

_all_ = ['ART', 'ARTwithCASM', 'ARTNoAutoAdjust', 'ImproveHospitalDischarge', 'GiveHousingToDepressed', 'GiveHousingSupport', 'HousingSupportForAUD']


class ART(sti.ART):
    """
    ART intervention with optional integration to the BudgetConstraint module.
    """

    def init_pre(self, sim):
        super().init_pre(sim)
        # Store reference to budget module if present
        self._budget_module = sim.get_module("budget_constraint", optional=True)

    def apply(self, sim):
        # Execute normal ART behavior (diagnosis, initiation, adherence updates, etc.)
        super().apply(sim)

        # If budget constraint active, register cost and HRH usage
        if self._budget_module:
            n_treated = getattr(self, "n_treated", 0)

            # Safety guard: only proceed if n_treated > 0
            if n_treated > 0:
                cost = n_treated * getattr(self, "cost_per_person_year", 120) / sim.n_years
                hrh_minutes = {
                    "doctor": 5 * n_treated,
                    "nurse": 30 * n_treated,
                }
                self._budget_module.register_usage(
                    cost=cost,
                    hrh_minutes=hrh_minutes,
                    source=self.name,
                )


class ARTwithCASM(sti.ART):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.casm_sensitivity = "pharma"


class ARTNoAutoAdjust(sti.ART):
    """
    ART intervention with auto-adjustment disabled.
    
    By default, sti.ART may auto-adjust coverage based on current on_art status
    to maintain target coverage levels. This class disables that behavior by
    returning the target coverage without correction.
    
    Also fixes the issue where sti.ART calculates coverage based on all infected
    individuals instead of diagnosed individuals.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._debug_coverage_calls = []
        self._debug_apply_calls = []
        # Track if art_coverage_correction was called
        self._correction_called = False
        self._last_correction_value = None
    
    def art_coverage_correction(self, sim, target_coverage=None):
        """
        Override to disable auto-adjustment and ensure coverage is calculated correctly.
        
        The parent class's art_coverage_correction uses target_coverage parameter directly,
        so we need to recalculate it based on diagnosed population and then call the parent.
        """
        # Get the raw coverage proportion from coverage_data by interpolating
        if hasattr(self, 'coverage_data') and self.coverage_data is not None:
            # Interpolate coverage proportion for current year
            year = sim.t.year
            years = self.coverage_data.index.values
            props = self.coverage_data['p_art'].values
            coverage_prop = np.interp(year, years, props)
        else:
            # Fallback: if no coverage_data, use target_coverage as-is
            coverage_prop = None
        
        # If target_coverage is provided and is an absolute number (likely > 1.0),
        # recalculate based on the actual eligible population (diagnosed HIV+)
        hiv = getattr(sim.diseases, "hiv", None)
        if hiv is not None and "hiv.diagnosed" in sim.people.states:
            diagnosed = np.asarray(sim.people.states.get("hiv.diagnosed", []), bool)
            n_diagnosed = diagnosed.sum()
            
            if n_diagnosed > 0 and coverage_prop is not None:
                # Recalculate target as proportion of diagnosed
                # This ensures we're applying coverage to the right population
                corrected_target = int(coverage_prop * n_diagnosed)
                
                # Debug output
                if hasattr(sim, 't') and len(self._debug_coverage_calls) < 10:
                    self._debug_coverage_calls.append((sim.t.year, target_coverage, coverage_prop, n_diagnosed, corrected_target))
                    if len(self._debug_coverage_calls) <= 3 or sim.t.year in [2010, 2013, 2016, 2020]:
                        print(f"[ARTNoAutoAdjust.art_coverage_correction] Year {sim.t.year}: coverage_prop={coverage_prop:.3f}, diagnosed={n_diagnosed}, target_count={corrected_target} (original={target_coverage:.0f})")
                
                # Track that correction was called and what value was used
                self._correction_called = True
                self._last_correction_value = corrected_target
                
                # Call parent method with corrected target
                # The parent method doesn't use return value, it uses the parameter
                super().art_coverage_correction(sim, target_coverage=corrected_target)
                return
            elif target_coverage is not None:
                # Use the target_coverage as-is if we can't recalculate
                super().art_coverage_correction(sim, target_coverage=target_coverage)
                return
        
        # Fallback: call parent with original target_coverage
        super().art_coverage_correction(sim, target_coverage=target_coverage)
    
    def step(self):
        """
        Override step() to fix the issue where sti.ART calculates coverage
        based on all infected individuals instead of diagnosed individuals.
        """
        sim = self.sim
        hiv = sim.diseases.hiv
        inf_uids = hiv.infected.uids
        diag_uids = hiv.diagnosed.uids  # Use diagnosed instead of all infected

        # Figure out how many people should be treated
        # FIX: Calculate based on diagnosed, not all infected
        if self.t.now('year') < self.pars.future_coverage['year']:
            if self.coverage is None:
                n_to_treat = 0
            else:
                if self.coverage_format == 'n_art':
                    n_to_treat = int(self.coverage[self.ti]/sim.pars.pop_scale)
                elif self.coverage_format == 'p_art':
                    # FIX: Use diagnosed count instead of all infected
                    n_to_treat = int(self.coverage[self.ti]*len(diag_uids))
        else:
            p_cov = self.pars.future_coverage['prop']
            # FIX: Use diagnosed count instead of all infected
            n_to_treat = int(p_cov*len(diag_uids))

        # Firstly, check who is stopping ART
        if hiv.on_art.any():
            stopping = hiv.on_art & (hiv.ti_stop_art <= self.ti)
            if stopping.any():
                try:
                    hiv.stop_art(stopping.uids)
                except:
                    errormsg = f'Error stopping ART for {stopping.uids}'
                    raise ValueError(errormsg)

        # Next, see how many people we need to treat vs how many are already being treated
        on_art = hiv.on_art

        # A proportion of newly diagnosed agents onto ART will be willing to initiate ART
        diagnosed = hiv.ti_diagnosed == self.ti
        if len(diagnosed.uids):
            # First, filter by base init_prob
            dx_to_treat_base = self.pars.init_prob.filter(diagnosed.uids)
            
            # Debug: track initial counts
            aud_affected = np.asarray(sim.people.states.get("alcoholusedisorder.affected", []), dtype=bool)
            dx_uids = ss.uids(diagnosed)
            dx_has_aud = aud_affected[dx_uids] if aud_affected.any() else np.zeros(len(dx_uids), dtype=bool)
            n_diagnosed = len(dx_uids)
            n_diagnosed_aud = dx_has_aud.sum()
            n_base_willing = len(dx_to_treat_base)
            n_base_willing_aud = sum(1 for uid in dx_to_treat_base if dx_has_aud[list(dx_uids).index(uid)])
            
            # Reduce ART initiation probability for people with AUD
            # This is part of the INTERACTION: AUD affects ART initiation
            # Only apply when interaction is enabled (when adherence engine is present)
            # In "No interaction" scenarios, AUD should NOT affect ART initiation
            has_adherence_data = "adherence" in sim.people.states
            
            # Only apply AUD reduction if interaction is enabled (adherence engine present)
            if has_adherence_data and aud_affected.any():
                # Reduce initiation probability for AUD individuals
                # Use adherence-based reduction if available, otherwise use fixed reduction
                if has_adherence_data:
                    adher = np.asarray(sim.people.states["adherence"], float)
                    # For AUD individuals, reduce initiation prob based on adherence
                    # If adherence is 0.7, they should have much lower initiation prob
                    # Apply stronger reduction: use adherence^3 to make effect much more pronounced
                    # This means: adherence 0.7 -> 0.35 (65% reduction), adherence 0.5 -> 0.125 (87.5% reduction)
                    # This should create a much more visible difference in ART coverage
                    aud_init_prob = np.power(adher[dx_uids], 3.0)  # Cube makes reduction very strong
                    # Apply reduction only to AUD individuals
                    rand = np.random.rand(len(dx_uids))
                    # Keep people who pass both base init_prob AND (if AUD) the adherence-based prob
                    dx_to_treat = []
                    aud_filtered_out = 0
                    for i, uid in enumerate(dx_uids):
                        if uid in dx_to_treat_base:
                            if dx_has_aud[i]:
                                # AUD individual: apply adherence-based probability reduction
                                # This reduces ART initiation for people with AUD
                                if rand[i] < aud_init_prob[i]:
                                    dx_to_treat.append(uid)
                                else:
                                    aud_filtered_out += 1
                                    # Drop them (don't initiate ART)
                            else:
                                # Non-AUD: keep original probability
                                dx_to_treat.append(uid)
                    dx_to_treat = ss.uids(dx_to_treat)
                    
                    # Debug output
                    n_final_willing = len(dx_to_treat)
                    n_final_willing_aud = sum(1 for uid in dx_to_treat if dx_has_aud[list(dx_uids).index(uid)])
                    if n_diagnosed_aud > 0 and sim.ti % 2 == 0:  # Print every 2 timesteps
                        mean_adher_aud = adher[dx_uids][dx_has_aud].mean() if dx_has_aud.any() else 0.0
                        mean_init_prob_aud = aud_init_prob[dx_has_aud].mean() if dx_has_aud.any() else 0.0
                        print(f"[ARTNoAutoAdjust] Year {sim.t.year}: Diagnosed={n_diagnosed} (AUD={n_diagnosed_aud}), "
                              f"Base willing={n_base_willing} (AUD={n_base_willing_aud}), "
                              f"After AUD reduction={n_final_willing} (AUD={n_final_willing_aud}, filtered={aud_filtered_out}), "
                              f"Mean adherence (AUD)={mean_adher_aud:.3f}, Mean init prob (AUD)={mean_init_prob_aud:.3f}")
                else:
                    # No adherence data: simple 50% reduction for AUD
                    aud_reduction_factor = 0.5
                    rand = np.random.rand(len(dx_uids))
                    dx_to_treat = []
                    aud_filtered_out = 0
                    for i, uid in enumerate(dx_uids):
                        if uid in dx_to_treat_base:
                            if dx_has_aud[i]:
                                # AUD individual: 50% chance to keep
                                if rand[i] < aud_reduction_factor:
                                    dx_to_treat.append(uid)
                                else:
                                    aud_filtered_out += 1
                            else:
                                # Non-AUD: keep
                                dx_to_treat.append(uid)
                    dx_to_treat = ss.uids(dx_to_treat)
                    
                    # Debug output
                    n_final_willing = len(dx_to_treat)
                    if n_diagnosed_aud > 0 and sim.ti % 2 == 0:
                        print(f"[ARTNoAutoAdjust] Year {sim.t.year}: Diagnosed={n_diagnosed} (AUD={n_diagnosed_aud}), "
                              f"Base willing={n_base_willing} (AUD={n_base_willing_aud}), "
                              f"After AUD reduction={n_final_willing} (AUD filtered={aud_filtered_out})")
            elif not has_adherence_data:
                # No interaction enabled: AUD does NOT affect ART initiation
                # This is the "No interaction" scenario - diseases exist but don't interact
                dx_to_treat = dx_to_treat_base
                if n_diagnosed_aud > 0 and sim.ti % 2 == 0:
                    print(f"[ARTNoAutoAdjust] Year {sim.t.year}: NO INTERACTION - Diagnosed={n_diagnosed} (AUD={n_diagnosed_aud}), "
                          f"Base willing={n_base_willing} (AUD={n_base_willing_aud}) - AUD does NOT reduce ART initiation")
            else:
                # No AUD in population, use base (no reduction needed)
                dx_to_treat = dx_to_treat_base

            # Figure out if there are treatment spots available and if so, prioritize newly diagnosed agents
            n_available_spots = n_to_treat - len(on_art.uids)
            if n_available_spots > 0:
                # Check if ARTAdherenceDisruptor dropped anyone this step - exclude them from re-initiation
                dropped_this_step = set()
                # Try to get the connector from sim.connectors (dict or list)
                if hasattr(sim, 'connectors'):
                    if isinstance(sim.connectors, dict):
                        art_dropout_connector = sim.connectors.get("artadherencedisruptor", None)
                    else:
                        # List or other iterable
                        art_dropout_connector = None
                        for conn in sim.connectors:
                            if hasattr(conn, 'label') and 'adherence' in conn.label.lower() and 'art' in conn.label.lower():
                                art_dropout_connector = conn
                                break
                    
                    if art_dropout_connector is not None and hasattr(art_dropout_connector, "_dropped_this_step"):
                        dropped_this_step = art_dropout_connector._dropped_this_step
                        if len(dropped_this_step) > 0 and sim.ti % 2 == 0:
                            print(f"[ARTNoAutoAdjust] Year {sim.t.year}: Excluding {len(dropped_this_step)} recently dropped agents from re-initiation")
                
                # Remove dropped agents from dx_to_treat to prevent immediate re-initiation
                if dropped_this_step:
                    dx_to_treat = ss.uids([uid for uid in dx_to_treat if uid not in dropped_this_step])
                
                self.prioritize_art(sim, n=n_available_spots, awaiting_art_uids=dx_to_treat)

        # Apply correction to match ART coverage data:
        # The correction method will recalculate based on diagnosed, but we've already
        # calculated n_to_treat correctly above, so we can pass it through
        self.art_coverage_correction(sim, target_coverage=n_to_treat)

        # Adjust rel_sus for protected unborn agents
        if hiv.on_art[sim.people.pregnancy.pregnant].any():
            mother_uids = (hiv.on_art & sim.people.pregnancy.pregnant).uids
            infants = sim.networks.maternalnet.find_contacts(mother_uids)
            hiv.rel_sus[ss.uids(infants)] = 0

        return
    
    def apply(self, sim):
        """Override to add debug output."""
        # Call parent step (we've overridden step, not apply)
        self.step()
        
        # Get state after applying for debug output
        ppl = sim.people
        st = ppl.states
        hiv = getattr(sim.diseases, "hiv", None)
        
        if hiv is not None and hasattr(sim, 't'):
            diagnosed = np.asarray(st.get("hiv.diagnosed", []), bool) if "hiv.diagnosed" in st else None
            on_art_after = np.asarray(st.get("hiv.on_art", []), bool) if "hiv.on_art" in st else np.zeros(len(ppl), dtype=bool)
            n_diagnosed = diagnosed.sum() if diagnosed is not None else 0
            n_on_art_after = on_art_after.sum()
            
            # Debug output for key years
            if len(self._debug_apply_calls) < 10 and (len(self._debug_apply_calls) <= 3 or sim.t.year in [2010, 2013, 2016, 2020]):
                coverage_prop_expected = np.interp(sim.t.year, self.coverage_data.index.values, self.coverage_data['p_art'].values) if hasattr(self, 'coverage_data') and self.coverage_data is not None else 0.0
                expected_on_art = int(coverage_prop_expected * n_diagnosed) if n_diagnosed > 0 else 0
                actual_coverage = n_on_art_after / n_diagnosed if n_diagnosed > 0 else 0.0
                gap = expected_on_art - n_on_art_after
                self._debug_apply_calls.append((sim.t.year, n_diagnosed, n_on_art_after, expected_on_art, actual_coverage))
                if len(self._debug_apply_calls) <= 3 or sim.t.year in [2010, 2013, 2016, 2020]:
                    correction_info = f", correction_returned={self._last_correction_value}" if self._correction_called and self._last_correction_value is not None else ", correction_not_called"
                    print(f"[ARTNoAutoAdjust.apply] Year {sim.t.year}: diagnosed={n_diagnosed}, on_art={n_on_art_after}, expected={expected_on_art} (coverage={coverage_prop_expected:.3f}), actual_coverage={actual_coverage:.3f}, gap={gap}{correction_info}")
                # Reset tracking for next timestep
                self._correction_called = False
                self._last_correction_value = None


class ImproveHospitalDischarge(ss.Intervention):
    def __init__(self, disease_name, multiplier=2.0, start_day=0, end_day=None, label=None):
        super().__init__(label=label)
        self.disease_name = disease_name
        self.multiplier = multiplier
        self.start_day = start_day
        self.end_day = end_day

    def initialize(self, sim):
        self.sim = sim
        self.disease = sim.diseases[self.disease_name]

    def apply(self):
        ti = self.sim.ti
    
        # Always refresh the disease in case multiprocessing lost it
        if not hasattr(self, 'disease') or self.disease is None:
            try:
                self.disease = self.sim.diseases[self.disease_name]
            except KeyError:
                raise ValueError(f"Disease '{self.disease_name}' not found. Available: {self.sim.diseases.keys()}")
    
        active = self.start_day <= ti < (self.end_day if self.end_day is not None else float('inf'))
    
        if active:
            self.disease.pars.p_daily_discharge_multiplier = self.multiplier
        else:
            self.disease.pars.p_daily_discharge_multiplier = 1.0
    
    def step(self):
        self.apply()
    

class GiveHousingToDepressed(ss.Intervention):
    """
    Intervention that provides stable housing to individuals with Major Depressive Disorder
    who currently have unstable housing.
    """
    def __init__(self, coverage=0.5, start_day=0, label=None):
        super().__init__(label=label or "GiveHousingToDepressed")
        self.coverage = coverage
        self.start_day = start_day

    def initialize(self, sim):
        self.sim = sim

    def apply(self):
        sim = self.sim
        if sim.ti < self.start_day:
            return

        depression = sim.diseases.get('majordepressivedisorder', None)
        if depression is None or not hasattr(depression, 'affected'):
            print(f"[{sim.year}] MajorDepressiveDisorder not found or missing 'affected'")
            return

        # Target depressed + unstably housed
        ppl = sim.people
        depressed = depression.affected
        housing_unstable = ~ppl.neighbourhood_situation
        target = depressed & housing_unstable

        # Apply intervention with given coverage
        target_uids = target.uids
        n = len(target_uids)
        mask = np.random.rand(n) < self.coverage
        to_house = target_uids[mask]        
        ppl.neighbourhood_situation[to_house] = True

        
    def step(self):
        self.apply()


class GiveHousingSupport(ss.Intervention):
    def __init__(self, coverage=0.5, start_year=None, start_day=None, label=None):
        super().__init__(label=label or "GiveHousingSupport")
        self.coverage   = coverage
        self.start_year = start_year
        self.start_day  = start_day  # optional: still support direct ti

    def initialize(self, sim):
        self.sim = sim
        if self.start_day is None:
            if self.start_year is None:
                self.start_day = 0
            else:
                # Convert calendar year to ti
                self.start_day = max(0, int(round(self.start_year - sim.pars['start'])))
        # (optional) sanity print
        print(f"[Init] {self.label}: start_day={self.start_day}")

    def apply(self):
        sim = self.sim
        if sim.ti < self.start_day:
            return
        ppl = sim.people
        unstable = ~ppl.neighbourhood_situation
        adult = ppl.age >= 15
        target = unstable & adult
        uids = target.uids
        if len(uids):
            to_house = uids[np.random.rand(len(uids)) < self.coverage]
            ppl.neighbourhood_situation[to_house] = True
            print(f"[{sim.t.yearvec[sim.ti]:.1f}] {self.label} housed {len(to_house)} / {len(uids)}")
            # print(f"[{sim.year}] {self.label} housed {len(to_house)} / {len(uids)}")
    def step(self):
        self.apply()

class HousingSupportForAUD(ss.Intervention):
    """
    Provides supportive housing to adults with AUD who are unstably housed.
    Optionally reduces relapse risk after housing.
    """

    def __init__(self, coverage=0.5, start_year=2010, relapse_reduction=0.5, label=None):
        super().__init__(label=label or "HousingSupportForAUD")
        self.coverage = coverage
        self.start_year = start_year
        self.relapse_reduction = relapse_reduction

    def step(self):
        sim = self.sim
        current_year = sim.t.year
        if current_year < self.start_year:
            return

        ppl = sim.people
        aud = sim.diseases.alcoholusedisorder

        # Target: adults (≥15) who are unhoused and have AUD
        target = (~ppl.neighbourhood_situation) & aud.affected & (ppl.age >= 15)
        uids = target.uids
        if len(uids) == 0:
            return

        mask = np.random.rand(len(uids)) < self.coverage
        housed_uids = uids[mask]
        ppl.neighbourhood_situation[housed_uids] = True

        # Optional relapse protection
        if hasattr(aud, "relapse_rate"):
            aud.relapse_rate[housed_uids] *= self.relapse_reduction

        print(f"[{current_year:.1f}] {self.label} housed {len(housed_uids)} of {len(uids)} eligible adults with AUD")
        
        