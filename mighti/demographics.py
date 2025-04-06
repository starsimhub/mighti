"""
Define pregnancy, deaths, migration, etc.
"""
import numpy as np
import starsim as ss
import sciris as sc
import pandas as pd

ss_float_ = ss.dtypes.float
ss_int_ = ss.dtypes.int

__all__ = ['Demographics', 'Births', 'Deaths', 'Pregnancy']


class Demographics(ss.Module):
    """
    A demographic module typically handles births/deaths/migration and takes
    place at the start of the timestep, before networks are updated and before
    any disease modules are executed.
    """
    pass

class Births(Demographics):
    """ Create births based on rates, rather than based on pregnancy """
    def __init__(self, pars=None, metadata=None, **kwargs):
        super().__init__()
        self.define_pars(
            unit = 'year',
            birth_rate = ss.peryear(20),
            rel_birth = 1,
            rate_units = 1e-3,  # assumes birth rates are per 1000. If using percentages, switch this to 1
        )
        self.update_pars(pars, **kwargs)

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = sc.mergedicts(
            sc.objdict(data_cols=dict(year='Year', value='CBR')),
            metadata,
        )

        # Process data, which may be provided as a number, dict, dataframe, or series
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.pars.birth_rate = self.standardize_birth_data()
        self.n_births = 0 # For results tracking
        return

    def init_pre(self, sim):
        """ Initialize with sim information """
        super().init_pre(sim)
        if isinstance(self.pars.birth_rate, pd.DataFrame):
            br_year = self.pars.birth_rate[self.metadata.data_cols['year']]
            br_val = self.pars.birth_rate[self.metadata.data_cols['cbr']]
            all_birth_rates = np.interp(self.timevec, br_year, br_val) # This assumes a year timestep -- probably ok?
            self.pars.birth_rate = all_birth_rates
        return

    def standardize_birth_data(self):
        """ Standardize/validate birth rates - handled in an external file due to shared functionality """
        birth_rate = ss.standardize_data(data=self.pars.birth_rate, metadata=self.metadata)
        if isinstance(birth_rate, (pd.Series, pd.DataFrame)):
            return birth_rate.xs(0,level='age')
        return birth_rate

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new',        dtype=int,   scale=True,  label='New births'),
            ss.Result('cumulative', dtype=int,   scale=True,  label='Cumulative births'),
            ss.Result('cbr',        dtype=float, scale=False, label='Crude birth rate'),
        )
        return

    def get_births(self):
        """
        Extract the right birth rates to use and translate it into a number of people to add.
        """
        sim = self.sim
        p = self.pars

        if isinstance(p.birth_rate, (pd.Series, pd.DataFrame)):
            available_years = p.birth_rate.index
            year_ind = sc.findnearest(available_years, sim.t.now('year'))
            nearest_year = available_years[year_ind]
            this_birth_rate = p.birth_rate.loc[nearest_year]
        else:
            this_birth_rate = p.birth_rate

        if isinstance(this_birth_rate, ss.TimePar):
            factor = 1.0
        else:
            factor = ss.time_ratio(unit1=self.t.unit, dt1=self.t.dt, unit2='year', dt2=1.0)

        scaled_birth_prob = this_birth_rate * p.rate_units * p.rel_birth * factor
        scaled_birth_prob = np.clip(scaled_birth_prob, a_min=0, a_max=1)
        
        n_new = np.random.binomial(n=sim.people.alive.count(), p=scaled_birth_prob) # Not CRN safe, see issue #404
        print(f"Number of new births: {n_new}")
        
        return n_new

    def step(self):
        new_uids = self.add_births()
        self.n_births = len(new_uids)
        return new_uids

    def add_births(self):
        """ Add n_new births to each state in the sim """
        people = self.sim.people
        n_new = self.get_births()
        new_uids = people.grow(n_new)
        people.age[new_uids] = 0
        return new_uids

    def update_results(self):
        # New births -- already calculated
        self.results.new[self.ti] = self.n_births

        # Calculate crude birth rate (CBR)
        inv_rate_units = 1.0/self.pars.rate_units
        births_per_year = self.n_births/self.sim.t.dt_year
        denom = self.sim.people.alive.sum()
        self.results.cbr[self.ti] = inv_rate_units*births_per_year/denom
        return

    def finalize(self):
        super().finalize()
        self.results.cumulative[:] = np.cumsum(self.results.new)
        return

class Deaths(Demographics):
    def __init__(self, pars=None, metadata=None, **kwargs):
        """
        Configure disease-independent "background" deaths.

        The probability of death for each agent on each timestep is determined
        by the `death_rate` parameter and the time step. The default value of
        this parameter is 0.02, indicating that all agents will
        face a 2% chance of death per year.

        However, this function can be made more realistic by using a dataframe
        for the `death_rate` parameter, to allow it to vary by year, sex, and
        age.  The separate 'metadata' argument can be used to configure the
        details of the input datafile.

        Alternatively, it is possible to override the `death_rate` parameter
        with a bernoulli distribution containing a constant value of function of
        your own design.

        Args:
            pars: dict with arguments including:
                rel_death: constant used to scale all death rates
                death_rate: float, dict, or pandas dataframe/series containing mortality data
                rate_units: units for death rates (see in-line comment on par dict below)

            metadata: data about the data contained within the data input.
                "data_cols" is is a dictionary mapping standard keys, like "year" to the
                corresponding column name in data. Similar for "sex_keys". Finally,
        """
        super().__init__()
        self.define_pars(
            unit = 'year',
            rel_death = 1,
            death_rate = ss.peryear(20),  # Default = a fixed rate of 2%/year, overwritten if data provided
            rate_units = 1e-3,  # assumes death rates are per 1000. If using percentages, switch this to 1
        )
        self.update_pars(pars, **kwargs)

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = sc.mergedicts(
            sc.objdict(
                data_cols = dict(year='Time', sex='Sex', age='AgeGrpStart', value='mx'),
                sex_keys = {'Female':'f', 'Male':'m'},
            ),
            metadata
        )

        # Process data, which may be provided as a number, dict, dataframe, or series
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.death_rate_data = self.standardize_death_data() # TODO: refactor
        self.pars.death_rate = ss.bernoulli(p=self.make_death_prob_fn)
        self.n_deaths = 0 # For results tracking
        self.death_tracking = {'Male': np.zeros(101), 'Female': np.zeros(101)}  # Initialize tracking dictionary
        self.infant_deaths = 0        
        return

    def standardize_death_data(self):
        """ Standardize/validate death rates - handled in an external file due to shared functionality """
        death_rate = ss.standardize_data(data=self.pars.death_rate, metadata=self.metadata)
        if isinstance(death_rate, (pd.Series, pd.DataFrame)):
            death_rate = death_rate.unstack(level='age')
            assert not death_rate.isna().any(axis=None) # For efficiency, we assume that the age bins are the same for all years in the input dataset
            print("Death rate data (after unstacking and checking for NaNs):")
            print(death_rate)
        return death_rate

    @staticmethod # Needs to be static since called externally, although it sure looks like a class method!
    def make_death_prob_fn(self, sim, uids):
        """ Take in the module, sim, and uids, and return the probability of death for each UID on this timestep """
        drd = self.death_rate_data
        if sc.isnumber(drd) or isinstance(drd, ss.TimePar):
            death_rate = drd

        # Process data
        else:
            ppl = sim.people

            # Performance optimization - the Deaths module checks for deaths for all agents
            # Therefore the UIDs requested should match all UIDs
            assert len(uids) == len(ppl.auids)

            available_years = drd.index.get_level_values('year')
            year_ind = sc.findnearest(available_years, sim.t.now('year')) # TODO: make work with different timesteps
            nearest_year = available_years[year_ind]

            death_rate = np.empty(uids.shape, dtype=ss_float_)

            if 'sex' in drd.index.names:
                s = drd.loc[nearest_year, 'f']
                binned_ages = np.digitize(ppl.age[ppl.female], s.index)-1 # Negative ages will be in the first bin - do *not* subtract 1 so that this bin is 0
                death_rate[ppl.female] = s.values[binned_ages]
                s = drd.loc[nearest_year, 'm']
                binned_ages = np.digitize(ppl.age[ppl.male], s.index)-1 # Negative ages will be in the first bin - do *not* subtract 1 so that this bin is 0
                death_rate[ppl.male] = s.values[binned_ages]
            else:
                s = drd.loc[nearest_year]
                binned_ages = np.digitize(ppl.age, s.index)-1 # Negative ages will be in the first bin - do *not* subtract 1 so that this bin is 0
                death_rate[:] = s.values[binned_ages]
                
        # Scale from rate to probability. Consider an exponential here.
        if isinstance(death_rate, ss.TimePar):
            factor = 1.0
        else:
            factor = ss.time_ratio(unit1=self.t.unit, dt1=self.t.dt, unit2='year', dt2=1.0)
        death_prob = death_rate * self.pars.rate_units * self.pars.rel_death * factor
        death_prob = np.clip(death_prob, a_min=0, a_max=1)
        return death_prob

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new',        dtype=int,   scale=True,   label='Deaths', auto_plot=False), # Use sim deaths instead
            ss.Result('cumulative', dtype=int,   scale=True,   label='Cumulative deaths', auto_plot=False),
            ss.Result('cmr',        dtype=float, scale=False,  label='Crude mortality rate'),
        )
        return

    def step(self):
        death_uids = self.pars.death_rate.filter()
        death_uids = np.unique(death_uids)  # Ensure death_uids are unique
        self.sim.people.request_death(death_uids)
        self.n_deaths = len(death_uids)
        people = self.sim.people

        infant_deaths = 0
        for uid in death_uids:
            age = int(people.age[uid])
            sex = 'Female' if people.female[uid] else 'Male'
            if age == 0:
                infant_deaths += 1
        
        # Track and print infant deaths
        self.infant_deaths = len([uid for uid in death_uids if self.sim.people.age[uid] < 1])        
        self.track_deaths_by_age_sex(death_uids)
        return self.n_deaths
    
    def track_deaths_by_age_sex(self, death_uids):
        people = self.sim.people
        age_sex_deaths = {}
    
        for uid in death_uids:
            age = int(people.age[uid])
            sex = 'Female' if people.female[uid] else 'Male'
            age_sex_key = (age, sex)
            if age_sex_key not in age_sex_deaths:
                age_sex_deaths[age_sex_key] = 0
            age_sex_deaths[age_sex_key] += 1
            
            if age == 0:
                self.infant_deaths += 1 
            self.death_tracking[sex][age] += 1        
        return age_sex_deaths
       
    def update_results(self):
        self.results['new'][self.ti] = self.n_deaths
        return
        
    def finalize(self):
        super().finalize()
        self.results.cumulative[:] = np.cumsum(self.results.new)
        units = self.pars.rate_units*self.sim.t.dt_year
        inds = self.match_time_inds()
        n_alive = self.sim.results.n_alive[inds]
        deaths = np.divide(self.results.new, n_alive, where=n_alive>0)
        self.results.cmr[:] = deaths/units
        return



class Pregnancy(Demographics):
    """ Create births via pregnancies """
    def __init__(self, pars=None, metadata=None, **kwargs):
        super().__init__()
        self.define_pars(
            unit = 'year',
            dur_pregnancy = ss.years(0.75), # Duration for pre-natal transmission
            dur_postpartum = ss.lognorm_ex(mean=ss.years(0.5), std=ss.years(0.5)), # Duration for post-natal transmission (e.g. via breastfeeding)
            fertility_rate = 0, # Can be a number of Pandas DataFrame
            rel_fertility = 1,
            p_maternal_death = ss.bernoulli(0),
            p_neonatal_death = ss.bernoulli(0),
            sex_ratio = ss.bernoulli(0.5), # Ratio of babies born female
            min_age = 15, # Minimum age to become pregnant
            max_age = 50, # Maximum age to become pregnant
            rate_units = 1e-3, # Assumes fertility rates are per 1000. If using percentages, switch this to 1
            burnin = True, # Should we seed pregnancies that would have happened before the start of the simulation?
            slot_scale = 5, # Random slots will be assigned to newborn agents between min=n_agents and max=slot_scale*n_agents
            min_slots  = 100, # Minimum number of slots, useful if the population size is very small
        )
        self.update_pars(pars, **kwargs)

        self.pars.p_fertility = ss.bernoulli(p=self.make_fertility_prob_fn)

        self.define_states(
            ss.State('fecund', default=True, label='Female of childbearing age'),
            ss.State('pregnant', label='Pregnant'),  # Currently pregnant
            ss.State('postpartum', label="Post-partum"),  # Currently post-partum
            ss.FloatArr('child_uid', label='UID of children, from embryo through postpartum'),
            ss.FloatArr('dur_postpartum', label='Post-partum duration'),  # Duration of postpartum phase
            ss.FloatArr('ti_pregnant', label='Time of pregnancy'),  # Time pregnancy begins
            ss.FloatArr('ti_delivery', label='Time of delivery'),  # Time of delivery
            ss.FloatArr('ti_postpartum', label='Time post-partum ends'),  # Time postpartum ends
            ss.FloatArr('ti_dead', label='Time of maternal death'),  # Maternal mortality
        )

        self.metadata = sc.mergedicts(
            sc.objdict(data_cols=dict(year='Time', age='AgeGrp', value='ASFR')),
            metadata,
        )
        self.choose_slots = None # Distribution for choosing slots; set in self.init()

        self.n_pregnancies = 0
        self.n_births = 0
        return

    @staticmethod
    def make_fertility_prob_fn(self, sim, uids):
        age = sim.people.age[uids]

        frd = self.fertility_rate_data
        fertility_rate = np.zeros(len(sim.people.uid.raw), dtype=ss_float_)

        time_factor = ss.time_ratio(unit1=self.t.unit, dt1=self.t.dt, unit2='year', dt2=1.0)
        if sc.isnumber(frd):
            fertility_rate[uids] = self.fertility_rate_data
            if isinstance(frd, ss.TimePar):
                time_factor = 1 
        else:
            year_ind = sc.findnearest(frd.index, self.t.now('year')-self.pars.dur_pregnancy.to('year'))
            nearest_year = frd.index[year_ind]

            age_bins = self.fertility_rate_data.columns.values
            age_bin_all = np.digitize(age, age_bins) - 1
            new_rate = self.fertility_rate_data.loc[nearest_year].values.copy()

            if (~self.fecund).any():
                v, c = np.unique(age_bin_all, return_counts=True)
                age_counts = np.zeros(len(age_bins))
                age_counts[v] = c

                age_bin_infecund = np.digitize(sim.people.age[~self.fecund], age_bins) - 1
                v, c = np.unique(age_bin_infecund, return_counts=True)
                infecund_age_counts = np.zeros(len(age_bins))
                infecund_age_counts[v] = c

                num_to_make = new_rate * age_counts
                new_denom = age_counts - infecund_age_counts
                np.divide(num_to_make, new_denom, where=new_denom>0, out=new_rate)

            fertility_rate[uids] = new_rate[age_bin_all]

        invalid_age = (age < self.pars.min_age) | (age > self.pars.max_age)
        fertility_prob = fertility_rate * (self.pars.rate_units * self.pars.rel_fertility) * time_factor
        fertility_prob[(~self.fecund).uids] = 0
        fertility_prob[uids[invalid_age]] = 0
        fertility_prob = np.clip(fertility_prob[uids], a_min=0, a_max=1)
        return fertility_prob

    def standardize_fertility_data(self):
        fertility_rate = ss.standardize_data(data=self.pars.fertility_rate, metadata=self.metadata)
        if isinstance(fertility_rate, (pd.Series, pd.DataFrame)):
            fertility_rate = fertility_rate.unstack()
            fertility_rate = fertility_rate.reindex(np.arange(fertility_rate.index.min(), fertility_rate.index.max() + 1)).interpolate()
            max_age = fertility_rate.columns.max()
            fertility_rate[max_age + 1] = 0
            assert not fertility_rate.isna().any(axis=None)
        return fertility_rate

    def init_pre(self, sim):
        super().init_pre(sim)
        self.fertility_rate_data = self.standardize_fertility_data()
        self.pars.p_fertility.set(p=self.make_fertility_prob_fn)

        low = sim.pars.n_agents + 1
        high = int(self.pars.slot_scale*sim.pars.n_agents)
        high = np.maximum(high, self.pars.min_slots)
        self.choose_slots = ss.randint(low=low, high=high, sim=sim, module=self)
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('pregnancies', dtype=int,   scale=True,  label='New pregnancies'),
            ss.Result('births',      dtype=int,   scale=True,  label='New births'),
            ss.Result('cbr',         dtype=float, scale=False, label='Crude birth rate'),
        )
        return

    def step(self):
        if self.ti == 0 and self.pars.burnin:
            dtis = np.arange(np.ceil(-1 * self.pars.dur_pregnancy), 0, 1).astype(int)
            for dti in dtis:
                self.t.ti = dti
                self.do_step()
            self.t.ti = 0
        new_uids = self.do_step()
        return new_uids

    def do_step(self):
        self.update_states()
        conceive_uids = self.make_pregnancies()
        self.n_pregnancies += len(conceive_uids)
        new_uids = self.make_embryos(conceive_uids)
        return new_uids

    def update_states(self):
        ti = self.ti
        deliveries = self.pregnant & (self.ti_delivery <= ti)
        self.n_births = np.count_nonzero(deliveries)
        self.pregnant[deliveries] = False
        self.postpartum[deliveries] = True
        self.fecund[deliveries] = False

        for lkey, layer in self.sim.networks.items():
            if layer.postnatal and self.n_births:
                prenatalnet = [nw for nw in self.sim.networks.values() if nw.prenatal][0]
                prenatal_ending = prenatalnet.edges.end <= ti
                new_mother_uids = prenatalnet.edges.p1[prenatal_ending]
                new_infant_uids = prenatalnet.edges.p2[prenatal_ending]

                if not set(new_mother_uids) == set(deliveries.uids):
                    errormsg = 'IDs of new mothers do not match IDs of new deliveries'
                    raise ValueError(errormsg)

                durs = self.dur_postpartum[new_mother_uids]
                start = np.full(self.n_births, fill_value=ti)

                prenatalnet.end_pairs()
                layer.add_pairs(new_mother_uids, new_infant_uids, dur=durs, start=start)

        postpartum = self.postpartum & (self.ti_postpartum <= ti)
        self.postpartum[postpartum] = False
        self.fecund[postpartum] = True
        self.child_uid[postpartum] = np.nan

        maternal_deaths = (self.ti_dead <= ti).uids
        self.sim.people.request_death(maternal_deaths)
        return

    def make_pregnancies(self):
        eligible_uids = self.sim.people.female.uids
        conceive_uids = self.pars.p_fertility.filter(eligible_uids)

        if len(conceive_uids) == 0:
            return ss.uids()

        if np.any(self.pregnant[conceive_uids]):
            which_uids = conceive_uids[self.pregnant[conceive_uids]]
            errormsg = f'New conceptions registered in {len(which_uids)} pregnant agent(s) at timestep {self.ti}.'
            raise ValueError(errormsg)

        self.set_prognoses(conceive_uids)
        return conceive_uids

    def make_embryos(self, conceive_uids):
        people = self.sim.people
        n_unborn = len(conceive_uids)
        if n_unborn == 0:
            new_uids = ss.uids()
        else:
            new_slots = self.choose_slots.rvs(conceive_uids)
            new_uids = people.grow(len(new_slots), new_slots)
            people.age[new_uids] = -self.pars.dur_pregnancy.to('year')
            people.slot[new_uids] = new_slots
            people.female[new_uids] = self.pars.sex_ratio.rvs(conceive_uids)
            people.parent[new_uids] = conceive_uids
            self.child_uid[conceive_uids] = new_uids

            for lkey, layer in self.sim.networks.items():
                if layer.prenatal:
                    durs = np.full(n_unborn, fill_value=self.pars.dur_pregnancy)
                    start = np.full(n_unborn, fill_value=self.ti)
                    layer.add_pairs(conceive_uids, new_uids, dur=durs, start=start)

        if self.ti < 0:
            people.age[new_uids] += -self.ti * self.sim.t.dt_year

        return new_uids

    def set_prognoses(self, uids):
        ti = self.ti
        self.fecund[uids] = False
        self.pregnant[uids] = True
        self.ti_pregnant[uids] = ti

        dur_preg = np.ones(len(uids))*self.pars.dur_pregnancy
        dur_postpartum = self.pars.dur_postpartum.rvs(uids)
        dead = self.pars.p_maternal_death.rvs(uids)
        self.ti_delivery[uids] = ti + dur_preg
        self.ti_postpartum[uids] = self.ti_delivery[uids] + dur_postpartum
        self.dur_postpartum[uids] = dur_postpartum

        if np.any(dead):
            self.ti_dead[uids[dead]] = ti + dur_preg[dead]
        return

    def get_births(self, year):
        """ Retrieve the number of births for a given year """
        births = 0
        for ti in range(len(self.sim.t.yearvec)):
            if self.sim.t.yearvec[ti] == year:
                births += self.results.births[ti]
        return births

    def update_results(self):
        ti = self.ti
        self.results['pregnancies'][ti] = self.n_pregnancies
        self.results['births'][ti] = self.n_births

        # Reset for the next step
        self.n_pregnancies = 0
        self.n_births = 0

        return

    def finalize(self):
        super().finalize()
        units = self.pars.rate_units*self.sim.t.dt_year
        inds = self.match_time_inds()
        n_alive = self.sim.results.n_alive[inds]
        births = np.divide(self.results['births'], n_alive, where=n_alive>0)
        self.results['cbr'][:] = births/units
        return