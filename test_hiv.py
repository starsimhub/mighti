import starsim as ss
import stisim as sti
import mighti as mi
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
from collections import defaultdict



class TrackValues(ss.Analyzer):
    # Track outputs for viral load and CD4 counts
    # Assumes no births; for diagnostic/debugging purposes only
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_pars(
            unit='month',
        )

    def init_pre(self, sim):
        super().init_pre(sim)
        self.n = len(sim.people)

        self.hiv_rel_sus = np.empty((sim.t.npts, self.n), dtype=ss.dtypes.float)
        self.hiv_rel_trans = np.empty((sim.t.npts, self.n), dtype=ss.dtypes.float)

        self.syph_rel_sus = np.empty((sim.t.npts, self.n), dtype=ss.dtypes.float)
        self.syph_rel_trans = np.empty((sim.t.npts, self.n), dtype=ss.dtypes.float)

        self.cd4 = np.empty((sim.t.npts, self.n), dtype=ss.dtypes.float)

        self.care_seeking = np.empty((sim.t.npts, self.n), dtype=ss.dtypes.float)

    @property
    def has_hiv(self):
        return 'hiv' in self.sim.diseases

    @property
    def has_syph(self):
        return isinstance(self.sim.diseases.get('syphilis'), sti.Syphilis)

    # def step(self):
    #     ti = self.ti
    #     if self.has_hiv:
    #         self.hiv_rel_sus[ti, :self.n] = self.sim.diseases.hiv.rel_sus.values[:self.n]
    #         self.hiv_rel_trans[ti, :self.n] = self.sim.diseases.hiv.rel_trans.values[:self.n]

    #     if self.has_syph:
    #         self.syph_rel_sus[ti, :self.n] = self.sim.diseases.syphilis.rel_sus.values[:self.n]
    #         self.syph_rel_trans[ti, :self.n] = self.sim.diseases.syphilis.rel_trans.values[:self.n]

    #     self.cd4[ti, :self.n] = self.sim.diseases.hiv.cd4.values[:self.n]
    #     self.care_seeking[ti, :self.n] = self.sim.diseases.hiv.care_seeking[:self.n]
    
    def step(self):
        ti = self.ti
        n_alive = len(self.sim.people)
        if self.has_hiv:
            self.hiv_rel_sus[ti, :n_alive] = self.sim.diseases.hiv.rel_sus.values[:n_alive]
            self.hiv_rel_trans[ti, :n_alive] = self.sim.diseases.hiv.rel_trans.values[:n_alive]
            self.cd4[ti, :n_alive] = self.sim.diseases.hiv.cd4.values[:n_alive]
            self.care_seeking[ti, :n_alive] = self.sim.diseases.hiv.care_seeking[:n_alive]

    def plot(self, agents: dict):
        """
        :param agents: Dictionary of events per agent {'agent_description':[('event_type', ti),...]}
        :return: Matplotlib figure
        """

        def plot_with_events(ax, x, y, agents, title):
            h = ax.plot(x, y)
            x_ev = []
            y_ev = []
            for i, events in enumerate(agents.values()):
                for event in events:
                    x_ev.append(self.sim.timevec[event[1]])
                    y_ev.append(y[event[1], i])
            ax.scatter(x_ev, y_ev, marker='*', color='yellow', edgecolor='red', s=100, linewidths=0.5, zorder=100)
            ax.set_title(title)
            return h

        if self.has_syph:
            fig, ax = plt.subplots(2, 4)
        else:
            fig, ax = plt.subplots(1, 2)

        ax = ax.ravel()

        h = plot_with_events(ax[0], self.sim.timevec, self.cd4, agents, 'CD4')
        # h = plot_with_events(ax[1], self.sim.timevec, self.hiv_rel_sus, agents, 'HIV rel_sus')
        h = plot_with_events(ax[1], self.sim.timevec, self.hiv_rel_trans, agents, 'HIV rel_trans')
        # h = plot_with_events(ax[3], self.sim.timevec, self.care_seeking, agents, 'HIV care seeking')

        if self.has_syph:
            h = plot_with_events(ax[4], self.sim.timevec, self.syph_rel_sus, agents, 'Syphilis rel_sus')
            h = plot_with_events(ax[5], self.sim.timevec, self.syph_rel_trans, agents, 'Syphilis rel_trans')

        # fig.legend(h, agents.keys(), loc='upper right', bbox_to_anchor=(1.1, 1))

        return fig


class PerformTest(ss.Intervention):

    def __init__(self, events=None):
        """
        :param events: List of (uid, 'event', ti) to apply events to an agent
        """
        super().__init__()
        self.define_pars(
            unit='month',
        )
        self.hiv_infections = defaultdict(list)
        self.syphilis_infections = defaultdict(list)
        self.art_start = defaultdict(list)
        self.art_stop = defaultdict(list)
        self.pregnant = defaultdict(list)

        if events:
            for uid, event, ti in events:
                if event == 'hiv_infection':
                    self.hiv_infections[ti].append(uid)
                elif event == 'syphilis_infection':
                    self.syphilis_infections[ti].append(uid)
                elif event == 'art_start':
                    self.art_start[ti].append(uid)
                elif event == 'art_stop':
                    self.art_stop[ti].append(uid)
                elif event == 'pregnant':
                    self.pregnant[ti].append(uid)
                else:
                    raise Exception(f'Unknown event "{event}"')

    def initiate_ART(self, uids):
        if len(uids):
            self.sim.diseases.hiv.start_art(ss.uids(uids))

    def end_ART(self, uids):
        if len(uids):
            self.sim.diseases.hiv.stop_art(ss.uids(uids))

    def set_pregnancy(self, uids):
        self.sim.demographics.pregnancy.pregnant[ss.uids(uids)] = True
        self.sim.demographics.pregnancy.ti_pregnant[ss.uids(uids)] = self.sim.ti

    def step(self):
        sim = self.sim
        ti = self.ti
        self.initiate_ART(self.art_start[ti])
        self.end_ART(self.art_stop[ti])
        if 'hiv' in sim.diseases:
            self.sim.diseases.hiv.set_prognoses(ss.uids(self.hiv_infections[ti]))
        if 'syphilis' in sim.diseases:
            self.sim.diseases.syphilis.set_prognoses(ss.uids(self.syphilis_infections[ti]))

        # Set pregnancies:
        self.set_pregnancy(self.pregnant[ti])
        
        
        
        
def test_hiv():
    # AGENTS
    agents = sc.odict()
    agents['No infection'] = []
    agents['Infection without ART'] = [('hiv_infection', 5)]
    agents['Goes onto ART early (CD4 > 200) and stays on forever'] = [('hiv_infection', 4), ('art_start', 1 * 12)]
    agents['Goes onto ART late (CD4 < 200) and stays on forever'] = [('hiv_infection', 3), ('art_start', 10 * 12)]
    agents['Goes off ART with CD4 > 200'] = [('hiv_infection', 2), ('art_start', 5 * 12), ('art_stop', 12 * 12)]
    agents['Goes off ART with CD4 < 200'] = [('hiv_infection', 2), ('art_start', 12 * 12), ('art_stop', 13 * 12)]
    agents['pregnant'] = [('pregnant', 5), ('hiv_infection', 10)]
    
    events = []
    for i, x in enumerate(agents.values()):
        for y in x:
            events.append((i,) + y)
            
    n_agents = len(agents)
    start = 2007
    stop = 2020
    dt = 1/12
    
    # Eswatini-style interventions (see previous answer for rationale)
    test_prob_years = [2007, 2012, 2016]
    test_prob_vals = [0.3, 0.7, 0.95]
    art_cov = {'future_coverage': {'year': 2010, 'prop': 0.95}}
    vmmc_cov = {'future_coverage': {'year': 2015, 'prop': 0.3}}
    prep_cov = {'coverage': [0, 0.05, 0.25], 'years': [2007, 2015, 2020]}

    interventions = [
        sti.HIVTest(test_prob_data=test_prob_vals, years=test_prob_years),
        sti.ART(pars=art_cov),
        sti.VMMC(pars=vmmc_cov),
        sti.Prep(pars=prep_cov),
    ]

    hiv = sti.HIV(
        init_prev=0.2,
        include_aids_deaths=True,
        beta={'structuredsexual': [0.001, 0.001], 'maternal': [0.01, 0.01]}
    )

    networks = [sti.StructuredSexual(), ss.MaternalNet()]
    demographics = [ss.Pregnancy(fertility_rate=0.03), ss.Deaths(death_rate=0.01)]
    output = TrackValues()

    # # Initialize the PrevalenceAnalyzer
    # survivorship_analyzer = mi.SurvivorshipAnalyzer()
    # deaths_analyzer = mi.DeathsByAgeSexAnalyzer()
    
    pars = dict(
        n_agents=n_agents,
        start=start,
        stop=stop,
        dt=dt,
        diseases=[hiv],
        networks=networks,
        demographics=demographics,
        interventions=interventions,
        analyzers=output,
        label="Eswatini HIV scenario"
    )

    sim = ss.Sim(pars, copy_inputs=False).run()
    fig = output.plot(agents)
    return sim

if __name__ == "__main__":
    test_hiv()
    plt.show()