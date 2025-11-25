"""
Test script to check if STIsim's base ART module has built-in dropout.
"""
import numpy as np
import pandas as pd
import starsim as ss
import stisim as sti
import mighti as mi

# Simple simulation setup
n_agents = 10_000
inityear = 2007
endyear = 2015  # Short simulation to see dropout quickly

# Create people
ppl = ss.People(n_agents)

# HIV disease
hiv = sti.HIV()
hiv.pars.include_care = True
hiv.pars.art_efficacy = 0.9
hiv.pars.init_prev = ss.bernoulli(p=0.15)

# ART intervention - use BASE sti.ART (not our extension)
art_coverage_data = pd.DataFrame(
    {"p_art": [0.10, 0.50, 0.74]},
    index=[2003, 2013, 2016],
)
hiv_test = sti.HIVTest(
    test_prob_data=[0.10, 0.60, 0.80],
    years=[2003, 2007, 2014],
)
art = sti.ART(coverage_data=art_coverage_data)  # BASE STIsim ART

# Add analyzer to track dropout
class DropoutTracker(ss.Analyzer):
    def __init__(self):
        super().__init__()
    
    def step(self):
        track_dropout(self.sim)

# Simple simulation with minimal networks
sim = ss.Sim(
    n_agents=n_agents,
    start=inityear,
    stop=endyear,
    people=ppl,
    networks=[ss.MaternalNet()],
    demographics=[ss.Deaths(), ss.Pregnancy()],
    diseases=[hiv],
    interventions=[hiv_test, art],
    analyzers=DropoutTracker(),
    rand_seed=42,
    label="Test STIsim ART dropout"
)

print("="*80)
print("TESTING STISIM BASE ART MODULE FOR DROPOUT")
print("="*80)
print(f"Simulation: {inityear} to {endyear}")
print(f"ART coverage: {art_coverage_data['p_art'].values}")
print()

# Track dropout over time
dropout_tracker = []

def track_dropout(sim):
    """Track dropout at each timestep."""
    st = sim.people.states
    ppl = sim.people
    
    hiv_diagnosed = np.asarray(st.get("hiv.diagnosed", []), bool)
    hiv_on_art = np.asarray(st.get("hiv.on_art", []), bool)
    alive = np.asarray(ppl.alive, bool)
    
    # Only look at alive, diagnosed people
    eligible = hiv_diagnosed & alive
    on_art = hiv_on_art & eligible
    
    # Check ti_stop_art
    ti_stop_art = None
    if "hiv.ti_stop_art" in st:
        ti_stop_art_raw = st.get("hiv.ti_stop_art", [])
        if hasattr(ti_stop_art_raw, 'values'):
            ti_stop_art = np.asarray(ti_stop_art_raw.values, dtype=float)
        else:
            ti_stop_art = np.asarray(ti_stop_art_raw, dtype=float)
    
    # Check ti_art (when they started)
    ti_art = None
    if "hiv.ti_art" in st:
        ti_art_raw = st.get("hiv.ti_art", [])
        if hasattr(ti_art_raw, 'values'):
            ti_art = np.asarray(ti_art_raw.values, dtype=float)
        else:
            ti_art = np.asarray(ti_art_raw, dtype=float)
    
    # Count people
    n_diagnosed = eligible.sum()
    n_on_art = on_art.sum()
    
    # Count people who ever started ART (have ti_art set)
    n_ever_started = 0
    if ti_art is not None:
        ever_started = (ti_art[eligible] >= 0) & np.isfinite(ti_art[eligible])
        n_ever_started = ever_started.sum()
    
    # Count people who have ti_stop_art set
    n_has_stop_scheduled = 0
    n_stopped = 0
    if ti_stop_art is not None:
        has_stop = (ti_stop_art[eligible] >= 0) & np.isfinite(ti_stop_art[eligible])
        n_has_stop_scheduled = has_stop.sum()
        
        # People who have ti_stop_art set but are NOT on ART (stopped)
        stopped = has_stop & ~on_art if len(has_stop) == len(on_art) else np.array([], dtype=bool)
        n_stopped = stopped.sum()
    
    # Count dropout (ever started but not currently on ART, and alive)
    n_dropout = 0
    if ti_art is not None:
        ever_started_mask = (ti_art[eligible] >= 0) & np.isfinite(ti_art[eligible])
        dropout_mask = ever_started_mask & ~on_art if len(ever_started_mask) == len(on_art) else np.array([], dtype=bool)
        n_dropout = dropout_mask.sum()
    
    # Print every year
    if sim.ti % 12 == 0 or sim.ti < 3:
        print(f"Year {sim.t.year:.1f}, ti={sim.ti}:")
        print(f"  Diagnosed (alive): {n_diagnosed}")
        print(f"  On ART: {n_on_art} ({n_on_art/n_diagnosed:.3f} if n_diagnosed > 0 else 0.0)")
        print(f"  Ever started ART: {n_ever_started}")
        print(f"  Has ti_stop_art set: {n_has_stop_scheduled}")
        print(f"  Stopped (ti_stop_art set AND not on ART): {n_stopped}")
        print(f"  Dropout (ever started but not on ART, alive): {n_dropout}")
        
        # Check ti_stop_art values for people on ART
        if ti_stop_art is not None and on_art.any():
            ti_stop_art_on_art = ti_stop_art[on_art & eligible]
            if len(ti_stop_art_on_art) > 0:
                print(f"  ti_stop_art range for people ON ART: [{ti_stop_art_on_art.min():.1f}, {ti_stop_art_on_art.max():.1f}]")
                print(f"  Current ti: {sim.ti}")
                # Count how many should have stopped by now
                should_stop = (ti_stop_art_on_art <= sim.ti) & np.isfinite(ti_stop_art_on_art)
                print(f"  Should have stopped by now (ti_stop_art <= sim.ti): {should_stop.sum()}")
        
        print()
    
    dropout_tracker.append({
        'year': sim.t.year,
        'ti': sim.ti,
        'n_diagnosed': n_diagnosed,
        'n_on_art': n_on_art,
        'n_ever_started': n_ever_started,
        'n_has_stop_scheduled': n_has_stop_scheduled,
        'n_stopped': n_stopped,
        'n_dropout': n_dropout,
    })

# Run simulation
print("Running simulation...")
print()
sim.run()

# Final summary
print("="*80)
print("FINAL SUMMARY")
print("="*80)
st = sim.people.states
ppl = sim.people

hiv_diagnosed = np.asarray(st.get("hiv.diagnosed", []), bool)
hiv_on_art = np.asarray(st.get("hiv.on_art", []), bool)
alive = np.asarray(ppl.alive, bool)

eligible = hiv_diagnosed & alive
on_art = hiv_on_art & eligible

ti_stop_art = None
if "hiv.ti_stop_art" in st:
    ti_stop_art_raw = st.get("hiv.ti_stop_art", [])
    if hasattr(ti_stop_art_raw, 'values'):
        ti_stop_art = np.asarray(ti_stop_art_raw.values, dtype=float)
    else:
        ti_stop_art = np.asarray(ti_stop_art_raw, dtype=float)

ti_art = None
if "hiv.ti_art" in st:
    ti_art_raw = st.get("hiv.ti_art", [])
    if hasattr(ti_art_raw, 'values'):
        ti_art = np.asarray(ti_art_raw.values, dtype=float)
    else:
        ti_art = np.asarray(ti_art_raw, dtype=float)

n_diagnosed = eligible.sum()
n_on_art = on_art.sum()

if ti_art is not None:
    ever_started = (ti_art[eligible] >= 0) & np.isfinite(ti_art[eligible])
    n_ever_started = ever_started.sum()
    dropout = ever_started & ~on_art if len(ever_started) == len(on_art) else np.array([], dtype=bool)
    n_dropout = dropout.sum()
    dropout_rate = n_dropout / n_ever_started if n_ever_started > 0 else 0.0
    print(f"Diagnosed (alive): {n_diagnosed}")
    print(f"On ART: {n_on_art} ({n_on_art/n_diagnosed:.3f})")
    print(f"Ever started ART: {n_ever_started}")
    print(f"Dropout (ever started but not on ART): {n_dropout} ({dropout_rate:.3f})")

if ti_stop_art is not None:
    has_stop = (ti_stop_art[eligible] >= 0) & np.isfinite(ti_stop_art[eligible])
    n_has_stop = has_stop.sum()
    stopped = has_stop & ~on_art if len(has_stop) == len(on_art) else np.array([], dtype=bool)
    n_stopped = stopped.sum()
    print(f"Has ti_stop_art set: {n_has_stop}")
    print(f"Stopped (ti_stop_art set AND not on ART): {n_stopped}")
    
    # Check ti_stop_art values
    if on_art.any():
        ti_stop_art_on_art = ti_stop_art[on_art & eligible]
        if len(ti_stop_art_on_art) > 0:
            print(f"ti_stop_art range for people ON ART: [{ti_stop_art_on_art.min():.1f}, {ti_stop_art_on_art.max():.1f}]")
            print(f"Current ti: {sim.ti}")
            should_stop = (ti_stop_art_on_art <= sim.ti) & np.isfinite(ti_stop_art_on_art)
            print(f"Should have stopped by now: {should_stop.sum()}")

print()
print("="*80)
print("CONCLUSION:")
print("="*80)
if n_dropout > 0:
    print("✓ STIsim ART DOES have dropout (people started but are not on ART)")
else:
    print("✗ STIsim ART does NOT have dropout (everyone who started is still on ART)")
if ti_stop_art is not None and n_has_stop > 0:
    if ti_stop_art_on_art.max() > sim.ti * 10:
        print(f"  → ti_stop_art is set to very large values (>{sim.ti * 10}), meaning no scheduled stops")
    else:
        print(f"  → ti_stop_art is set, but may not be triggering stops")
print()

