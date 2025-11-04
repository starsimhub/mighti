import starsim as ss
import stisim as sti
import mighti as mi

def test_budget_constraint_runs():
    ppl = ss.People(n_agents=5000)
    hiv = sti.HIV()
    art = mi.interventions.ART(cost_per_person_year=120)
    econ = mi.economics.BudgetConstraint(pars="data/examples/eswatini_hbp_template.csv")

    sim = ss.Sim(people=ppl, diseases=[hiv], interventions=[art], modules=[econ])
    sim.run()
    res = sim.results["budget_constraint"]
    assert res["total_cost"] > 0
    