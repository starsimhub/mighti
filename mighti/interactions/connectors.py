"""
Interactions between conditions: having one condition changes the risk of acquiring another

Each interaction is an `ss.modify()` of the susceptibility (`rel_sus`) of the condition acquired, where
the other condition holds. Modifiers multiply a base that is never overwritten, so several interactions
on one condition combine, and stop when the condition that causes them ends (e.g. at remission).
"""

from collections import defaultdict
import pandas as pd
import starsim as ss

__all__ = ['NCDHIVConnector', 'read_interactions', 'create_connectors', 'create_dynamic_connector']


def key(condition):
    """ The name of a condition in the sim, e.g. 'type2diabetes' for 'Type2Diabetes' """
    return str(condition).strip().lower()


def in_sim(condition, diseases):
    """ Whether a condition is among the diseases of the sim (by name), or no list of them was given """
    return diseases is None or key(condition) in {key(d) for d in diseases}


def NCDHIVConnector(rel_sus, diseases=None):
    """
    Raise the susceptibility to conditions among people with HIV

    Args:
        rel_sus (dict): for each condition (by name, e.g. 'Type2Diabetes'), the relative susceptibility of people with HIV (e.g. the `rel_sus` column of the parameter file)
        diseases (list): the names of the conditions in the sim, if `rel_sus` has others (a modifier of a condition that is not in the sim is an error)

    Returns:
        A list of `ss.modify()`, one per condition, to pass as `connectors`
    """
    return [ss.modify(f'{key(cond)}.rel_sus', multiply=float(rr), where=ss.state('hiv.infected'), name=f'ncd_hiv_{key(cond)}')
            for cond, rr in rel_sus.items() if key(cond) != 'hiv' and pd.notna(rr) and in_sim(cond, diseases)]


def read_interactions(datafile=None):
    """
    Read the relative susceptibilities between conditions from a CSV file: a row per condition that raises the
    risk, and a column per condition whose risk it raises (blank for none)

    Returns:
        dict: for each condition, a dict of the relative susceptibility to each other condition
    """
    df = pd.read_csv(datafile or 'rel_sus.csv', index_col=0)
    rel_sus = defaultdict(dict)
    for condition1 in df.index:
        for condition2 in df.columns:
            value = df.at[condition1, condition2]
            if condition1 != condition2 and not pd.isna(value):
                rel_sus[condition1][condition2] = value
    return rel_sus


def create_connectors(rel_sus, diseases=None):
    """ An interaction for each pair of conditions in `rel_sus` (see `read_interactions()`), of those among `diseases` (by name) if given """
    return [create_dynamic_connector(c1, c2, rr) for c1, others in rel_sus.items() for c2, rr in others.items() if in_sim(c1, diseases) and in_sim(c2, diseases)]


def create_dynamic_connector(condition1, condition2, rel_sus, state1='affected'):
    """
    People with `condition1` have `rel_sus` times the susceptibility to `condition2`

    Args:
        condition1 (str): the condition that raises the risk, by name (e.g. 'Type2Diabetes', or 'HIV')
        condition2 (str): the condition whose risk it raises
        rel_sus (float): the relative susceptibility
        state1 (str): the state of `condition1` that raises the risk: `affected`, or `infected` for HIV and the generic infections (found automatically for 'HIV' and the classes `GenericSIS` and `GenericSIR`)

    Returns:
        An `ss.modify()`, to pass as one of the `connectors`
    """
    c1, c2 = key(condition1), key(condition2)
    where = ss.state(f'{c1}.infected' if c1 == 'hiv' or is_infection(condition1) else f'{c1}.{state1}')
    return ss.modify(f'{c2}.rel_sus', multiply=float(rel_sus), where=where, name=f'{c1}_{c2}')


def is_infection(condition):
    """ Whether a condition, by name, is one of MIGHTI's generic infections, whose state is `infected` """
    import mighti as mi
    cls = getattr(mi.diseases, str(condition).strip(), None)
    return isinstance(cls, type) and issubclass(cls, (mi.diseases.GenericSIS, mi.diseases.GenericSIR))
