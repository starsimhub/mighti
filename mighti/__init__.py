from .version import __version__, __versiondate__, __license__

from .utils import *
from .sdoh import *
from .people_extend import *
from .life_expectancy import *
from .interactions import *
from .analyzers import *
from .disease_definitions import *  
from .plot_functions import *
from .diseases import *  
from .interventions import *
from .adherence import *
from . import economics  # ensure `mighti.economics` is available for tests/backcompat


import sciris as sc
rootdir = sc.thispath(__file__).parent


# ---------------------------------------------------------------------
# Starsim 3.x compatibility shims
# ---------------------------------------------------------------------
try:
    import starsim as ss  # noqa: F401

    if not hasattr(ss.Sim, "get_module"):
        def _get_module(self, name, optional=False):
            """
            Backwards-compatible accessor for modules by label/name.

            Starsim 3.x no longer exposes `Sim.get_module()` in some releases,
            but MIGHTI code and tests still call it.
            """
            target = str(name).lower()

            mods = getattr(self, "modules", None)
            if mods is None:
                if optional:
                    return None
                raise KeyError(f"No modules attached; cannot find '{name}'")

            # dict / odict-like
            if hasattr(mods, "items"):
                for k, m in mods.items():
                    if str(k).lower() == target:
                        return m
                    lab = getattr(m, "label", None)
                    if isinstance(lab, str) and lab.lower() == target:
                        return m
                    if m.__class__.__name__.lower() == target:
                        return m
                if optional:
                    return None
                raise KeyError(f"Module '{name}' not found; available={list(mods.keys())}")

            # list-like
            for m in mods:
                if m is None:
                    continue
                lab = getattr(m, "label", None)
                if isinstance(lab, str) and lab.lower() == target:
                    return m
                if m.__class__.__name__.lower() == target:
                    return m

            if optional:
                return None
            raise KeyError(f"Module '{name}' not found")

        ss.Sim.get_module = _get_module
except Exception:
    # If starsim isn't importable in some environments, don't prevent importing mighti
    pass

# Import the version and print the license
print(__license__)