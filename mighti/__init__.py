from .version import __version__, __versiondate__, __license__

from .utils import *
from .life_expectancy import *
from .interactions import *
from .analyzers import *
from .disease_definitions import *  
from .prevalence_analyzer import *
from .plot_functions import *
from .diseases import *  
from .interventions import *


import sciris as sc
rootdir = sc.thispath(__file__).parent

# Import the version and print the license
print(__license__)