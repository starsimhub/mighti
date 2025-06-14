from .version import __version__, __versiondate__, __license__
from .utils import *
from .life_expectancy import *
from .interactions import *
from mighti.diseases import *  
from .sdoh import *

# Newly defined files
from .analyzers import *
from .disease_definitions import *  
from .prevalence_analyzer import *
from .survivorship_analyzer import *
from .plot_functions import *



# Set the root directory for the codebase
import pathlib
rootdir = pathlib.Path(__file__).parent

# Import the version and print the license
print(__license__)