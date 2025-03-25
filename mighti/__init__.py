from .version import __version__, __versiondate__, __license__
from .utils import *
from .custom_death import *
from .conditions import *
from .interactions import *
from .disease_definitions import *  
from .prevalence_analyzer import *  
from .plot_functions import *
from .run_simulation import *
from .life_expectancy import *
from .cea import *



# Set the root directory for the codebase
import pathlib
rootdir = pathlib.Path(__file__).parent

# Import the version and print the license
print(__license__)

