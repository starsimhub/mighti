from .version import __version__, __versiondate__, __license__
from .utils import *
from .conditions import *
# from .conditions import initialize_conditions, disease_classes, get_param, create_disease_class
from .interactions import *
from .disease_definitions import *  
from .prevalence_analyzer import *  


# Set the root directory for the codebase
import pathlib
rootdir = pathlib.Path(__file__).parent

# Import the version and print the license
print(__license__)

