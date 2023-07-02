"""
On this page we need to import all folders and core functions, as well as doing metawork
"""

"""
=========
META WORK
=========
"""
#from ._globals import 
#from . import dtypes #could be useful to define own datatypes
from .exceptions import *
# items to trigger warnings/errors
__deprecationlist__ = {}
__expirationlist__ = {}
__all__ = [] #from pyfit import * shouldn't do anything


"""
============
CORE AND LIB
============
"""

from . import core
from .core import *
from . import lib
from .lib import *


"""
=======
FOLDERS
=======
"""
from . import datasets
from . import visualization
from . import nn
from . import rf

hello_world()