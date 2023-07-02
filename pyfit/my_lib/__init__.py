"""
This is a private module that should contain the misc objects that you want in full pyfit namespace for submodule purposes
Basically, a general_helpers file. It should exists inside other modules... 
But maybe this is a bad idea because numpy was hoping to remove it to clean up the namespace, these are just floating around
So we can keep in for now, but probably will try not to use
"""

from . import example_file
from .example_file import *