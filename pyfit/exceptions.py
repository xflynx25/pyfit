"""
Overview: we define custom errors and warnings
"""

__all__ = ["DisplayDeprecationWarning", "MyCustomError"] #for export *


# will create a warning on screen about deprecation
class DisplayDeprecationWarning(UserWarning):
    pass

# create custom error for importing
class MyCustomError(RuntimeError):
    pass