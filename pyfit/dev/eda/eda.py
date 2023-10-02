__all__ = ['EDA']

from . import bivariate, univariate, multivariate, dataClean
from .. import dev_helpers
import os

class EDA:
    def __init__(self, rootdir, X, y):
        self.working_directory = os.path.join(rootdir, 'eda')
        dev_helpers.init_directories(self.working_directory)

        self.data = X
        self.target = y
        #self.univariate = bivariate.UnivariateAnalysis(data, target)
        #self.bivariate = univariate.BivariateAnalysis(data, target)
        #self.multivariate = multivariate.MultivariateAnalysis(data, target)
        self.cleaning_info = dataClean.DataCleaningInfo(X, y)
        

    def default_eda(self):
        # You can call the get_info method here to print out and return data cleaning info
        cleaning_info = self.cleaning_info.get_info()
        # Then you might call default analysis methods from your univariate, bivariate, and multivariate classes
        # For example:
        # self.univariate.default_analysis()
        # self.bivariate.default_analysis()
        # self.multivariate.default_analysis()
        # And so on...
    
    # maybe some json like order form, of comparisons / experiments to look at
    def specific_eda(self, columns):
        # Call specific EDA methods for provided columns
        pass