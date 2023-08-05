import numpy as np 
from sklearn.datasets import make_classification, make_regression

__all__ = ['DataGen']


class DataGen:
    def __init__(self):
        pass

    # first dim of dimensions is number of samples. Rest is size of the data
    def make_data(self, style: str, dimensions: np.array):
        if style == 'random': 
            self.data = [[1,2],[2,3]]
        else:
            self.data = "something nice"
            
    @staticmethod
    def make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1):
        """
        Wrapper around sklearn.datasets.make_classification for consistency
        """
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, 
                                   n_redundant=n_redundant, n_clusters_per_class=n_clusters_per_class)
        return X, y

    @staticmethod
    def make_regression(n_samples=100, n_features=1, noise=0.1):
        """
        Wrapper around sklearn.datasets.make_regression for consistency
        """
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
        return X, y

    @staticmethod
    def random_data(n_samples=100, n_features=2):
        """
        Generates completely random dataset
        """
        X = np.random.random((n_samples, n_features))
        y = np.random.randint(0, 2, n_samples)
        return X, y
