# datagen makes data for you, for your practice
# 1. Random classification or regression, of different difficulties
# 2. random data
# 3. premade datasets
# all are accessible via the wrapper make_data
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification, make_regression

class DataGen:
    def __init__(self, n_samples=100, n_features=2, difficulty=1, noise=0.1):
        self.n_samples = n_samples
        self.n_features = n_features
        self.difficulty = difficulty
        self.noise = noise

    def make_dataset(self, **kwargs):
        name = kwargs.get('name') 
        classification_datasets = ['iris', 'digits', 'wine', 'breast_cancer']
        regression_datasets = ['boston', 'diabetes', 'linnerud']

        if name in classification_datasets:
            loader = getattr(datasets, f"load_{name}")
            data = loader()
            return data.data, data.target
        elif name in regression_datasets:
            loader = getattr(datasets, f"load_{name}")
            data = loader()
            return data.data, data.target
        else:
            raise ValueError(f"Unknown dataset name: {name}")

    def make_classification(self, **kwargs):
        n_samples = kwargs.get('n_samples') or self.n_samples
        difficulty = kwargs.get('difficulty') or self.difficulty

        # other parameter checks and default assignments can go here
        if not 1 <= difficulty <= 5:
            raise ValueError("Difficulty level must be between 1 and 5")

        n_features = int((30 - difficulty * 5) * n_samples / 30)
        n_informative = max(2, n_features - difficulty)
        n_redundant = min(difficulty, n_features - n_informative)
        n_clusters_per_class = 1 if difficulty < 3 else 2

        X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                   n_informative=n_informative, n_redundant=n_redundant,
                                   n_clusters_per_class=n_clusters_per_class)
        return X, y

    def make_regression(self, **kwargs):
        n_samples = kwargs.get('n_samples') or self.n_samples
        n_features = kwargs.get('n_features') or self.n_features
        noise = kwargs.get('noise', self.noise)

        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
        return X, y

    def make_random_data(self, **kwargs):
        n_samples = kwargs.get('n_samples') or self.n_samples
        n_features = kwargs.get('n_features') or self.n_features

        X = np.random.random((n_samples, n_features))
        y = np.random.randint(0, 2, n_samples)
        return X, y

    def make_data(self, style: str, *args, **kwargs):
        if hasattr(self, f"make_{style}"):
            method = getattr(self, f"make_{style}")
            return method(**kwargs)
        else:
            raise ValueError(f"Unknown data generation style: {style}")
