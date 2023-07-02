import numpy as np 

__all__ = ['Gen']

# generate dataset for practicing machine learning tasks for different types of hoped for data
class Gen():
    def __init__(self) -> None:
        self.data = 1

    # first dim of dimensions is number of samples. Rest is size of the data
    def make_data(self, style: str, dimensions: np.array):
        if style == 'random': 
            self.data = [[1,2],[2,3]]
        else:
            self.data = "something nice"

    def __str__(self) -> str:
        return str(self.data)