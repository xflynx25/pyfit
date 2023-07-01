import numpy as np 

# generate dataset for practicing machine learning tasks for different types of hoped for data
class Gen():
    def __init__(self) -> None:
        pass

    # first dim of dimensions is number of samples. Rest is size of the data
    def make_data(style: str, dimensions: np.array):
        if style == 'random': 
            return [[1,2],[2,3]]