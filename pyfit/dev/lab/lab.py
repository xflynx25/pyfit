

__all__ = ['Lab']

from ..dataManager import DataManager
from ..eda import EDA
from ..transformer import Transformer
from ..model import ModelInterface
from ..comparator import Comparator


class Lab:
    def __init__(self, working_directory):
        """
        Initialize the Lab with a specified working directory.
        """
        self.working_directory = working_directory


        
        self.data_manager = DataManager(working_directory)
        self.eda = EDA(working_directory)
        self.transformer = Transformer(working_directory)
        self.model_interface = ModelInterface(working_directory)
        self.comparator = Comparator(working_directory)
        print('init lab')
    
    def load_data(self, file_path=None):
        """
        Load data using DataManager. If file_path is None, prompt the user for input.
        """
        # Implementation here
    
    def preprocess_data(self):
        """
        Preprocess data using DataManager.
        """
        # Implementation here
    
    def perform_eda(self):
        """
        Perform exploratory data analysis using EDA object.
        """
        # Implementation here
    
    def transform_data(self):
        """
        Transform data using Transformer object.
        """
        # Implementation here
    
    def train_model(self):
        """
        Train model using ModelInterface object.
        """
        # Implementation here
    
    def compare_models(self):
        """
        Compare models using Comparator object.
        """
        # Implementation here
