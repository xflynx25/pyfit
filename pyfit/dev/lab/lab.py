

__all__ = ['Lab']

import os

from ..dataManager import DataManager
from ..eda import EDA
from ..transformer import Transformer
from ..model import ModelInterface
from ..comparator import Comparator
from .. import dev_helpers


class Lab:
    def __init__(self, working_directory):
        """
        Initialize the Lab with a specified working directory.
        """
        self.working_directory = working_directory
        dev_helpers.init_directories(working_directory)
        
        self.data_manager = DataManager(working_directory)


        self.transformer = Transformer(working_directory)
        self.model_interface = ModelInterface(working_directory)
        self.comparator = Comparator(working_directory)
        print('init lab')

    def load_data(self, file_path=None):
        """
        If file_path is provided, load data from it; otherwise, generate data.
        """
        if file_path:
            # Load data from file_path
            pass
        else:
            # Call self.data_manager.generate_data() with the appropriate parameters
            pass

    def seed_data(self, dataset_name, **kwargs):
        """
        Seed data by generating synthetic data or loading a common dataset, then save it.
        """
        self.data_manager.generate_data(dataset_name, **kwargs)

    def show_data_head(self, n=5):
        """
        Display the head of the active dataset.
        """
        return self.data_manager.get_data_head(n)
    
    
    def preprocess_data(self):
        """
        Preprocess data using DataManager.
        """
        # Implementation here
    
    def perform_eda(self):
        """
        Perform exploratory data analysis using EDA object.
        """
        X = self.data_manager.get_X()
        Y = self.data_manager.get_Y()
        self.eda = EDA(self.working_directory, X, Y)
        self.eda.default_eda()

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
