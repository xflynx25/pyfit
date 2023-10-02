
__all__ = ['DataManager']
import os
import pandas as pd
from .datagen import DataGen

# think, train test as seperate csv, or as a column in the total database? 
# probably, seperate, because you might have diff features and stuff
# and maybe one model to predict many things, so 
class DataManager:
    def __init__(self, working_directory, project_name = "default"):
        self.working_directory = working_directory
        self.datasets_directory = os.path.join(working_directory, 'datasets')
        self._init_directories()
        self.project_name = project_name

        self.raw_data_path = None 
        self.active_data_path = None  # Initialize with no active dataset
        self.active_dataset = None 

        print('init data manager')

    def _init_directories(self):
        if not os.path.exists(self.datasets_directory):
            os.makedirs(self.datasets_directory)

    def set_active_dataset(self, dataset_name, dataset_type='raw'):
        """
        Set the active dataset by specifying its name and type.
        """
        valid_dataset_types = ['raw', 'processed']
        if dataset_type not in valid_dataset_types:
            raise ValueError(f"Invalid dataset type. Expected one of: {valid_dataset_types}")

        file_path = os.path.join(self.datasets_directory, f"db_{dataset_name}_{dataset_type}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{dataset_type.capitalize()} dataset '{dataset_name}' not found in {self.datasets_directory}")

        self.active_data_path = file_path

    def load_active_dataset(self):
        """
        Load the dataset that is set as active.
        """
        if self.active_data_path is None:
            raise ValueError("No active dataset set. Set an active dataset first.")
        self.active_dataset = pd.read_csv(self.active_data_path)
        return self.active_dataset
    def get_data_head(self, n=5):
        """
        Get the head of the active dataset.
        """
        if self.active_dataset is None:
            raise ValueError("No active dataset. Load a dataset first.")
        return self.active_dataset.head(n)
    
    def generate_data(self, dataset_name, **kwargs):
        """
        Generate and save synthetic data using DataGen.
        Sets the active dataset as the raw data, as is only one before transformations
        """
        data_gen = DataGen()
        X, y = data_gen.make_data(dataset_name, **kwargs) #calls if relevant
        df = pd.DataFrame(X)
        df['target'] = y #shouldn't do this, rather, should have a malleable attribute that says what the target column is
        self.active_data_path = os.path.join(self.datasets_directory, f"db_{self.project_name}_raw.csv")
        df.to_csv(self.active_data_path, index=False)

    def save_processed_data(self, data, name):
        """
        Save the processed data with a specified name.
        """
        # Implementation here
