
__all__ = ['hi', 'DataManager']

def hi():
    print('hi in datamanager')

class DataManager:
    def __init__(self, working_directory):
        self.working_directory = working_directory
        print('init data manager')

    def load_data(self, file_path=None):
        """
        Load data from file_path or prompt user for data.
        """
        # Implementation here
    
    def clean_data(self, data):
        """
        Clean the loaded data.
        """
        # Implementation here
    
    def save_processed_data(self, data, name):
        """
        Save the processed data with a specified name.
        """
        # Implementation here
