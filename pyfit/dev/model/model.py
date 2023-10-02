__all__ = ['ModelInterface'] # will have the subclasses? or maybe init will be different

class ModelInterface:
    def __init__(self, working_directory):
        self.working_directory = working_directory
        print('init model interface')
    
    def train_model(self, model_type, data, features, target):
        """
        Train a specified model type with provided data, features, and target.
        """
        # Implementation here
    
    def evaluate_model(self, model, evaluation_metric):
        """
        Evaluate the trained model using a specified metric.
        """
        # Implementation here
    
    def save_model(self, model, name):
        """
        Save the trained model with a specified name.
        """
        # Implementation here
