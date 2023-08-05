import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from time import time

from pyfit.preprocessing import Preprocessing

__all__ = ['csv_to_linreg']

def csv_to_linreg(file_or_df, dependent_var_column, normalize_columns=None, missing_values_strategy='mean', plot=False):
    """
    Pipeline that takes in a CSV or DataFrame, a dependent variable column, and returns the results of a linear regression.

    Parameters:
    - file_or_df: Either a string path to a CSV file, or a Pandas DataFrame.
    - dependent_var_column: The column of the DataFrame that the model should predict.
    - normalize_columns: List of columns to normalize. If None, all columns are normalized.
    - missing_values_strategy: Strategy to handle missing values. Could be 'mean', 'median', 'mode' or 'constant'.
    - plot: Whether or not to plot the results. Defaults to False.
    """
    if isinstance(file_or_df, str):
        df = pd.read_csv(file_or_df)
    elif isinstance(file_or_df, pd.DataFrame):
        df = file_or_df
    else:
        raise ValueError("file_or_df must be either a string path to a CSV file, or a Pandas DataFrame.")

    # Handle missing values
    df = Preprocessing.handle_missing_values(df, strategy=missing_values_strategy)

    # Normalize columns
    if normalize_columns is None:
        normalize_columns = df.columns
    df = Preprocessing.normalize(df, normalize_columns)

    # Create training and testing sets
    X = df.drop(dependent_var_column, axis=1)
    y = df[dependent_var_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    start = time()
    model.fit(X_train, y_train)
    end = time()
    train_time = end - start

    # Test the model
    start = time()
    y_pred = model.predict(X_test)
    end = time()
    predict_time = end - start

    # Calculate statistics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # If plot is True, generate a scatter plot
    if plot:
        plt.scatter(X_test, y_test, color='black')
        plt.plot(X_test, y_pred, color='blue', linewidth=3)
        plt.title('Linear Regression Result')
        plt.show()

    return {
        'model': model,
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'mse': mse,
        'r2': r2,
        'train_time': train_time,
        'predict_time': predict_time,
    }
