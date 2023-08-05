import numpy as np
import pandas as pd
from sklearn import preprocessing

__all__ = ['normalize', 'one_hot_encode', 'handle_missing_values']

def normalize(df: pd.DataFrame, columns: list):
    """
    Normalize specified columns in a DataFrame.
    """
    df_copy = df.copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    for column in columns:
        df_copy[column] = min_max_scaler.fit_transform(df_copy[column].values.reshape(-1, 1))
    return df_copy


def one_hot_encode(df: pd.DataFrame, columns: list):
    """
    One hot encode specified columns in a DataFrame.
    """
    df_copy = df.copy()
    for column in columns:
        dummies = pd.get_dummies(df_copy[column], prefix=column)
        df_copy = pd.concat([df_copy, dummies], axis=1)
        df_copy = df_copy.drop(column, axis=1)
    return df_copy


def handle_missing_values(df: pd.DataFrame, strategy='mean', fill_value=None):
    """
    Handle missing values in a DataFrame. Strategy could be 'mean', 'median', 'mode' or 'constant'.
    If strategy is 'constant', fill_value will be used to fill missing values.
    """
    df_copy = df.copy()
    if strategy == 'mean':
        df_copy.fillna(df.mean(), inplace=True)
    elif strategy == 'median':
        df_copy.fillna(df.median(), inplace=True)
    elif strategy == 'mode':
        df_copy.fillna(df.mode().iloc[0], inplace=True)
    elif strategy == 'constant' and fill_value is not None:
        df_copy.fillna(fill_value, inplace=True)
    else:
        raise ValueError("Invalid strategy. Strategy should be 'mean', 'median', 'mode' or 'constant'")
    return df_copy
