import pandas as pd

class DataCleaningInfo:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def get_info(self):
        print(f"\n------- CLEANING INFORMATION --------\n=====================================")
        print(f"Size of Target data (y): {len(self.y)}")
        print(f"Size of Input data (X): {self.X.shape[0]} rows, {self.X.shape[1]} columns\n")

        # Information about the target variable
        target_nulls = self.y.isnull().sum()
        print(f"Total Null counts in Target: {target_nulls}")
        if target_nulls > 0:
            target_null_indices = self.y[self.y.isnull()].index.tolist()
            print(f"Indices with Nulls in Target: {target_null_indices[:10]} (showing first 10 if available)\n")

        # Information about the data
        data_nulls = self.X.isnull().sum()
        total_data_nulls = data_nulls.sum()
        print(f"Total Null counts in Data: {total_data_nulls}")
        
        cols_with_nulls = data_nulls[data_nulls > 0].index.tolist()
        if cols_with_nulls:
            print(f"Columns with Nulls in Data: {cols_with_nulls}")

            # Displaying rows where null values are present (displaying indices of first 10 rows)
            null_data_indices = self.X[self.X.isnull().any(axis=1)].index.tolist()
            print(f"Indices of Rows with Nulls in Data: {null_data_indices[:10]} (showing first 10 if available)")
        
        # Columns without null values
        cols_without_nulls = data_nulls[data_nulls == 0].index.tolist()
        if cols_without_nulls:
            print(f"Columns without Nulls in Data: {cols_without_nulls[:10]} (showing first 10 if available)\n")

        # Duplicate information
        duplicate_data = self.X.duplicated()
        total_duplicates = duplicate_data.sum()
        print(f"Total Duplicate rows in Data: {total_duplicates}")
        if total_duplicates > 0:
            duplicate_indices = duplicate_data[duplicate_data].index.tolist()
            print(f"Indices of Duplicate Rows in Data: {duplicate_indices[:10]} (showing first 10 if available)")

        # Return information as dictionary
        info_dict = {
            'target_size': len(self.y),
            'data_size': (self.X.shape[0], self.X.shape[1]),
            'target_nulls': target_nulls,
            'data_nulls': data_nulls,
            'total_data_nulls': total_data_nulls,
            'cols_with_nulls': cols_with_nulls,
            'cols_without_nulls': cols_without_nulls,
            'duplicate_indices': duplicate_indices if total_duplicates > 0 else []
        }
        return info_dict
