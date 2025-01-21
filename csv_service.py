import numpy as np
import pandas as pd

def clean_data(csv_url):
    data = pd.read_csv(csv_url)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    try:
        # Remove duplicate rows
        data = data.drop_duplicates()

        # Strip whitespace from string columns
        for column in data.select_dtypes(include=['object']).columns:
            data[column] = data[column].str.strip()

        # Replace infinite values with NaN
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill NaN values based on column data types
        for column in data.columns:
            if data[column].dtype == 'object':  # String type
                data[column] = data[column].fillna('')
            elif data[column].dtype == 'float64':  # Float type
                data[column] = data[column].fillna(0.0)
            elif data[column].dtype == 'int64':  # Integer type
                data[column] = data[column].fillna(0)
            elif data[column].dtype == 'bool':  # Boolean type
                data[column] = data[column].fillna(False)
            elif data[column].dtype == 'datetime64[ns]':  # Datetime type
                data[column] = data[column].fillna(pd.NaT)
            elif data[column].dtype == 'timedelta64[ns]':  # Timedelta type
                data[column] = data[column].fillna(pd.Timedelta(0))
            elif data[column].dtype.name == 'category':  # Categorical type
                data[column] = data[column].fillna(data[column].cat.categories[0] if len(data[column].cat.categories) > 0 else None)
            elif data[column].dtype == 'complex128':  # Complex number type
                data[column] = data[column].fillna(complex(0, 0))
            else:  # For other types, default to None
                data[column] = data[column].fillna(None)

        # Remove constant columns (columns with only one unique value)
        constant_columns = [col for col in data.columns if data[col].nunique() <= 1]
        data = data.drop(columns=constant_columns)
        #print(f"Data cleaning complete. Removed {len(constant_columns)} constant columns and duplicates.")
        return data
    
    except Exception as e:
        #print(f"Error occurred during data cleaning: {e}")
        raise e

