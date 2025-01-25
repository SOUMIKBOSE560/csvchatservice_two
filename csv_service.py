import numpy as np
import pandas as pd
from fastapi.encoders import jsonable_encoder
from typing import Any
from datetime import datetime, date, time, timedelta





from fastapi.encoders import jsonable_encoder
import numpy as np
import pandas as pd
from typing import Any
from datetime import datetime, date, time, timedelta

def safe_jsonable_encoder(obj: Any) -> Any:
    """
    A custom JSON encoder that handles all NumPy types, Pandas DataFrames, and other non-standard types.
    """
    # Handle NumPy types
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.datetime64, np.timedelta64)):
        return str(obj)  # Convert datetime64 and timedelta64 to string
    elif isinstance(obj, (np.ndarray)):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, (np.void)):  # Handle structured arrays
        if hasattr(obj.dtype, 'names') and obj.dtype.names:
            return {key: safe_jsonable_encoder(obj[key]) for key in obj.dtype.names}
        else:
            return str(obj)  # Fallback for unstructured void
    elif isinstance(obj, (np.generic)):  # Handle generic NumPy types
        return obj.item()  # Convert to Python scalar

    # Handle Pandas types
    elif isinstance(obj, (pd.DataFrame)):
        return obj.to_dict(orient="records")  # Convert DataFrame to list of dictionaries
    elif isinstance(obj, (pd.Series)):
        return obj.tolist()  # Convert Series to list
    elif isinstance(obj, (pd.Index)):
        return obj.tolist()  # Convert Index to list
    elif isinstance(obj, (pd.Timestamp)):
        return obj.isoformat()  # Convert Timestamp to ISO format string
    elif isinstance(obj, (pd.Timedelta)):
        return str(obj)  # Convert Timedelta to string

    # Handle Python datetime types
    elif isinstance(obj, (datetime, date, time, timedelta)):
        return obj.isoformat()  # Convert datetime objects to ISO format string

    # Handle objects with __dict__ attribute
    elif hasattr(obj, "__dict__"):
        return {key: safe_jsonable_encoder(value) for key, value in vars(obj).items()}

    # Handle iterables (lists, tuples, sets)
    elif isinstance(obj, (list, tuple)):
        return [safe_jsonable_encoder(item) for item in obj]
    elif isinstance(obj, (set)):
        return [safe_jsonable_encoder(item) for item in obj]  # Convert set to list

    # Handle dictionaries
    elif isinstance(obj, dict):
        return {key: safe_jsonable_encoder(value) for key, value in obj.items()}

    # Handle bytes and bytearray
    elif isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="ignore")  # Convert bytes to string

    # Handle pd.NA and np.nan
    elif pd.isna(obj) or (isinstance(obj, (np.ndarray)) and np.isnan(obj).all()):
        return None

    # Fallback to FastAPI's jsonable_encoder
    else:
        try:
            return jsonable_encoder(obj)
        except TypeError as e:
            print(f"Failed to serialize object of type {type(obj)}: {e}")
            return str(obj)  # If all else fails, convert to string
        
        
        
        
        
        
        
        
        

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

