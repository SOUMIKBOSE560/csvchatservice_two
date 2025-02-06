import numpy as np
import pandas as pd
from fastapi.encoders import jsonable_encoder
from typing import Any
from datetime import datetime, date, time, timedelta
from smolagents import Tool


# Generic JSON encoder to handle all numpy types
def custom_jsonable_encoder(obj):
    # Handle numpy scalar types
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.str_)):
        return str(obj)
    # Handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle numpy structured arrays and record arrays
    elif isinstance(obj, (np.void, np.record)):
        return {key: custom_jsonable_encoder(obj[key]) for key in obj.dtype.names}
    # Handle other non-serializable objects
    else:
        # Fall back to the default jsonable_encoder for other types
        return jsonable_encoder(obj)
    
    
def get_csv_basic_info(csv_path):
    """
    Get basic information about a CSV file including:
    - Row count
    - Column count
    - Column names
    - First two rows
    
    Parameters:
    csv_path (str): Path to the CSV file
    
    Returns:
    dict: Dictionary containing basic file information or error message
    """
    try:
        # Read the CSV file
        df = clean_data(csv_path)
        
        print(f"CSV file read successfully: {csv_path}")
        
        return {
            'row_count': df.shape[0],
            'col_count': df.shape[1],
            'col_names': df.columns.tolist(),
            'first_two_rows': df.head(2).to_dict('records'),
            'error': None
        }
    except Exception as e:
        return {
            'error': f"Error reading CSV file: {str(e)}",
        }
        



def safe_jsonable_encoder(obj: Any) -> Any:
    """
    Handles all NumPy/Pandas types, custom objects, and edge cases without errors.
    """
    # Handle None and basic types first
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # ----------------------------------
    # Handle NumPy types
    # ----------------------------------
    # Numpy scalars
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, np.timedelta64):
        return str(obj)
    
    # Numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Numpy dtypes (e.g., np.dtype('int64'))
    if isinstance(obj, np.dtype):
        return str(obj)
    
    # Numpy void (structured arrays)
    if isinstance(obj, np.void):
        if hasattr(obj.dtype, 'names') and obj.dtype.names:
            return {key: safe_jsonable_encoder(obj[key]) for key in obj.dtype.names}
        return None
    
    # Generic numpy types (fallback)
    if isinstance(obj, np.generic):
        return obj.item()

    # ----------------------------------
    # Handle Pandas types
    # ----------------------------------
    # Pandas DataFrame/Series/Index
    if isinstance(obj, pd.DataFrame):
        return obj.replace({np.nan: None}).to_dict(orient="records")
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.replace({np.nan: None}).tolist()
    
    # Pandas Timestamp/Timedelta
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    
    # Pandas nullable types (e.g., Int64Dtype, pd.NA)
    if isinstance(obj, pd.api.extensions.ExtensionArray):
        return obj.astype(object).tolist()
    if pd.isna(obj):  # Handles pd.NA, np.nan, etc.
        return None

    # ----------------------------------
    # Handle Python types
    # ----------------------------------
    # Datetime objects
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, timedelta):
        return str(obj)
    
    # Bytes/bytearray
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="ignore")
    
    # Iterables (lists, tuples, sets)
    if isinstance(obj, (list, tuple, set)):
        return [safe_jsonable_encoder(item) for item in obj]
    
    # Dictionaries
    if isinstance(obj, dict):
        return {k: safe_jsonable_encoder(v) for k, v in obj.items()}

    # ----------------------------------
    # Handle custom objects
    # ----------------------------------
    # Objects with __dict__ (but skip modules/classes)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        try:
            return {k: safe_jsonable_encoder(v) for k, v in vars(obj).items()}
        except TypeError:
            pass  # Fall through to other checks
    
    # ----------------------------------
    # Final fallbacks
    # ----------------------------------
    try:
        # FastAPI's default encoder
        return jsonable_encoder(obj)
    except TypeError:
        # Convert to string as last resort
        return str(obj)
        
        
        
        
        
        

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



########################################################################################################################################


class CleanDataTool(Tool):
    name = "clean_data"
    description = """Cleans a dataset from a provided CSV URL. The cleaning process includes:
    - Removing duplicate rows.
    - Stripping whitespace from string columns.
    - Replacing infinite values with NaN.
    - Filling NaN values based on column data types.
    - Removing constant columns (columns with only one unique value)."""
    inputs = {
        "csv_url": {"type": "string", "description": "The URL of the CSV file to clean."}
    }
    output_type = "any"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            raise ImportError(
                "You must install packages `pandas` and `numpy` to run this tool: for instance run `pip install pandas numpy`."
            )

    def clean_data(self, csv_url):
        import pandas as pd
        import numpy as np
        
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
            return data
        
        except Exception as e:
            raise e

    def forward(self, csv_url):
        return self.clean_data(csv_url)
