import pandas as pd
from fuzzywuzzy import fuzz
import numpy as np
import difflib
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import re

def load_data(file_path:str):
  """
  Loads data from a file into a pandas DataFrame.

  Args:
    file_path: Path to the file.

    Returns:
        A pandas DataFrame containing the loaded data.

    Raises:
        ValueError: If file is empty or has unsupported format
        FileNotFoundError: If file does not exist
    """
  try:
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else: 
        raise ValueError("Unsupported file format. Please provide a CSV, Excel, or JSON file.")
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("File is empty")
        
    return df
        
  except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}")
  except Exception as e:
        raise ValueError(f"Error loading data: {e}")

def profile_data(file_path:str):
  """
  Profiles a pandas DataFrame.

  Args:
    file_path: The path to the file that contains DataFrame.

  Returns:
    A dictionary containing the profile information.
  """
  df=load_data(file_path)
  profile = {}

  # Basic Information
  profile['shape'] = df.shape
  profile['columns'] = df.columns.tolist()

  # Data Types
  profile['dtypes'] = df.dtypes.to_dict()

  # Numerical Features
  numerical_cols = df.select_dtypes(include=['number']).columns
  profile['numerical_features'] = numerical_cols
  for col in numerical_cols:
    profile[f'stats_{col}'] = df[col].describe().to_dict()

  # Categorical Features
  categorical_cols = df.select_dtypes(include=['object']).columns
  profile['categorical_features'] = categorical_cols
  for col in categorical_cols:
    profile[f'unique_values_{col}'] = df[col].nunique()
    profile[f'value_counts_{col}'] = df[col].value_counts().to_dict()

  # Missing Values
  profile['missing_values'] = df.isnull().sum().to_dict()
  profile['summary_stats'] = df.describe().to_dict()
  return profile

def handle_missing_values(file_path, strategy='mean'):
    """
    Handle missing values in a CSV file.

    Args:
        file_path: Path to the CSV file.
        strategy: Strategy to handle missing values. Can be 'mean', 'median', 'mode', or 'ffill'.

    Returns:
        tuple: (new file path, data profile dictionary)
    """
    df = load_data(file_path)
    
    if strategy == 'mean':
        df.fillna(df.mean(), inplace=True)
    elif strategy == 'median':
        df.fillna(df.median(), inplace=True)
    elif strategy == 'mode':
        df.fillna(df.mode().iloc[0], inplace=True)
    elif strategy == 'ffill':
        df.fillna(method='ffill', inplace=True)
    else:
        raise ValueError("Invalid strategy. Please choose from 'mean', 'median', 'mode', or 'ffill'.")

    # Save processed data
    new_file_path = save_processed_df(file_path, df, 'handle_missing_values')
    
    # Get data profile
    profile = profile_data(new_file_path)
    
    return new_file_path, profile

def handle_outliers(file_path:str, method:str='z-score', threshold:float=3.0):
  """
  Handles outliers in a DataFrame.

  Args:
    file_path: The path to the file that contains DataFrame.
    method: The method to use for outlier detection. Can be 'z-score' or 'iqr'.
    threshold: The threshold for outlier detection.

  Returns:
    A DataFrame with outliers handled.
  """
  df=load_data(file_path)
  if method == 'z-score':
    z_scores = np.abs((df - df.mean()) / df.std())
    df = df[(z_scores < threshold).all(axis=1)]
  elif method == 'iqr':
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
  else:
    raise ValueError("Invalid method. Please choose from 'z-score' or 'iqr'.")

  return df

def normalize_data(file_path:str, method:str='standard'):
  """
  Normalizes numerical features in a DataFrame.

  Args:
    file_path: The path to the file that contains DataFrame.
    method: The normalization method to use. Can be 'standard' or 'min-max'.

  Returns:
    A DataFrame with normalized numerical features.
  """
  df=load_data(file_path)
  numerical_cols = df.select_dtypes(include=np.number).columns

  if method == 'standard':
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
  elif method == 'min-max':
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
  else:
    raise ValueError("Invalid normalization method. Please choose from 'standard' or 'min-max'.")

  return df


def reduce_noise(file_path:str, method:str='rolling_mean', window_size=5):
  """
  Reduces noise in a DataFrame.

  Args:
    file_path: The path to the file that contains DataFrame.
    method: The method to use for noise reduction. Can be 'rolling_mean' or 'median_filter'.
    window_size: The size of the window for the rolling mean or median filter.

  Returns:
    A DataFrame with reduced noise.
  """
  df=load_data(file_path)
  if method == 'rolling_mean':
    df = df.rolling(window=window_size).mean()
  elif method == 'median_filter':
    df = df.rolling(window=window_size).median()
  else:
    raise ValueError("Invalid method. Please choose from 'rolling_mean' or 'median_filter'.")

  return df

def merge_data_sources(file_paths):
  """
  Merges multiple data sources into a single DataFrame.

  Args:
    file_paths: A list of paths to the data sources.

  Returns:
    A merged DataFrame.
  """

  dataframes = []
  for file_path in file_paths:
    df=load_data(file_path)
    dataframes.append(df)

  # Merge DataFrames based on a common key (adjust as needed)
  merged_df = pd.concat(dataframes, ignore_index=True)

  return merged_df

  
def encode_categorical_data(file_path:str, encoding_type='one-hot'):
  """
  Encodes categorical data in a DataFrame.

  Args:
    file_path: The path to the file that contains DataFrame.
    encoding_type: The encoding type to use. Can be 'one-hot' or 'label'.

  Returns:
    A DataFrame with encoded categorical data.
  """
  df=load_data(file_path)
  if encoding_type == 'one-hot':
    df = pd.get_dummies(df)
  elif encoding_type == 'label':
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
      df[col] = le.fit_transform(df[col])

  else:
    raise ValueError("Invalid encoding type. Please choose from 'one-hot' or 'label'.")

  return df

def check_data_quality(file_path:str):
  """
  Checks the quality of a DataFrame.

  Args:
    file_path: The path to the file that contains DataFrame.

  Returns:
    A dictionary containing information about data quality issues.
  """
  df=load_data(file_path)
  quality_report = {}

  # Check for missing values
  missing_values = df.isnull().sum()
  quality_report['missing_values'] = missing_values

  # Check for duplicate rows
  duplicate_rows = df.duplicated().sum()
  quality_report['duplicate_rows'] = duplicate_rows

  # Check for inconsistent data types
  inconsistent_dtypes = df.dtypes.value_counts()
  quality_report['inconsistent_dtypes'] = inconsistent_dtypes

  # Check for outliers (optional)
  # ... (use methods like z-scores or IQR to identify outliers)

  # Check for data consistency (optional)
  # ... (compare data with external sources or reference data)

  return quality_report


def check_data_quality(file_path:str):
  """
  Checks the quality of a DataFrame.

  Args:
    file_path: The path to the file that contains DataFrame.

  Returns:
    A dictionary containing information about data quality issues.
  """
  df=load_data(file_path)

  quality_report = {}

  # Missing Values
  missing_values = df.isnull().sum()
  quality_report['missing_values'] = missing_values

  # Duplicate Rows
  duplicate_rows = df.duplicated().sum()
  quality_report['duplicate_rows'] = duplicate_rows

  # Inconsistent Data Types
  inconsistent_dtypes = df.dtypes.value_counts()
  quality_report['inconsistent_dtypes'] = inconsistent_dtypes

  # Outliers
  # Assuming numerical columns
  numerical_cols = df.select_dtypes(include=np.number).columns
  for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    quality_report[f'outliers_{col}'] = outliers

  # Data Consistency (Basic)
  # Check for invalid values in categorical columns
  categorical_cols = df.select_dtypes(include=['object']).columns
  for col in categorical_cols:
    valid_values = df[col].unique()
    invalid_values = df[col].isin(valid_values).sum()
    quality_report[f'invalid_values_{col}'] = invalid_values

  return quality_report

def save_preprocessed_df(original_file_path: str, df: pd.DataFrame, preprocess_step: str) -> str:
    """
    Save preprocessed DataFrame to a new file.
    
    Args:
        original_file_path: Original file path in format {path}/{filename}
        df: Preprocessed DataFrame
        preprocess_step: Name of the preprocessing function
    
    Returns:
        str: Path to the new file
    """
    # Split path and filename
    path, filename = os.path.split(original_file_path)
    
    # Get file extension
    base_name, extension = os.path.splitext(filename)
    
    # Check if filename already follows format (xxx-step-name)
    pattern = r"^(\d{3})-([a-zA-Z_]+)-(.+)$"
    match = re.match(pattern, base_name)
    
    if match:
        # If already follows format, extract information
        step_number = int(match.group(1))
        original_name = match.group(3)
        # Increment step number
        new_step_number = str(step_number + 1).zfill(3)
        new_filename = f"{new_step_number}-{preprocess_step}-{original_name}{extension}"
    else:
        # If first processing, start from 000
        new_filename = f"000-{preprocess_step}-{filename}"
    
    # Build complete new path
    new_file_path = os.path.join(path, new_filename)
    
    # Save according to original file format
    if extension.lower() == '.csv':
        df.to_csv(new_file_path, index=False)
    elif extension.lower() == '.xlsx':
        df.to_excel(new_file_path, index=False)
    elif extension.lower() == '.json':
        df.to_json(new_file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")
    
    return new_file_path