import pandas as pd
from fuzzywuzzy import fuzz
import numpy as np
import difflib
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(file_path):
  """
  Loads data from a file into a pandas DataFrame.

  Args:
    file_path: Path to the file.

  Returns:
    A pandas DataFrame containing the loaded data.
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
    return df
  except Exception as e:
    print(f"Error loading data: {e}")
    return None

def profile_data(df):
  """
  Profiles a pandas DataFrame.

  Args:
    df: The DataFrame to profile.

  Returns:
    A dictionary containing the profile information.
  """

  profile = {}

  # Basic Information
  profile['shape'] = df.shape
  profile['columns'] = df.columns.tolist()

  # Data Types
  profile['dtypes'] = df.dtypes

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

  return profile

def handle_missing_values(file_path, strategy='mean'):
  """
  Handles missing values in a CSV file.

  Args:
    file_path: Path to the CSV file.
    strategy: Strategy to handle missing values. Can be 'mean', 'median', 'mode', or 'ffill'.

  Returns:
    A DataFrame with missing values handled.
  """

  df = pd.read_csv(file_path)

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

  return df


def handle_outliers(df, method='z-score', threshold=3):
  """
  Handles outliers in a DataFrame.

  Args:
    df: The DataFrame to process.
    method: The method to use for outlier detection. Can be 'z-score' or 'iqr'.
    threshold: The threshold for outlier detection.

  Returns:
    A DataFrame with outliers handled.
  """

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

def normalize_data(df, method='standard'):
  """
  Normalizes numerical features in a DataFrame.

  Args:
    df: The DataFrame to process.
    method: The normalization method to use. Can be 'standard' or 'min-max'.

  Returns:
    A DataFrame with normalized numerical features.
  """

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


def reduce_noise(df, method='rolling_mean', window_size=5):
  """
  Reduces noise in a DataFrame.

  Args:
    df: The DataFrame to process.
    method: The method to use for noise reduction. Can be 'rolling_mean' or 'median_filter'.
    window_size: The size of the window for the rolling mean or median filter.

  Returns:
    A DataFrame with reduced noise.
  """

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
    df = pd.read_csv(file_path)
    dataframes.append(df)

  # Merge DataFrames based on a common key (adjust as needed)
  merged_df = pd.concat(dataframes, ignore_index=True)

  return merged_df



def resolve_entity_conflicts(df1, df2, key_columns=['name', 'address']):
  """
  Resolves entity conflicts between two DataFrames.

  Args:
    df1: The first DataFrame.
    df2: The second DataFrame.
    key_columns: A list of columns to use for matching entities.

  Returns:
    A merged DataFrame with resolved entity conflicts.
  """

  merged_df = pd.concat([df1, df2], ignore_index=True)

  # Deduplication using fuzzy matching
  def fuzzy_match(row1, row2):
    score = 0
    for col in key_columns:
      score += fuzz.token_sort_ratio(str(row1[col]), str(row2[col]))
    return score

  def group_similar_records(df):
    groups = []
    for i, row1 in df.iterrows():
      group = [i]
      for j, row2 in df.iterrows():
        if i != j and fuzzy_match(row1, row2) > 90:
          group.append(j)
      groups.append(group)
    return groups

  groups = group_similar_records(merged_df)

  # Merge similar records within each group
  def merge_records(df, group):
    merged_row = df.loc[group].mode().iloc[0]
    for col in key_columns:
      if merged_row[col] == merged_row[col]:  # Check for NaN values
        continue
      unique_values = df.loc[group, col].unique()
      if len(unique_values) > 1:
        # Handle conflicts, e.g., prioritize certain sources or use majority voting
        merged_row[col] = choose_best_value(unique_values)
    return merged_row

  merged_df = merged_df.groupby(groups).apply(merge_records)

  return merged_df



def resolve_entity_conflicts_difflib(df1, df2, key_columns=['name', 'address']):
  """
  Resolves entity conflicts between two DataFrames using difflib.

  Args:
    df1: The first DataFrame.
    df2: The second DataFrame.
    key_columns: A list of columns to use for matching entities.

  Returns:
    A merged DataFrame with resolved entity conflicts.
  """

  merged_df = pd.concat([df1, df2], ignore_index=True)

  def fuzzy_match_difflib(row1, row2):
    score = 0
    for col in key_columns:
      score += difflib.SequenceMatcher(None, str(row1[col]), str(row2[col])).ratio() * 100
    return score
  
def encode_categorical_data(df, encoding_type='one-hot'):
  """
  Encodes categorical data in a DataFrame.

  Args:
    df: The DataFrame to process.
    encoding_type: The encoding type to use. Can be 'one-hot' or 'label'.

  Returns:
    A DataFrame with encoded categorical data.
  """

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

def check_data_quality(df):
  """
  Checks the quality of a DataFrame.

  Args:
    df: The DataFrame to check.

  Returns:
    A dictionary containing information about data quality issues.
  """

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

def create_new_features(df, target_column):
  """
  Creates new features from existing features in a DataFrame.

  Args:
    df: The DataFrame to process.
    target_column: The target column for supervised learning tasks.

  Returns:
    A DataFrame with new features.
  """

  # Numerical Features
  # - Polynomial Features:
  from sklearn.preprocessing import PolynomialFeatures
  poly = PolynomialFeatures(degree=2, interaction_only=True)
  df_poly = poly.fit_transform(df.select_dtypes(include=['number']))
  df_poly = pd.DataFrame(df_poly, columns=poly.get_feature_names_out())
  df = pd.concat([df, df_poly], axis=1)

  # - Interaction Terms:
  df['interaction_term'] = df['feature1'] * df['feature2']

  # Categorical Features
  # - One-Hot Encoding:
  df = pd.get_dummies(df, columns=['categorical_feature'])

  # Time Series Features
  # - Time-Based Features:
  df['year'] = df['timestamp'].dt.year
  df['month'] = df['timestamp'].dt.month
  df['day'] = df['timestamp'].dt.day
  df['hour'] = df['timestamp'].dt.hour


  # - Lag Features:
  df['lag_1'] = df[target_column].shift(1)

  # - Rolling Features:
  df['rolling_mean_7'] = df[target_column].rolling(window=7).mean()
  df['rolling_std_7'] = df[target_column].rolling(window=7).std()

  return df

def check_data_quality(df):
  """
  Checks the quality of a DataFrame.

  Args:
    df: The DataFrame to check.

  Returns:
    A dictionary containing information about data quality issues.
  """

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
    invalid_values = df[col].isin(valid_values).sum()
    quality_report[f'invalid_values_{col}'] = invalid_values

  return quality_report