from docx import Document
import pdfplumber
import pandas as pd
import numpy as np
from typing import Dict, Any
from timeit import default_timer

def text_extraction(
    file_path : str
) -> str:
    """
    Extract text content from PDF or DOCX files.
    
    This function reads the content of a file and returns the extracted text as a string.
    It supports both PDF files (using pdfplumber) and DOCX files (using python-docx).
    
    Args:
        file_path (str): Path to the file to be processed. Must end with '.pdf' or '.docx'.
        
    Returns:
        str: The extracted text content from the file, with leading/trailing whitespace removed.
        
    Raises:
        TypeError: If the file extension is not '.pdf' or '.docx'.
        FileNotFoundError: If the specified file path does not exist.
        PermissionError: If the file cannot be accessed due to insufficient permissions.

    """
    if file_path.endswith("pdf"):
        with pdfplumber.open(file_path) as pdf:
            text = " ".join([p.extract_text() for p in pdf.pages])
            return text.strip()
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text.strip()
    elif file_path.endswith(".txt"):
        with open(file = file_path, mode = "r") as file:
            return file.read()
    else:
        raise TypeError("the file extension must be either '.pdf', '.docx', or '.txt'")

def print_execution_status(func):
    """wrapper function that allows to keep track of the progress"""
    def wrapper(*args, **kargs):
        print(f"executing function: '{func.__name__}'")
        result = func(*args, **kargs)
        print(f"'{func.__name__}' was successfully executed!")
        print("-"*30)
        return result
    return wrapper

def execution_time(func):
    def wrapper(*args, **kargs):
        start_time = default_timer()
        result = func(*args, **kargs)
        end_time = default_timer()
        time_elapsed = end_time - start_time
        print(f"'{func.__name__}' took {time_elapsed:.3f} seconds")
        return result
    return wrapper


def scores_to_dataframe(candidates_scores: Dict[str, Dict[str, int]], n_groups: int) -> pd.DataFrame:
    """
    Converts a nested dictionary of candidate scores into a pandas DataFrame and
    randomly assigns candidates to groups of approximately equal size.
    
    Args:
        candidates_scores: A dictionary with candidate IDs as keys and their scores as nested dictionaries.
        n_groups: The desired number of groups to create.
        
    Returns:
        A pandas DataFrame with scores and a new 'group_id' column.
    """
    if not candidates_scores:
        return pd.DataFrame()

    data_list = []
    for candidate_id, scores_dict in candidates_scores.items():
        row_data = {'candidate_id': candidate_id}
        row_data.update(scores_dict)
        data_list.append(row_data)

    df = pd.DataFrame(data_list)
    
    # Randomly assign candidates to groups
    num_candidates = len(df)
    group_ids = np.repeat(np.arange(n_groups), num_candidates // n_groups)
    
    # Handle the remainder of candidates by adding them to the end
    remainder = num_candidates % n_groups
    group_ids = np.concatenate([group_ids, np.arange(remainder)])
    
    # Shuffle the group IDs to ensure random assignment
    np.random.shuffle(group_ids)
    
    # Assign the shuffled group IDs to the DataFrame
    df['group_id'] = group_ids
    
    # Set the group_id as index as well 
    df.set_index(keys = ['candidate_id', 'group_id'], inplace = True)
    
    return df

def auxiliary_col_calutation(
    df : pd.DataFrame,
    columns : list,
    weight : float,
    output_col_name : str
):
    """
    Calculate auxiliary columns for a DataFrame based on a subset of columns.

    This function creates a copy of the input DataFrame and calculates three auxiliary columns:
    1. A count of NaN values for each row across the specified subset of columns.
    2. A sum of non-NaN values for each row across the specified subset of columns.
    3. A weighted average of the non-NaN values for each row across the specified subset of columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): A list of column names to be used for calculations.
    - weight (float): The weight to apply in the weighted average calculation.
    - output_col_name (str): The base name for the output columns. The function will create
      columns named '<output_col_name>_na_count', '<output_col_name>_sum', and '<output_col_name>_average'.

    Returns:
    - pd.DataFrame: A copy of the input DataFrame with the additional auxiliary columns.

    Notes:
    - If all values in the specified columns are NaN for a row, the weighted average will be set to NaN.
    - The function does not modify the original DataFrame; it operates on a copy.
    """
    # create a copy of the dataframe
    df_copy = df.copy()
    # counts the number of na values for each row over a set of columns
    df_copy[f"{output_col_name}_na_count"] = df_copy[columns].isna().sum(axis = 1)
    # adds the rows of a subset of columns if the values are not na
    df_copy[f"{output_col_name}_sum"] = df_copy[columns].sum(axis = 1, skipna = True)
    # calculates the weighted average of the subset of columns for the non-na values
    df_copy[f"{output_col_name}_average"] = np.where(
        df_copy[f"{output_col_name}_na_count"] == len(columns), # in case all the columns are nan
        np.nan, # if all the nan then the result should be nan
        df_copy[f"{output_col_name}_sum"] * (weight / (len(columns) - df_copy[f"{output_col_name}_na_count"]))  
    )
    # some of the axiliary columns can be dropped after the calculation
    df_copy.drop(columns = [f"{output_col_name}_na_count", f"{output_col_name}_sum"], inplace = True)
    return df_copy

def refined_overall_calculation(
    df : pd.DataFrame,
    columns : list,
    output_col_name : str
):
    """
    Calculate a refined overall score for each row in a DataFrame, adjusting for missing (NaN) values.

    This function creates a copy of the input DataFrame and computes a new column named
    '<output_col_name>_refined', which represents the sum of the specified columns for each row,
    scaled to account for any missing values. If some columns are NaN for a row, the sum is
    proportionally scaled up as if all columns were present. If all columns are present, the sum
    is unchanged.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): List of column names to include in the calculation.
    - output_col_name (str): Base name for the output column. The function will create a column
      named '<output_col_name>_refined'.

    Returns:
    - pd.DataFrame: A copy of the input DataFrame with the additional refined score column.

    Notes:
    - If all values in the specified columns are NaN for a row, the result will be NaN.
    - The function does not modify the original DataFrame.
    """
    # create a copy of the data
    df_copy = df.copy()
    # calculate potential na values
    df_copy[f"{output_col_name}_na_count"] = df_copy[columns].isna().sum(axis = 1)
    # calculate the sum of the non-na values
    df_copy[f"{output_col_name}_sum"] = df_copy[columns].sum(axis = 1, skipna = True)
    # scale the sum based on the na values
    df_copy[f"{output_col_name}_refined"] = np.where(
        df_copy[f"{output_col_name}_na_count"] == len(columns),
        np.nan,
        np.where(
            df_copy[f"{output_col_name}_na_count"] > 0, 
            df_copy[f"{output_col_name}_sum"] * (len(columns) / (len(columns) - df_copy[f"{output_col_name}_na_count"])),
            df_copy[f"{output_col_name}_sum"]
        )
    )
    df_copy.drop(columns = [f"{output_col_name}_na_count", f"{output_col_name}_sum"], inplace = True)
    return df_copy