from docx import Document
import pdfplumber
import pandas as pd
import numpy as np
from typing import Dict, Any

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
    
    return df