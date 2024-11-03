import os
import fnmatch
import cv2  # OpenCV for video processing
from PIL import Image  # For working with PIL Images
import numpy as np  # For array manipulation
import math
import time
from functools import wraps
import pandas as pd
import ast
import re
import random
from scipy.stats import spearmanr


def merge_model_human_responses(model_csv, human_csv):
    """
    Merges model and human responses, calculating the mean and storing individual response columns for each video.

    Parameters:
    - model_csv: Path to the model CSV file.
    - human_csv: Path to the human responses CSV file.

    Returns:
    - df: Updated DataFrame with new columns 'mean_human_score_<question>' and individual 'human_<n>' columns.
    """
    
    # Load the input DataFrames
    df = pd.read_csv(model_csv)
    df_h = pd.read_csv(human_csv)

    # Get question name from the model CSV
    question_name = df['question_name'].iloc[0]

    # Define new column names
    mean_human_score_column = f'mean_human_score_{question_name}'

    # Initialize new column for mean human scores in the model DataFrame
    df[mean_human_score_column] = np.nan

    # Initialize a dictionary to keep track of the maximum number of responses
    max_responses = 0

    # Iterate through unique video names to calculate responses and mean
    for video in df['video_name'].unique():
        # Extract responses specific to the current video and question
        responses = df_h[(df_h['question_name'] == question_name) & 
                         (df_h['video_name'] == os.path.basename(video))]['likert_response'].tolist()

        # Convert responses to 0-1 scale (from 1-5)
        responses = [(x-1) / 4 for x in responses]

        # Calculate the mean if there are responses
        if responses:
            df.loc[df['video_name'] == video, mean_human_score_column] = np.mean(responses)
        
        # Update maximum number of responses if this video has more responses
        max_responses = max(max_responses, len(responses))

        # Add individual human response columns to the DataFrame
        for i, response in enumerate(responses):
            df.loc[df['video_name'] == video, f'virtual_human_{i+1}'] = response

    # Add empty columns for any human response column not yet in the DataFrame (up to max_responses)
    for i in range(1, max_responses + 1):
        if f'virtual_human_{i}' not in df.columns:
            df[f'virtual_human_{i}'] = np.nan

    # Write the updated DataFrame to the model CSV file
    df.to_csv(model_csv, index=False)

    return df


def human_model_comparison(model_csv):
    """
    Loads the model CSV, identifies relevant columns, computes the Spearman correlation matrix,
    and calculates mean and standard deviation for model-to-human and human-to-human correlations,
    including the correlation between the model score and mean human score, ignoring missing values.

    Parameters:
    - model_csv: Path to the model CSV file.

    Returns:
    - correlation_matrix: DataFrame containing the Spearman correlation matrix.
    - model_to_human_stats: Tuple containing mean and standard deviation of model-to-human correlations.
    - human_to_human_stats: Tuple containing mean and standard deviation of human-to-human correlations.
    - model_mean_human_corr: Spearman correlation between model score and mean human score.
    """
    
    # Load the input DataFrame
    df = pd.read_csv(model_csv)

    # Extract question name to build column names
    question_name = df.loc[0, 'question_name']
    model_score_column = f'model_score_{question_name}'
    mean_human_score_column = f'mean_human_score_{question_name}'

    # List of columns to include in the correlation matrix, up to virtual_human_10
    columns_to_include = [model_score_column, mean_human_score_column]
    for i in range(1, 11):
        virtual_column = f'virtual_human_{i}'
        if virtual_column in df.columns:
            columns_to_include.append(virtual_column)

    # Subset DataFrame to include only the selected columns
    df_subset = df[columns_to_include]

    # Compute the Spearman correlation matrix, ignoring missing values
    correlation_matrix = df_subset.corr(method='spearman', min_periods=1)

    # Calculate model-to-human correlations (excluding mean_human_score)
    model_to_human_correlations = [
        correlation_matrix.loc[model_score_column, f'virtual_human_{i}']
        for i in range(1, 11) if f'virtual_human_{i}' in correlation_matrix.columns
    ]
    model_to_human_mean = np.nanmean(model_to_human_correlations)
    model_to_human_std = np.nanstd(model_to_human_correlations)

    # Calculate human-to-human correlations
    human_columns = [f'virtual_human_{i}' for i in range(1, 11) if f'virtual_human_{i}' in correlation_matrix.columns]
    human_to_human_correlations = correlation_matrix.loc[human_columns, human_columns].values
    human_to_human_mean = np.nanmean(human_to_human_correlations[np.triu_indices_from(human_to_human_correlations, k=1)])
    human_to_human_std = np.nanstd(human_to_human_correlations[np.triu_indices_from(human_to_human_correlations, k=1)])

    # Calculate correlation between model score and mean human score
    model_mean_human_corr = correlation_matrix.loc[model_score_column, mean_human_score_column]

    # Print the statistics with question name
    print(f"Question: {question_name}")
    print(f"Model to Mean Human Correlation: {model_mean_human_corr}")
    print(f"Model to Human Correlations - Mean: {model_to_human_mean}, Std Dev: {model_to_human_std}")
    print(f"Human to Human Correlations - Mean: {human_to_human_mean}, Std Dev: {human_to_human_std}")

    # Create statistics results
    model_to_human_stats = (model_to_human_mean, model_to_human_std)
    human_to_human_stats = (human_to_human_mean, human_to_human_std)
    
    return correlation_matrix, model_to_human_stats, human_to_human_stats, model_mean_human_corr



