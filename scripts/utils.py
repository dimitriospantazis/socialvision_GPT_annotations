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


# List of common video file extensions
def find_videos(directory):
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']
    video_files = []
    # Walk through directory and subdirectories
    for root, _, files in os.walk(directory):
        for ext in video_extensions:
            for filename in fnmatch.filter(files, ext):
                video_files.append(os.path.join(root, filename))
    return video_files



def video2images(video_path, num_frames):
    """
    Extracts equispaced frames from video

    Parameters:
    - video_path: The path of the video to search for (e.g., 'my_video.mp4').
    - num_frames: The number of frames

    Returns:
    - PIL images.
    """  
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames and the frame rate (frames per second)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if num_frames > total_frames:
        raise ValueError("Number of frames requested exceeds total frames in the video.")

    # Calculate the interval in frames to extract equispaced frames
    interval = max(1, total_frames // num_frames)

    pil_images = []  # List to store extracted PIL images

    # Iterate through the video, extracting frames at the computed interval
    for i in range(num_frames):
        frame_number = i * interval  # Compute the frame number to extract
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Set position
        ret, frame = cap.read()  # Read the frame

        if ret:
            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the NumPy array to a PIL Image
            pil_image = Image.fromarray(frame_rgb)
            pil_images.append(pil_image)
        else:
            print(f"Could not extract frame at {frame_number}. Skipping...")

    cap.release()  # Release the video capture object
    return pil_images, total_frames, fps



def find_video_path(video_name, search_directory):
    """
    Search for a specific video in a directory and its subdirectories.

    Parameters:
    - video_name: The name of the video to search for (e.g., 'my_video.mp4').
    - search_directory: The path to the directory where the search begins.

    Returns:
    - Full path to the video if found, else None.
    """
    for root, _, files in os.walk(search_directory):
        if video_name in files:
            return os.path.join(root, video_name)  # Return full path

    # If video is not found, return None
    return None


def parse_model_responses(input_csv):
    """
    Loads a CSV file containing a DataFrame with 'model_response*' fields, 
    extracts the score, confidence, people count, and indoor/outdoor information,
    and adds them to corresponding new columns in the DataFrame only if they contain non-blank values.
    Any completely empty columns are removed before saving.
    
    Parameters:
    - input_csv: Path to the input CSV file.
    
    Returns:
    - df: The modified DataFrame with new columns for each extracted field.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Define a function to extract the required fields from the 'model_response*' column
    def extract_fields(response):
        # Remove any extra backslashes from the response string
        response = response.replace('\\', '')

        # Extract the score using regex
        score_match = re.search(r'<score>\s*=\s*(\d+(\.\d+)?)', response)
        score = float(score_match.group(1)) if score_match else None

        # Extract the confidence using regex
        confidence_match = re.search(r'<confidence>\s*=\s*(\d+(\.\d+)?)', response)
        confidence = float(confidence_match.group(1)) if confidence_match else None

        # Extract the people count using regex
        people_count_match = re.search(r'<people_count>\s*=\s*(\d+\??)', response)
        people_count = people_count_match.group(1) if people_count_match else None

        # Extract indoor/outdoor using regex
        indoor_outdoor_match = re.search(r'<indoor_outdoor>\s*=\s*(Indoor|Outdoor)', response)
        indoor_outdoor = indoor_outdoor_match.group(1) if indoor_outdoor_match else None

        # Extract indoor/outdoor confidence using regex
        indoor_outdoor_confidence_match = re.search(r'<indoor_outdoor_confidence>\s*=\s*(\d+(\.\d+)?)', response)
        indoor_outdoor_confidence = float(indoor_outdoor_confidence_match.group(1)) if indoor_outdoor_confidence_match else None

        return score, confidence, people_count, indoor_outdoor, indoor_outdoor_confidence

    # Iterate through all columns to find those that start with 'model_response'
    for col in df.columns:
        if col.startswith('model_response'):
            # Extract the suffix from 'model_response<suffix>'
            suffix = col[len('model_response'):]

            # Define new columns for extracted fields
            score_col = f'model_score{suffix}' if suffix != '_scene_analysis' else 'model_score_scene_analysis'
            confidence_col = f'model_confidence{suffix}' if suffix != '_scene_analysis' else 'model_confidence_scene_analysis'
            people_count_col = f'people_count{suffix}'

            # Apply the extract_fields function to extract values
            extracted_values = df[col].apply(lambda x: pd.Series(extract_fields(x)))
            score, confidence, people_count, indoor_outdoor, indoor_outdoor_confidence = extracted_values.T.values

            # Only add columns if they contain non-blank values
            if not pd.isnull(score).all():
                df[score_col] = score
            if not pd.isnull(confidence).all():
                df[confidence_col] = confidence
            if not pd.isnull(people_count).all():
                df[people_count_col] = people_count
            if not pd.isnull(indoor_outdoor).all():
                df['model_score_scene_analysis'] = indoor_outdoor
            if not pd.isnull(indoor_outdoor_confidence).all():
                df['model_confidence_scene_analysis'] = indoor_outdoor_confidence

    # Remove any columns that are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    # Sort the DataFrame by 'video_name' before saving
    if 'video_name' in df.columns:
        df.sort_values(by='video_name', inplace=True)

    # Optionally, save the modified DataFrame to the same file or a new file
    df.to_csv(input_csv, index=False)  # Uncomment if you want to save the result
    
    return df



def merge_csv_columns(csv_files, output_file):
    """
    Merges specific columns from multiple CSV files based on the 'video_name' column,
    and saves the resulting dataframe to a CSV file. It checks each CSV file for the presence 
    of the required columns and merges all available ones.
    
    Parameters:
    csv_files (list): List of CSV file paths to be merged.
    output_file (str): Path where the merged dataframe will be saved.

    Returns:
    DataFrame: The merged dataframe with selected columns based on 'video_name'.
    """
    # List of columns to merge
    columns_to_merge = [
        'model_score_expanse', 'model_confidence_expanse', 'expanse_videos',
        'model_score_distance', 'model_confidence_distance', 'distance',
        'model_score_arousal', 'model_confidence_arousal', 'arousal',
        'model_score_communicating', 'model_confidence_communicating', 'communicating',
        'model_score_cooperation', 'model_confidence_cooperation', 'cooperation',
        'model_score_dominance', 'model_confidence_dominance', 'dominance',
        'model_score_joint', 'model_confidence_joint', 'joint_videos',
        'model_score_object', 'model_confidence_object', 'object',
        'model_score_valence', 'model_confidence_valence', 'valence', 'model_score_facingness', 'model_confidence_facingness', 
        'model_score_scene_analysis','model_confidence_scene_analysis','people_count_scene_analysis'
    ]

    # Create a list to store the dataframes
    dataframes = []

    # Load each CSV and select the 'video_name' and available specific columns to merge
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # Initialize columns to keep with 'video_name'
        available_columns = ['video_name'] + [col for col in columns_to_merge if col in df.columns]

        # If 'video_name' and at least one of the required columns are present, keep the dataframe
        if len(available_columns) > 1:  # Ensuring there is more than just 'video_name'
            #print(f"Using columns from {csv_file}: {available_columns}")
            dataframes.append(df[available_columns])
        else:
            print(f"Skipping {csv_file}, no relevant columns found.")
    
    # Check if there are any dataframes to merge
    if not dataframes:
        raise ValueError("No relevant columns found in any of the provided CSV files.")

    # Start merging with the first dataframe
    merged_df = dataframes[0]

    # Merge all the other dataframes one by one using 'video_name' as the key
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on='video_name', how='outer')

    # Save the merged dataframe to a CSV file
    merged_df.to_csv(output_file, index=False)

    return merged_df



def videos_with_missing_scores(input_csv, videos_path=None):
    """
    Loads a CSV file into a DataFrame, identifies missing values in a 
    column named 'model_score_<question>', and returns a list of 
    'video_name' values corresponding to those missing entries.
    
    Parameters:
    - input_csv: str, the path to the input CSV file.
    
    Returns:
    - missing_video_names: list, containing 'video_name' values with missing 'model_score_<question>'.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Extract the 'question' from the first row in the 'question' column
    question_name = df['question_name'].iloc[0]

    # Define the target column name based on the question value
    target_column = f'model_score_{question_name}'

    # Check if the target column exists
    if target_column not in df.columns:
        print(f"Column '{target_column}' does not exist in the data.")
        return []

    # Identify rows where 'model_score_<question>' has missing values
    missing_rows = df[df[target_column].isna()]

    # Extract 'video_name' values for rows with missing 'model_score_<question>'
    missing_video_names = missing_rows['video_name'].tolist()

    # Find the path of the missing videos
    # Find paths in video_paths that match names in missing_video_names
    if videos_path is not None:
        missing_video_paths = [
            path for path in videos_path
            if os.path.splitext(os.path.basename(path))[0] in " ".join(missing_video_names)
        ]
    else:
        missing_video_paths = None

    return missing_video_names, missing_video_paths


