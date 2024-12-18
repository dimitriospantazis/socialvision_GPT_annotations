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
        'model_distance_score', 'model_distance_confidence',
        'model_object_score', 'model_object_confidence',
        'model_expanse_score', 'model_expanse_confidence',
        'model_facingness_score', 'model_facingness_confidence', 
        'model_communicating_score', 'model_communicating_confidence',
        'model_joint_score', 'model_joint_confidence',
        'model_valence_score', 'model_valence_confidence',
        'model_arousal_score', 'model_arousal_confidence',
        'model_peoplecount','model_peoplecount_certain','model_location','model_location_confidence'
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


