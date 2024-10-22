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


def execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Execution time of {func.__name__}: {elapsed_time:.4f} seconds")
        return result  # Return the function's result
    return wrapper




def parse_model_responses(input_csv):
    """
    Loads a CSV file containing a DataFrame with a 'model_response' field, 
    extracts the score from the 'model_response', and adds it to a new column 'model_score'.
    
    Parameters:
    - input_csv: Path to the input CSV file.
    
    Returns:
    - df: The modified DataFrame with the new 'model_score' column.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Define a function to extract the score from the 'model_response' column
    def extract_score(response):
        # Use regex to find the score in the format "<score> = X.X"
        match = re.search(r'<score>\s*=\s*(\d+(\.\d+)?)', response)
        if match:
            return float(match.group(1))  # Extract the score as a float
        return None  # Return None if no score is found

    # Apply the extract_score function to the 'model_response' column and create the 'model_score' column
    df['model_score'] = df['model_response'].apply(extract_score)

    # Save the modified DataFrame to the same file or a new file (optional)
    df.to_csv(input_csv, index=False)  # Uncomment if you want to save the result
    
    return df



def combine_human_model_responses(model_csv, human_csv, prompt_type):
    """
    Loads a DataFrame with mean human responses and lists of responses for each video,
    and writes the results to a new CSV file.

    Parameters:
    - input_csv: Path to the input CSV file.
    - output_csv: Path to the output CSV file where results will be saved.
    - df_h: DataFrame containing 'question_name', 'video_name', and 'likert_response'.

    Returns:
    - df: Updated DataFrame with new columns 'mean_human_response' and 'human_response'.
    """

    # Load the input DataFrames
    df = pd.read_csv(model_csv)
    df_h = pd.read_csv(human_csv)

    # Initialize new columns with None values
    df['mean_human_response'] = None
    df['human_response'] = None

    # Loop through each video in the DataFrame
    for video in df['video_name']:
        # Extract the likert responses for the current video
        likert_response = list(
            df_h[(df_h['question_name'] == prompt_type) & 
                 (df_h['video_name'] == os.path.basename(video))]['likert_response']
        )

        # Update 'mean_human_response' with the mean of the responses
        df.loc[df['video_name'] == video, 'mean_human_response'] = np.mean(likert_response)

        # Store the list of responses as a string in 'human_response'
        df.loc[df['video_name'] == video, 'human_response'] = str(likert_response)


    # Convert the string to a list using ast.literal_eval()
    df['human_response'] = df['human_response'].apply(ast.literal_eval)


    # Function to select 5 random elements without replacement from a list
    def select_random_elements(lst, num_elements=5):
        # Ensure that the list has at least num_elements to sample
        if len(lst) >= num_elements:
            return random.sample(lst, num_elements)
        else:
            return lst + [None] * (num_elements - len(lst))  # Fill with None if not enough elements

    # Apply the function to get five unique selections
    df[['virtual_human1', 'virtual_human2', 'virtual_human3', 'virtual_human4', 'virtual_human5']] = pd.DataFrame(
        df['human_response'].apply(lambda x: select_random_elements(x, 5)).tolist(),
        index=df.index
    )

    # Write the updated DataFrame to the output CSV file
    df.to_csv(model_csv, index=False, mode='w')

    # print(f"Results saved to {output_csv}")
    return df



def human_model_correlation(input_csv):
    """
    Load a DataFrame, extract numeric values from model response columns, 
    and compute the correlation between the average model response and the human response.
    
    Parameters:
    - input_csv: Path to the input CSV file.
    
    Returns:
    - correlation: The computed correlation coefficient.
    """
    # Load the input DataFrame
    df = pd.read_csv(input_csv)

    # Select the columns of interest: 'model_score' and 'virtual_human1' to 'virtual_human5'
    columns_of_interest = ['model_score', 'mean_human_response', 'virtual_human1', 'virtual_human2', 'virtual_human3', 'virtual_human4', 'virtual_human5']

    # Compute the correlation matrix between all the columns
    corr_matrix = df[columns_of_interest].corr()
    print('Pearsons correlation matrix:')
    print(corr_matrix)

    # Compute Spearman correlation for all combinations
    spearman_corr_matrix = df[columns_of_interest].corr(method='spearman')
    print('Pearsons correlation matrix:')
    print(spearman_corr_matrix)

    return corr_matrix, spearman_corr_matrix





