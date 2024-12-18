
# Add scripts to Python's path
import sys
sys.path.append('./scripts')

import os
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint
from scripts.utils_inference import openai_process_image_multiple, openai_process_videos_multiple_parallel 
from prompts_final import get_prompts
from utils import find_videos, find_video_path, video2images, merge_csv_columns
from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Load videos paths   
#videos_dir = os.path.join(os.getcwd(),'videos','human_annotated_videos_720')
#type = 'structured_notcombined'
#group = '720'
#videos_path = find_videos(videos_dir)


type = 'npeople2'
group = '1'
videos_dir = os.path.join(os.getcwd(),'videos',f'{type}_training',group)
videos_path = find_videos(videos_dir)

# Load prompts
prompts = get_prompts()



# Dictionary to store output CSV file paths for each question_type
output_files = {}

# Model inference for all questions
now = datetime.now().strftime("%Y%m%d_%H%M%S")
for question_type, prompt in prompts.items():

    # Generate a unique filename for each question type
    filename = f'000_labels_{type}_{group}_{question_type}_{now}.csv'
    output_csv_file = os.path.join(videos_dir,filename)
    output_files[question_type] = output_csv_file  # Store filename in dictionary

    # Model inference (parallel execution)
    df = openai_process_videos_multiple_parallel(client, videos_path, question_type, prompt, output_csv_file, model="gpt-4o-2024-08-06", num_frames=4, verbose=False, rate_limit=60)


# Merge the CSV files and save the result to 'merged_data.csv'
csv_files = list(output_files.values())
output_file = os.path.join(videos_dir,f'000_labels_{type}_{group}_merged_{now}.csv')
merged_df = merge_csv_columns(csv_files, output_file)




    
