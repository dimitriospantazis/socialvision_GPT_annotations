
# Add scripts to Python's path
import sys
sys.path.append('./scripts')

import os
import matplotlib.pyplot as plt
from datetime import datetime
from prompts_final import get_prompts
from utils import find_videos
from openai import OpenAI
import requests
from utils_inference import create_jsonl_file, upload_jsonl_file, status_batch_job, create_batch_job, list_openai_files, delete_openai_file, download_openai_results, parse_openai_results
import time
import pandas as pd

# OpenAI key
api_key = os.getenv("OPENAI_API_KEY")

# Customize to load your videos 
#videos_dir = os.path.join(os.getcwd(),'videos','human_annotated_videos_720')
#type = 'structured_notcombined'
#group = '720'
#videos_path = find_videos(videos_dir)


for group in ['1','2','3','4','5']:
    type = 'npeople2'
    videos_dir = os.path.join(os.getcwd(),'videos',f'{type}_training',group)
    videos_path = find_videos(videos_dir)

    # Load prompts
    prompts = get_prompts()

    # housekeeping: list uploaded files
    files = list_openai_files(api_key)
    # delete_openai_file(api_key, 'all') # caution: it will delete results too!

    # Create jsonl task file
    chunk_size = 155
    for i in range(0, len(videos_path), chunk_size):
        # Extract the current chunk of videos
        chunk = videos_path[i:i + chunk_size]
        chunk_number = i // chunk_size
        jsonlfile = f'task_{type}_{group}_chunk{chunk_number}.jsonl'
        create_jsonl_file(chunk, prompts, output_jsonl_file=jsonlfile, model="gpt-4o-2024-08-06", num_frames=4, max_tokens=1500)




# Upload jsonl files and create batch jobs
group = '2'
type = 'npeople2'
file_id = {}
for chunk_number in range(0,13):
    print(f'\nchunk number: {chunk_number}')
    jsonlfile = f'task_{type}_{group}_chunk{chunk_number}.jsonl'
    file_id = upload_jsonl_file(api_key=api_key, filepath=jsonlfile, purpose="batch")
    time.sleep(2) # wait 2 seconds between uploads
    json_response = create_batch_job(api_key, file_id, record_file=f'task_{type}_{group}.txt')


# Check status of batch jobs
df = pd.read_csv(f'task_{type}_{group}.txt')
for chunk_number in range(len(df)):
    print(f'chunk number: {chunk_number}')
    json_response = status_batch_job(api_key, df['batch_id'][chunk_number])


# Download jsonl results
record_file = f'task_{type}_{group}.txt'
download_openai_results(record_file, api_key)


# Parse jsonl results
merged_df = pd.DataFrame()
for chunk_number in range(13):
    input_file = f'task_{type}_{group}_chunk{chunk_number}_results.jsonl'
    results_df = parse_openai_results(input_file)
    merged_df = pd.concat([merged_df, results_df], axis=0)
# Optional: Reset the index of the merged DataFrame after concatenation
merged_df.reset_index(drop=True, inplace=True)
merged_df.to_csv(f'task_{type}_{group}_results.csv', index=False)







    
