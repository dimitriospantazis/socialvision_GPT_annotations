
import os
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint
from scripts.utils_inference_parallel import openai_process_image_multiple, openai_process_videos_multiple_parallel 
from scripts.prompts_final import get_prompts
from scripts.utils import find_videos, find_video_path, video2images, parse_model_responses, merge_csv_columns, videos_with_missing_scores
from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Load videos paths   
# videos_dir = os.path.join(os.getcwd(),'videos','human_annotated_videos_720')
type = 'npeople2'
group = '1'
videos_dir = os.path.join(os.getcwd(),'videos',f'{type}_training',group)
videos_path = find_videos(videos_dir)

# Load prompts
prompts = get_prompts()


# $5.23 -> 

# Dictionary to store output CSV file paths for each question_type
output_files = {}

# Model inference for all questions
now = datetime.now().strftime("%Y%m%d_%H%M%S")
for question_type, prompt_text in prompts.items():

    # Generate a unique filename for each question type
    filename = f'000_labels_{type}_{group}_{question_type}_{now}.csv'
    output_csv_file = os.path.join(videos_dir,filename)
    output_files[question_type] = output_csv_file  # Store filename in dictionary

    # Model inference (parallel execution)
    df = openai_process_videos_multiple_parallel(client, videos_path, question_type, prompt_text, output_csv_file, model="gpt-4o-2024-08-06", num_frames=3, verbose=False)
    df = parse_model_responses(output_csv_file)



# Loop to revisit files and add missing scores
for question_type, output_csv_file in output_files.items():

    # Parse model scores, get missing videos, call inference again
    df = parse_model_responses(output_csv_file)
    missing_video_names, missing_video_paths = videos_with_missing_scores(output_csv_file, videos_path=videos_path)
    print(f"Missing scores for condition '{question_type}' are: {len(missing_video_names)}")
    if len(missing_video_names) > 0:
        df = openai_process_videos_multiple_parallel(client, missing_video_paths, question_type, prompt_text, output_csv_file, model="gpt-4o-2024-08-06", num_frames=6, verbose=False)
        df = parse_model_responses(output_csv_file)
        missing_video_names, missing_video_paths = videos_with_missing_scores(output_csv_file, videos_path=videos_path)
        print(f"Missing scores for condition '{question_type}' are (postprocessed): {len(missing_video_names)}")



# Merge the CSV files and save the result to 'merged_data.csv'
csv_files = list(output_files.values())
output_file = os.path.join(videos_dir,f'000_labels_{type}_{group}_merged_{now}.csv')
merged_df = merge_csv_columns(csv_files, output_file)




















