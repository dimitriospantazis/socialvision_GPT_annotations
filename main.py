
import os
from utils import find_videos, find_video_path, video2images, combine_human_model_responses, human_model_correlation, parse_model_responses
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from inference import openai_process_image, openai_process_videos, openai_process_image_multiple, openai_process_videos_multiple
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint

from openai import OpenAI
import os

# Set your OpenAI API key
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Load human annotations for 250 videos
humanannotfile = os.path.join(os.getcwd(),'social_data','individual_participant_ratings_fmri-train-test.csv')
df_h = pd.read_csv(humanannotfile)
videos_annot = df_h['video_name'].unique()

# Find videos with human annotations
video_dir = os.path.join(os.getcwd(),'social_data','videos')
videos_path = find_videos(video_dir)
videos_path_annot = [video for video in videos_path if os.path.basename(video) in set(videos_annot)]




question_type = 'communicating'
# correlation = -0.57 multiple images (humans: 0.57-0.72)
prompt = """
Here are several frames from a video. On a scale from 0 to 1, where 0 means the individuals are not communicating with each other at all (such as being entirely focused on separate activities without any visible interaction), and 1 means the individuals are fully engaged in communication (such as talking, gesturing, or making eye contact), how much are the people in this scene communicating with each other? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

Example 1:
Scene: Two people are sitting in the same room, but one is reading a book while the other is on the phone, without any interaction between them.
Answer: <score> = 0.0

Example 2:
Scene: Two individuals are sitting across from each other, speaking and making hand gestures as they converse.
Answer: <score> = 1.0

Now, please estimate the score for the scene in the provided images:
Answer: <score> =

Explain why you gave this score, especially if the difference is subtle:
"""



#Here are several frames from a video. Please analyze the sequence of frames and describe the action being performed. 





# Model test
videos_path = find_video_path('flickr-3-3-2-8-3-2-3-7-2833283237_47.mp4', os.path.join(os.getcwd(),'social_data','videos'))
images, total_frames, fps = video2images(videos_path, num_frames=4)
response, model_response = openai_process_image(client, prompt, images[2], model="gpt-4o-2024-08-06", verbose=True)
response, model_response = openai_process_image_multiple(client, prompt, images, model="gpt-4o-2024-08-06", verbose=True)

# Model test 2
images, total_frames, fps = video2images(videos_path_annot[11], num_frames=4)
response, model_response = openai_process_image(client, prompt, images[2], model="gpt-4o-2024-08-06", verbose=True)



# Model Inference
output_csv_file = f'video_results_{question_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv'
df = openai_process_videos_multiple(client, videos_path_annot, question_type, prompt, output_csv_file, model="gpt-4o-2024-08-06", num_frames=3, verbose=False)

# Try parallel
from inference_parallel import openai_process_videos_multiple_parallel
output_csv_file = f'video_results_{question_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv'
df = openai_process_videos_multiple_parallel(client, videos_path_annot, question_type, prompt, output_csv_file, model="gpt-4o-2024-08-06", num_frames=3, verbose=False)




# Model evaluation
model_csv = output_csv_file
human_csv = os.path.join(os.getcwd(),'social_data','individual_participant_ratings_fmri-train-test.csv')
df = parse_model_responses(model_csv)
df = combine_human_model_responses(model_csv, human_csv, question_type)
corr, spear_corr = human_model_correlation(model_csv)


# Scatter plots
df['mean_human_response']
df['model_vote']
# Create the scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(df['mean_human_response'], df['model_score'], alpha=0.7)
plt.title('Scatterplot of Mean Human Response vs Model Vote')
plt.xlabel('Mean Human Response')
plt.ylabel('Model Vote')
plt.grid(True)
plt.show(block=False)












