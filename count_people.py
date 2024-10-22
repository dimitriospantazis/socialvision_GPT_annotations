
import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint
import cv2  # OpenCV for video processing
import torch



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




# BLIP2, https://huggingface.co/docs/transformers/model_doc/blip-2
from transformers import ViltProcessor, ViltForQuestionAnswering
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# BLIP, https://huggingface.co/docs/transformers/model_doc/blip
from transformers import AutoProcessor, TFBlipForQuestionAnswering
model = TFBlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")




# Model test
videos_path = os.path.join(os.getcwd(),'social_data','videos','brushing','flickr-3-3-2-8-3-2-3-7-2833283237_47.mp4')
images, total_frames, fps = video2images(videos_path, num_frames=3)
image = images[0]


text = "How many people are there?"


# Display the image
#image.show()

# prepare inputs
inputs = processor(image, text, return_tensors="pt")



# forward pass
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])








