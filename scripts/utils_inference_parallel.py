import concurrent.futures
import os
import pandas as pd
import threading
from tqdm import tqdm
import os
import numpy as np
import torch
from scripts.utils import find_videos, video2images, find_video_path
from pprint import pprint
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import time

# Resize image to a maximum dimension of 512
def resize_image(image, max_size=512):

    width, height = image.size
    if max(width, height) > max_size:  # Check if either dimension is larger than the max size
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image



def openai_process_image_multiple(client, prompt, images, model="gpt-4o-mini", verbose=False):
    """
    Processes multiple images using OpenAI's GPT model, sending both the images and a prompt.
    The images are resized, converted to Base64 format, and analyzed by the model.

    Args:
        prompt (str): The text prompt to accompany the images.
        images (list of PIL.Image): A list of PIL Image objects to be processed.
        model (str): The OpenAI model to use for processing (default: "gpt-4o-mini").
        verbose (bool): If True, prints additional details including the images and token usage.

    Returns:
        response (ChatCompletion): The response object from OpenAI's API containing the model's answer and token usage.
    """

    # Step 1: Resize each image to a maximum dimension of 512 to reduce processing cost.
    # This ensures that either the height or width does not exceed 512 pixels, maintaining aspect ratio.
    images = [resize_image(image) for image in images]

    # Step 2: Convert each image to a Base64-encoded string.
    img_b64_str = [image2base64(image) for image in images]

    # Display the first image if verbose mode is enabled (useful for debugging).
    if verbose:
        images[0].show()

    # Step 3: Define the image type (MIME type), typically "image/jpeg".
    img_type = "image/jpeg"  # Adjust this based on the actual image type (e.g., "image/png").

    # Step 4: Build the content list dynamically for each image.
    content_list = [{"type": "text", "text": prompt}]
    for img_str in img_b64_str:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:{img_type};base64,{img_str}"}
        })

    # Step 5: Send the prompt and the images to OpenAI's model via the API.
    response = client.chat.completions.create(
        model=model,  # Use the specified GPT model, e.g., gpt-4o-mini.
        messages=[
            {
                "role": "user",  # The user is asking a question or providing context.
                "content": content_list,  # Dynamically created content list with multiple images.
            }
        ],
    )


    # Step 6: Print the response from the model (useful for debugging or verbose mode).
    model_response = response.choices[0].message.content
    if verbose:
        print(model_response)

    # Step 7: Extract token usage details from the response, if available.
    # Token usage information helps in understanding the cost of the API call.
    if verbose and hasattr(response, 'usage'):
        total_tokens = response.usage.total_tokens  # Total tokens used (prompt + response).
        prompt_tokens = response.usage.prompt_tokens  # Tokens used for the prompt.
        completion_tokens = response.usage.completion_tokens  # Tokens used for the model's response.

        # Print token usage details.
        print(f"Total tokens used: {total_tokens}")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Completion tokens: {completion_tokens}")

        # Step 7: Calculate the estimated cost based on token usage.
        # The example rate is $0.001275 per 1,000,000 tokens. Adjust this based on actual model pricing.
        cost_per_1m_tokens = 0.001275  # Example rate for gpt-4o-mini, adjust as needed.
        total_cost = (total_tokens / 1_000_000) * cost_per_1m_tokens

        # Print the estimated cost of the API call.
        print(f"Estimated cost: ${total_cost:.6f}")

    # Return the API response object for further use.
    return response, model_response


def image2base64(image, format="JPEG"):

    # Convert the image to a Base64 string, as expected by GPT models.
    
    # Create a BytesIO buffer in memory to temporarily hold the image data.
    buffer = BytesIO()

    # Save the image to the buffer in JPEG format (or another format if needed).
    image.save(buffer, format)  # Change format to "PNG" or others as required.

    # Extract the byte data from the buffer.
    image_bytes = buffer.getvalue()

    # Encode the byte data into a Base64 string (used for embedding the image in the API request).
    img_b64_str = base64.b64encode(image_bytes).decode('utf-8')

    return img_b64_str



import concurrent.futures
import os
import time
import threading
from tqdm import tqdm
import pandas as pd

# Create a lock for CSV writing and a semaphore for rate-limiting the API calls
csv_lock = threading.Lock()

# Assuming OpenAI allows 60 requests per minute (1 request per second)
rate_limit = 20  # OpenAI API limit: x requests per minute
semaphore = threading.Semaphore(rate_limit)  # Semaphore to limit requests

def openai_process_videos_multiple_parallel(
    client,
    videos_path, 
    question_type, 
    prompt, 
    output_csv_file=None, 
    model="gpt-4o-2024-08-06",
    num_frames=3,
    verbose=False,
    rate_limit=60
):
    """
    Function to process videos, perform inference on selected frames, and store the results in a CSV file.
    Limits API calls to adhere to OpenAI rate limits.

    Parameters:
    - client: The OpenAI client to interact with GPT models.
    - videos_path: List of paths to videos or a single path to a video folder.
    - question_type: Type of question being asked for model response.
    - prompt: Text prompt for the model.
    - output_csv_file: Path to the output CSV file where results are saved.
    - num_frames: Number of equispaced frames to extract from each video.
    - verbose: If True, displays images and prints additional debug information.
    
    Returns:
    - updated_df: The updated DataFrame with all processed results.
    """

    # Load the existing CSV file if it exists
    if output_csv_file and os.path.isfile(output_csv_file):
        existing_df = pd.read_csv(output_csv_file)
    else:
        existing_df = pd.DataFrame()

    # Helper function to process each video
    def process_video(video):
        # Extract equispaced frames from the video
        images, total_frames, fps = video2images(video, num_frames=num_frames+1)  # Get extra frame, discard the first at t=0
        images = images[1:]  # Exclude first image

        # Rate-limiting logic: Acquire semaphore before making the API call
        semaphore.acquire()  # Limit the rate of requests

        try:
            # Process the selected image using OpenAI's GPT model
            response, model_response = openai_process_image_multiple(client, prompt, images, model=model)
            
            # Throttle the request by introducing a delay (1 second per request for 60 req/min rate limit)
            time.sleep(60 / rate_limit)  # Ensure we're not making more than allowed calls per minute
        finally:
            semaphore.release()  # Release semaphore after processing

        # Create a DataFrame to store the result for the current video
        new_data = pd.DataFrame({
            'video_name': [os.path.basename(video)],  # Extract the file name
            'model': model,
            'question_name': question_type,
            'question': prompt,
            'model_response_'+question_type: model_response
        })

        return new_data

    # Process videos in parallel using ThreadPoolExecutor
    processed_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=120) as executor:
        futures = [executor.submit(process_video, video) for video in videos_path]

        # Show progress using tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(videos_path), desc="Processing Videos"):
            try:
                processed_data.append(future.result())
            except Exception as e:
                print(f"Error processing video: {e}")
    
    # Combine all processed data into a single DataFrame
    processed_df = pd.concat(processed_data, ignore_index=True)

    if not existing_df.empty:
        # Drop rows from existing_df where 'video_name' matches any in processed_df
        existing_df = existing_df[~existing_df['video_name'].isin(processed_df['video_name'])]
        
        # Concatenate remaining rows in existing_df with all rows in processed_df
        updated_df = pd.concat([existing_df, processed_df], ignore_index=True)
    else:
        updated_df = processed_df

    # Save the updated DataFrame to the output CSV file
    if output_csv_file:
        updated_df.to_csv(output_csv_file, index=False)

    # Return the updated DataFrame
    return updated_df


