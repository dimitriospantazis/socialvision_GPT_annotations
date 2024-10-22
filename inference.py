import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from utils import find_videos, video2images, find_video_path
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


def openai_process_image(client, prompt, image, model="gpt-4o-mini", verbose=False):
    """
    Processes an image using OpenAI's GPT model, sending both the image and a prompt.
    The image is resized, converted to Base64 format, and analyzed by the model.

    Args:
        prompt (str): The text prompt to accompany the image.
        image (PIL.Image): A PIL Image object to be processed.
        model (str): The OpenAI model to use for processing (default: "gpt-4o-mini").
        verbose (bool): If True, prints additional details including the image and token usage.

    Returns:
        response (ChatCompletion): The response object from OpenAI's API containing the model's answer and token usage.
    """

    # Step 1: Resize image to a maximum dimension of 512 to reduce processing cost.
    # Ensures that either the height or width does not exceed 512 pixels, maintaining aspect ratio.
    image = resize_image(image)

    # Display the image if verbose mode is enabled (useful for debugging).
    if verbose:
        image.show()

    # Step 2: Convert the image to a Base64 string, as expected by GPT models.
    
    # Create a BytesIO buffer in memory to temporarily hold the image data.
    buffer = BytesIO()

    # Save the image to the buffer in JPEG format (or another format if needed).
    image.save(buffer, format="JPEG")  # Change format to "PNG" or others as required.

    # Extract the byte data from the buffer.
    image_bytes = buffer.getvalue()

    # Encode the byte data into a Base64 string (used for embedding the image in the API request).
    img_b64_str = base64.b64encode(image_bytes).decode('utf-8')

    # Step 3: Define the image type (MIME type), which tells the model what kind of image is being processed.
    img_type = "image/jpeg"  # Adjust this based on the actual image type (e.g., "image/png").

    # Step 4: Send the prompt and the image to OpenAI's model via the API.
    response = client.chat.completions.create(
        model=model,  # Use the specified GPT model, e.g., gpt-4o-mini.
        messages=[
            {
                "role": "user",  # The user is asking a question or providing context.
                "content": [
                    {"type": "text", "text": prompt},  # Textual prompt sent with the image.
                    {
                        "type": "image_url",  # The image data is sent as a URL-encoded Base64 string.
                        "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},  # Embedding the image in the request.
                    },
                ],
            }
        ],
    )

    # Step 5: Print the response from the model (useful for debugging or verbose mode).
    model_response = response.choices[0].message.content
    if verbose:
        print(model_response)

    # Step 6: Extract token usage details from the response, if available.
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


def openai_process_videos(
    client,
    videos_path, 
    question_type, 
    prompt, 
    output_csv_file = None, 
    model = "gpt-4o-2024-08-06",
    num_frames=4,  # Add num_frames as a parameter to control number of frames
    verbose = False
):
    """
    Function to process videos, perform inference on selected frames, and store the results in a CSV file.

    Parameters:
    - videos_path: List of paths to videos or a single path to a video folder.
    - question_type: Type of question being asked for model response.
    - prompt: Text prompt for the model.
    - output_csv_file: Path to the output CSV file where results are saved.
    - num_frames: Number of equispaced frames to extract from each video.
    - verbose: If True, displays images and prints additional debug information.
    
    Returns:
    - df: A DataFrame containing the last processed video result.
    """
    
    # If a single directory is passed, find all video paths inside the folder.
    if isinstance(videos_path, str):
        videos_path = [find_video_path(videos_path, os.path.join(os.getcwd(),'social_data','videos'))]
        if verbose:
            print(f"Video paths found: {videos_path}")

    # Process each video in the list
    for video in tqdm(videos_path, desc="Processing Videos"):

        # Extract equispaced frames from the video
        images, total_frames, fps = video2images(video, num_frames=num_frames)
        image = images[2]  # Select the middle frame for processing

        # Process the selected image using OpenAI's GPT model
        response, model_response = openai_process_image(client, prompt, image, model=model)

        # Show the image if verbose is True
        if verbose:
            plt.close() #close previous image
            plt.imshow(image)
            plt.axis('off')
            plt.show(block=False)
            time.sleep(0.1)
            pprint(f'Model response: {model_response}')



        # Create a DataFrame to store the result for the current video
        df = pd.DataFrame({
            'video_name': [os.path.basename(video)],  # Extract the file name
            'model': model,
            'question_name': question_type,
            'question': prompt,
            'model_response': model_response
        })

        # Write or append the result to the CSV file
        if output_csv_file is not None:
            if not os.path.isfile(output_csv_file):
                # Write the file with headers if it doesn't exist
                df.to_csv(output_csv_file, index=False, mode='w')
            else:
                # Append the result without headers if the file exists
                df.to_csv(output_csv_file, index=False, mode='a', header=False)

    # Return the DataFrame containing the last processed video's result
    return df














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




def openai_process_videos_multiple(
    client,
    videos_path, 
    question_type, 
    prompt, 
    output_csv_file = None, 
    model = "gpt-4o-2024-08-06",
    num_frames=3,  # Add num_frames as a parameter to control number of frames
    verbose = False
):
    """
    Function to process videos, perform inference on selected frames, and store the results in a CSV file.

    Parameters:
    - videos_path: List of paths to videos or a single path to a video folder.
    - question_type: Type of question being asked for model response.
    - prompt: Text prompt for the model.
    - output_csv_file: Path to the output CSV file where results are saved.
    - num_frames: Number of equispaced frames to extract from each video.
    - verbose: If True, displays images and prints additional debug information.
    
    Returns:
    - df: A DataFrame containing the last processed video result.
    """
    
    # If a single directory is passed, find all video paths inside the folder.
    if isinstance(videos_path, str):
        videos_path = [find_video_path(videos_path, os.path.join(os.getcwd(),'social_data','videos'))]
        if verbose:
            print(f"Video paths found: {videos_path}")

    # Process each video in the list
    for video in tqdm(videos_path, desc="Processing Videos"):

        # Extract equispaced frames from the video
        images, total_frames, fps = video2images(video, num_frames=num_frames+1) #get extra frame but then discard the first at t=0
        images = images[1:]  # Exclude first image

        # Process the selected image using OpenAI's GPT model
        response, model_response = openai_process_image_multiple(client, prompt, images, model=model)

        # Show an image if verbose is True
        if verbose:
            plt.close() #close previous image
            plt.imshow(images[0])
            plt.axis('off')
            plt.show(block=False)
            time.sleep(0.1)
            pprint(f'Model response: {model_response}')


        # Create a DataFrame to store the result for the current video
        df = pd.DataFrame({
            'video_name': [os.path.basename(video)],  # Extract the file name
            'model': model,
            'question_name': question_type,
            'question': prompt,
            'model_response': model_response
        })

        # Write or append the result to the CSV file
        if output_csv_file is not None:
            if not os.path.isfile(output_csv_file):
                # Write the file with headers if it doesn't exist
                df.to_csv(output_csv_file, index=False, mode='w')
            else:
                # Append the result without headers if the file exists
                df.to_csv(output_csv_file, index=False, mode='a', header=False)

    # Return the DataFrame containing the last processed video's result
    return df



