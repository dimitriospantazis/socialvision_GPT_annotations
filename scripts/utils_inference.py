import concurrent.futures
import os
import pandas as pd
import threading
from tqdm import tqdm
import os
import numpy as np
from scripts.utils import find_videos, video2images, find_video_path
from pprint import pprint
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import time
from typing import Optional
from pydantic import BaseModel
from textwrap import dedent
import json
import requests
import re






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


# Define the Pydantic model for parsing responses
class FrameAnalysis(BaseModel):
    score: Optional[float] = None
    confidence: Optional[float] = None
    explanation: Optional[str] = None

class SceneAnalysis(BaseModel):
    people_count: Optional[int] = None  # Number of people
    certain: Optional [bool] = None  # Whether the count is certain
    location_type: Optional [str] = None  # "Indoor" or "Outdoor"
    location_confidence: Optional [float] = None  # Confidence for location type
    explanation: Optional[str] = None  # Optional explanation



def openai_process_image_multiple(client, prompt, images, model="gpt-4o-2024-08-06", verbose=False):
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

    # Step 4: Build the content list dynamically for all images.
    content_list = [{"type": "text", "text": dedent(prompt['user'])}]
    for img_str in img_b64_str:
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{img_type};base64,{img_str}",
                "detail": "low"  # Use low detail for image processing
            }
        })

    # Step 5: Send the prompt and the images to OpenAI's model via the API.
    try:
        # API call with parsing
        response = client.beta.chat.completions.parse(
            model=model,  # Use the specified GPT model, e.g., gpt-4o-mini.
            messages=[
                { "role": "user", "content": content_list}
            ],
            max_tokens=500,
            response_format=SceneAnalysis if "people_count" in prompt['user'] else FrameAnalysis  # Use the Pydantic model as the response format
        )

        # Access the structured response
        structured_response = response.choices[0].message.parsed
        if verbose:
            print(structured_response)

    except Exception as e:
        response = []
        # Handle specific exceptions or generic ones
        structured_response = SceneAnalysis() if "people_count" in prompt['user'] else FrameAnalysis()
        structured_response.explanation = f"Failed to process response due to: {e}"
        if verbose:
            print(f"Error during API call: {e}")


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
    return response, structured_response





import concurrent.futures
import os
import time
import threading
from tqdm import tqdm
import pandas as pd


def openai_process_videos_multiple_parallel(
    client,
    videos_path, 
    question_type, 
    prompt, 
    output_csv_file=None, 
    model="gpt-4o-2024-08-06",
    num_frames=4,
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

    # Semaphore to limit the number of concurrent requests
    semaphore = threading.Semaphore(rate_limit)

    # Load the existing CSV file if it exists
    if output_csv_file and os.path.isfile(output_csv_file):
        existing_df = pd.read_csv(output_csv_file)
    else:
        existing_df = pd.DataFrame()

    # Helper function to process each video
    def process_video(video):
        # Extract equispaced frames from the video
        images, total_frames, fps = video2images(video, num_frames=num_frames)  

        # Rate-limiting logic: Acquire semaphore before making the API call
        semaphore.acquire()  # Limit the rate of requests

        try:
            # Process the selected image using OpenAI's GPT model
            response, structured_response = openai_process_image_multiple(client, prompt, images, model=model)
            
            # Throttle the request by introducing a delay (1 second per request for 60 req/min rate limit)
            time.sleep(60 / rate_limit)  # Ensure we're not making more than allowed calls per minute
        finally:
            semaphore.release()  # Release semaphore after processing

        # Create a DataFrame to store the result for the current video
        if question_type != "scene_analysis":
            new_data = pd.DataFrame({
                'video_name': [os.path.basename(video)],  # Extract the file name
                'model': model,
                'model_response': [response],
                'question_name': question_type,
                'user_prompt': dedent(prompt['user']).lstrip("\n"),
                f'model_{question_type}_explanation': dedent(structured_response.explanation).lstrip("\n"),
                f'model_{question_type}_confidence': structured_response.confidence,
                f'model_{question_type}_score': structured_response.score,
                })
        else:
            new_data = pd.DataFrame({
                'video_name': [os.path.basename(video)],  # Extract the file name
                'model': model,
                'model_response': [response],
                'question_name': question_type,
                'user_prompt': dedent(prompt['user']).lstrip("\n"),
                f'model_peoplecount_certain': structured_response.certain,
                f'model_peoplecount': structured_response.people_count,
                f'model_location_explanation': dedent(structured_response.explanation).lstrip("\n"),
                f'model_location_confidence': structured_response.location_confidence,
                f'model_location': structured_response.location_type,
                })

        return new_data

    # Process videos in parallel using ThreadPoolExecutor
    processed_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:  #Can use max_workers = 120
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

    # Sort the rows based on the video name
    updated_df = updated_df.sort_values(by='video_name').reset_index(drop=True)

    # Save the updated DataFrame to the output CSV file
    if output_csv_file:
        updated_df.to_csv(output_csv_file, index=False)


    # Return the updated DataFrame
    return updated_df








#More info here: https://platform.openai.com/docs/guides/batch/getting-started?lang=node
def create_jsonl_file(
    videos_path, 
    prompts, 
    output_jsonl_file=None, 
    model="gpt-4o-2024-08-06",
    num_frames=4,
    max_tokens=1500
):
    """
    Generate batch request entries from a list of videos and write them to a JSONL file.

    This function extracts a specified number of frames from each video, encodes them as 
    base64 images, pairs them with a provided prompt, and formats them into a request entry 
    suitable for batch processing with the OpenAI API. Each request entry is appended as a 
    new line to the specified JSONL file.

    Args:
        videos_path (list[str]): A list of paths to the input video files.
        question_type (str): A descriptor for the type of question being asked.
            This is appended to the custom_id for each request.
        prompts (list): A list of dictionaries, each containing the user prompt (e.g., {'user': 'Describe the scene'}).
        output_jsonl_file (str): The path to the output JSONL file where request entries will be appended.
        model (str, optional): The OpenAI model to be used. Defaults to "gpt-4o-2024-08-06".
        num_frames (int, optional): The number of frames to extract per video. Defaults to 4.
        max_tokens (int, optional): The maximum tokens for the response. Defaults to 1500.

    Returns:
        None
    """

    for video_path in tqdm(videos_path):

        # Extract frames from the video. This should return a list of PIL Images.
        images, total_frames, fps = video2images(video_path, num_frames=num_frames)

        # Resize and convert each image to base64.
        resized_images = [resize_image(img) for img in images]
        img_b64_list = [image2base64(img) for img in resized_images]

        for question_type, prompt in prompts.items():

            # Derive a custom_id from the question type and the video filename.
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            custom_id = f"{question_type}_{video_name}"

            # Build the user content: start with the prompt text.
            # The prompt is dedented to remove any unwanted indentation.
            content_list = [{"type": "text", "text": dedent(prompt['user'])}]

            # Append each image as a data URL.
            img_type = "image/jpeg"
            for img_b64 in img_b64_list:
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img_type};base64,{img_b64}",
                        "detail": "low"
                    }
                })

            # Construct the messages array. Currently, it just includes the user message with images.
            messages = [{"role": "user", "content": content_list}]

            # Prepare the request body for the API call.
            body = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens
            }

            # Create the final request entry.
            request_entry = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }

            # Append the request_entry as a line to the output JSONL file.
            with open(output_jsonl_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(request_entry) + "\n")



def upload_jsonl_file(api_key: str, filepath: str, purpose: str = "batch", verbose=False) -> str:
    """
    Upload a JSONL file to the OpenAI API.

    Args:
        api_key (str): Your OpenAI API key.
        filepath (str): The path to the JSONL file to upload.
        purpose (str): The purpose of the file (default: "batch").

    Returns:
        str: The ID of the uploaded file.

    Raises:
        Exception: If the file upload fails.
    """

    # The 'files' parameter in requests.post automatically sets multipart/form-data.
    with open(filepath, "rb") as f:
        response = requests.post(
            "https://api.openai.com/v1/files",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (filepath, f)},
            data={"purpose": purpose}
        )

    if response.status_code == 200:
        print("File uploaded successfully!")
        json_response = response.json()
        if verbose:
            print(json_response)
        file_id = json_response["id"]
        return file_id
    else:
        print("Failed to upload file.")
        print(response.text)
        raise Exception(f"File upload failed with status code {response.status_code}")



def list_openai_files(api_key: str) -> list:
    """
    List all files uploaded to the OpenAI API and print their 'id' and 'filename'.

    Args:
        api_key (str): Your OpenAI API key.

    Returns:
        list: A list of file objects, each represented as a dictionary.
    """
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.get("https://api.openai.com/v1/files", headers=headers)

    if response.status_code == 200:
        files = response.json().get("data", [])
        for f in files:
            print(f"ID: {f.get('id')}, Filename: {f.get('filename')}")
        return files
    else:
        print("Failed to retrieve files.")
        print(response.text)
        raise Exception(f"List files failed with status code {response.status_code}")
    


def delete_openai_file(api_key: str, file_id: str):
    """
    Delete one or all files from the OpenAI API by their ID.

    Args:
        api_key (str): Your OpenAI API key.
        file_id (str): The ID of the file to delete, or 'all' to delete all files.

    Returns:
        dict or None: The JSON response from the API after deleting a single file,
                      or None if multiple files are deleted.
    """

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    if file_id.lower() == 'all':
        # Delete all files
        files = list_openai_files(api_key)
        for f in files:
            fid = f.get('id')
            print(f"Attempting to delete file with ID: {fid}")
            response = requests.delete(f"https://api.openai.com/v1/files/{fid}", headers=headers)
            if response.status_code == 200:
                print(f"Successfully deleted file: {fid}")
            else:
                print(f"Failed to delete file: {fid}")
                print(response.text)
        return None
    else:
        # Delete a single file
        response = requests.delete(f"https://api.openai.com/v1/files/{file_id}", headers=headers)
        if response.status_code == 200:
            json_response = response.json()
            print(f"Successfully deleted file: {file_id}")
            return json_response
        else:
            print(f"Failed to delete file: {file_id}")
            print(response.text)
            raise Exception(f"Delete file failed with status code {response.status_code}")



def create_batch_job(api_key: str, uploaded_file_id: str, record_file: str = None, verbose=False) -> dict:
    """
    Create a batch (patch) job with the given file_id, and then check its status.
    Optionally include a 'custom_id' field to help identify this job.
    Additionally, store the batch_id in the file 'file_id' using pandas.

    Args:
        api_key (str): Your OpenAI API key.
        uploaded_file_id (str): The ID of the file previously uploaded.
        record_file (str): The file storing the batch IDs (batch_id). 

    Returns:
        dict: The JSON response of the batch status request.

    Raises:
        Exception: If the batch creation or status retrieval fails.
    """

    # Prepare the batch data
    batch_data = {
        "input_file_id": uploaded_file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
    }

    # Create the headers for the API request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Create the batch job
    batch_response = requests.post(
        "https://api.openai.com/v1/batches",
        headers=headers,
        json=batch_data
    )

    if batch_response.status_code != 200:
        print("Failed to create batch (patch) job:", batch_response.text)
        raise Exception(f"Batch creation failed with status code {batch_response.status_code}")

    # Extract the batch_id from the response
    batch_id = batch_response.json()["id"]

    # Check the status of the batch
    status_url = f"https://api.openai.com/v1/batches/{batch_id}"
    status_response = requests.get(status_url, headers={"Authorization": f"Bearer {api_key}"})

    if status_response.status_code != 200:
        print("Failed to retrieve batch status:", status_response.text)
        raise Exception(f"Batch status retrieval failed with status code {status_response.status_code}")

    status_json = status_response.json()
    print(f"\nBatch status: {status_json['status']}")
    if verbose:
        print("\nBatch status response:", status_json)

    # Append to record_file using pandas
    file_exists = os.path.isfile(record_file)

    # Prepare the data for the new row
    new_row = {"batch_id": batch_id}

    if file_exists:
        # If the file exists, read it into a DataFrame
        df = pd.read_csv(record_file)
        # Append the new row
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # If the file does not exist, create a new DataFrame with one row
        df = pd.DataFrame([new_row])

    # Write the DataFrame back to the CSV file
    df.to_csv(record_file, index=False)

    return batch_id, status_json



def status_batch_job(api_key: str, batch_id: str, verbose=False) -> dict:
    """
    Check the status of a batch job on the OpenAI API.

    Args:
        api_key (str): Your OpenAI API key.
        batch_id (str): The ID of the batch job to check.

    Returns:
        dict: The full JSON response of the batch status.
    """
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    status_url = f"https://api.openai.com/v1/batches/{batch_id}"
    status_response = requests.get(status_url, headers=headers)

    if status_response.status_code != 200:
        print("Failed to retrieve batch status:", status_response.text)
        raise Exception(f"Batch status retrieval failed with status code {status_response.status_code}")

    json_response = status_response.json()
    print("Batch Status:", json_response.get('status'))

    if verbose:
        print("\nFull Response:", json_response)

    return json_response




def download_openai_results(record_file, api_key):
    """
    Downloads result files from OpenAI for a given batch of jobs stored in record_file.

    Args:
        record_file (str): Path to the CSV file containing batch IDs.
        api_key (str): OpenAI API key for authentication.

    Returns:
        None
    """
    # Read the CSV file to get batch IDs
    df = pd.read_csv(record_file)
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    # Iterate through each batch ID and download results
    for chunk_number, batch_id in enumerate(df['batch_id']):
        print(f'Processing chunk: {chunk_number}')

        # Get the status of the batch job
        json_response = status_batch_job(api_key, batch_id)
        output_file_id = json_response.get('output_file_id')
        results_file_name = f"{record_file[:-4]}_chunk{chunk_number}_results.jsonl"

        if output_file_id:
            download_url = f"https://api.openai.com/v1/files/{output_file_id}/content"
            download_response = requests.get(download_url, headers=headers)
            
            if download_response.status_code == 200:
                # Save the results to a file
                with open(results_file_name, "w", encoding="utf-8") as outfile:
                    outfile.write(download_response.text)
                print(f"Batch results saved to {results_file_name}")
            else:
                print(f"Failed to download results for chunk {chunk_number}: {download_response.text}")
        else:
            print(f"No output file ID found for chunk {chunk_number}.") 





def parse_openai_results(input_file):
    """
    Parse the results from an OpenAI batch job stored in a JSONL file.
    Creates a DataFrame upfront and appends rows for each parsed entry.
    
    Args:
        input_file (str): The path to the JSONL file containing the batch results.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - custom_id
            - video_name
            - question_type
            - scene_analysis (as a dict)
    """

    question_types = [
        'distance', 'object', 'expanse', 'facingness', 
        'communicating', 'joint', 'valence', 'arousal', 'scene_analysis'
    ]

    dtypes = {}
    dtypes['video_name'] = str
    for qtype in question_types[:-1]:
        dtypes[f"model_{qtype}_score"] = np.float64
        dtypes[f"model_{qtype}_confidence"] = np.float64
        dtypes[f"model_{qtype}_explanation"] = str

    # scene analysis columns
    dtypes["model_peoplecount"] = np.float64
    dtypes["model_peoplecount_certain"] = str  # boolean could also be used if values are bool
    dtypes["model_location"] = str
    dtypes["model_location_confidence"] = np.float64
    dtypes["model_location_explanation"] = str

    results_df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in dtypes.items()})
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)

            custom_id = data.get("custom_id")
            if not custom_id:
                continue

            # Extract question_type and video_name from custom_id
            match = re.match(f"^({'|'.join(question_types)})_(.+)$", custom_id)
            if not match:
                continue

            question_type, video_name = match.groups()
            video_name += ".mp4"

            # Extract the assistant message content
            try:
                content = data["response"]["body"]["choices"][0]["message"]["content"]
            except KeyError:
                continue

            # Try to check if json exists in the content but was not written as json
            if all(word in content.lower() for word in ['explanation', 'score','confidence']) and not any(word in content.lower() for word in ['```json']):
                # Add the missing json text
                content = f"```json\n{content}\n```"

            # Try to check if json exists in the content but was not written as json (for the scene_analysis case)
            if all(word in content.lower() for word in ['explanation', 'people_count','people_count_certain','location_type','location_confidence']) and not any(word in content.lower() for word in ['```json']):
                # Add the missing json text
                content = f"```json\n{content}\n```"

            # Find the JSON block inside the code block
            json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', content, flags=re.DOTALL)

            if not json_block_match:
                # Store the json_block_match message to the explanation column
                val = question_type if question_type != "scene_analysis" else "location"
                update_dict = {
                    "video_name": video_name,
                    f"model_{question_type}_explanation": content
                }
                dataframe_update_or_add_row(results_df, video_name, update_dict)
                continue

            json_str = json_block_match.group(1)
            
            # Parse the JSON string inside the code block
            try:
                response_data = json.loads(json_str)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for custom_id: {custom_id}")
                continue
            
            if question_type != "scene_analysis":
                # Add the parsed data to the DataFrame
                update_dict = {
                    "video_name": video_name,
                    f"model_{question_type}_score": response_data.get("score"),
                    f"model_{question_type}_confidence": response_data.get("confidence"),
                    f"model_{question_type}_explanation": response_data.get("explanation")
                }
            else:
                # Add the parsed data to the DataFrame
                update_dict = {
                    "video_name": video_name,
                    "model_peoplecount": response_data.get("people_count"),
                    "model_peoplecount_certain": response_data.get("people_count_certain"),
                    "model_location": response_data.get("location_type"),
                    "model_location_confidence": response_data.get("location_confidence"),
                    "model_location_explanation": response_data.get("explanation")
                }

            # Update the DataFrame with the parsed data
            dataframe_update_or_add_row(results_df, video_name, update_dict)

            """
            # Check if the video_name already exists in the DataFrame
            existing_idx = results_df.index[results_df["video_name"] == video_name]

            if not existing_idx.empty:
                # Update the existing row
                idx = existing_idx[0]
                for col, val in update_dict.items():
                    results_df.at[idx, col] = val
            else:
                # Create a new row
                row = {"video_name": video_name}
                row.update(update_dict)
                results_df.loc[len(results_df)] = row
            """

    return results_df



def dataframe_update_or_add_row(df, video_name, update_dict):
    """
    Update the row with the given video_name in the DataFrame if it exists, 
    otherwise add a new row.

    :param df: The DataFrame to update.
    :param video_name: The video name to search for.
    :param update_dict: A dictionary of columns and their values to update or add.
    """
    existing_idx = df.index[df["video_name"] == video_name]
    if not existing_idx.empty:
        # Update the existing row
        df.loc[existing_idx[0], update_dict.keys()] = update_dict.values()
    else:
        # Add a new row
        df.loc[len(df)] = {"video_name": video_name, **update_dict}






