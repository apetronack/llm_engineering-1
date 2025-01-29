# imports
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import requests
from bs4 import BeautifulSoup
import re

# Initialization
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
MODEL = "gpt-4o-mini"
openai = OpenAI()

# System Prompt Definition
SYSTEM_PROMPT = "You are a helpful assistant who\
    generates datasets for a given data analysis task."

REFINING_SYSTEM_PROMPT = "You are a helpful assistant who \
    suggests improvements to a Python script."

# Function to generate user prompt
def generate_initial_user_prompt(data_analysis_task, file_format, num_waveforms):
    user_prompt = f"""
    The data analysis task is: {data_analysis_task}. \n
    You should draft a Python script that generates dataset outputs for this task in {file_format} format. \n
    This script should generate {num_waveforms} datasets. \n
    The script should use only common libraries such as numpy, pandas, matplotlib, etc. \n
    The script should include comments to explain what each part of the code does. \n
    """
    return user_prompt

# Function to generate code refining user prompt
def generate_refining_user_prompt(filename, error_message):
    code = open(filename).read()
    
    user_prompt = f"""
    The following Python script is not working properly: \n
    {code} \n
    The error message is: {error_message} \n
    Please suggest improvements to this script. \n
    Return the entire improved Python script. \n
    """
    return user_prompt

# Function to stream response
def stream_initial_gpt_response(data_analysis_task, file_format, num_samples):
    user_prompt = generate_initial_user_prompt(data_analysis_task, 
                                       file_format, num_samples)
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        stream=True
    )
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        yield response

    return response

# Function to stream response for refining code
def stream_refining_gpt_response(filename, error_message):
    user_prompt = generate_refining_user_prompt(filename, error_message)
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": REFINING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        stream=True
    )
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        yield response

    return response

# Define a function to extract and save the Python script from the response
def extract_code(text):
    filename = "synthetic_data_generator.py"
    
    # Regular expression to find text between ``python and ``
    match = re.search(r'```python(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        print("No code found")
        return None
    
    # Save the code to a .py file in the current directory
    with open(filename, 'w') as file:
        file.write(code)
    print(f"Code saved to {filename}")
    return filename

# Test function
DATASET_FORMAT = "CSV"
FILE_FORMAT = "CSV"
NUM_WAVEFORMS = 100
DATA_ANALYSIS_TASK = "Please create synthetic hydrophone sensor data that \
    contains a recording of a blast wave on Channel A that has a compressional peak that ranges from 0.5 - 1.5V. \
    The blast wave should have a very fast rise time (10-20ns), a slower decay time (1-2us), \
    and a longer rarefaction time (5-10 us). The rarefaction phase should range from -0.1 to -0.4V.\
    Additionally, it should have a recording of a 5-15 microsecond, 3000 - 4000V pulse on on Channel B. \
    The voltage pulse should have a fast (100-200ns) rise time, and a slow (1-3us) fall time. \
    and a recording of a 1-3 microsecond, 130 - 180A current pulse on Channel C. \
    The current pulse should have a fast (400-600ns) rise time, and a slower (750 - 1500ns) fall time. \
    All three channels should be sampled at 250 MHz for 50 microseconds. \
    The voltage pulse should start to occur first 10 microseconds into the recording. \
    The current pulse should start to occur next as soon as the voltage pulse starts to decay. \
    The blast wave should start to occur last 1-3 microseconds after the current pulse hits its peak. \
    Finally, a column should be added to the dataset that contains the time of each sample in seconds. \
    This column should be named 'time' and should be the first column in the dataset. \
    Please randomize the variable durations and values using a normally distributed random number generator. \
    Please save datasets to a folder in the current working directory called 'datasets'. \
    At the end of the script, please plot the last dataset in the folder using matplotlib with a legend and all three channels.\
    Please scale the channel outputs to be in the range of -1 to 5 to be visible on the plot. Include the scaling factors in the legend."

response_stream = stream_initial_gpt_response(DATA_ANALYSIS_TASK, FILE_FORMAT, NUM_WAVEFORMS)
response = ""

for response_so_far in response_stream:
    if response_so_far:
        # Save the full response
        response = response_so_far

# Once the response is received, extract the code
filename = extract_code(response)

# Try to run the extracted code and refine up to 3 times if it fails
max_attempts = 3
attempts = 0

while attempts < max_attempts and filename:
    try:
        exec(open(filename).read())
        print("Code executed successfully")
        break
    except Exception as e:
        error_message = str(e)
        print(f"Attempt {attempts + 1} - Error: {error_message}")
        attempts += 1
        if attempts < max_attempts:
            # Try refining the code
            response_stream = stream_refining_gpt_response(filename, error_message)
            response = ""
            for response_so_far in response_stream:
                # Save the full response
                response = response_so_far
            # Extract the refined code
            filename = extract_code(response)
        else:
            print("Max attempts reached. Could not execute the code successfully.")

# Save the full response to a Markdown file in the current directory
with open("response.md", "w") as file:
    file.write(response)