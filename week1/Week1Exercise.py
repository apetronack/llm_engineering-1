# Week 1 Exercise

# Imports
import os
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from openai import OpenAI
import ollama

# Constants
MODEL_GPT = 'gpt-4o-mini'
MODEL_LLAMA = 'llama3.2'
OLLAMA_API = "http://localhost:11434/api/chat"

# Set up environment
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# Check the key
if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not api_key.startswith("sk-proj-"):
    print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif api_key.strip() != api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
# else:
#     print("API key found and looks good so far!")

openai = OpenAI()

# Function to create the message for the model
# Inputs:     system_prompt: The system prompt to guide the model's behavior
#             user_prompt: The user's question or prompt for the model
# Outputs:    message:     List of dictionaries that include system and user prompts
def create_message(system_prompt, user_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# Function to stream response from OpenAI
# Inputs:     message_input: List of dictionaries that include system and user prompts
#             model_select: The model to use for the response (i.e., gpt-4o-mini or llama3.2)
#             stream_choice: Boolean to determine if the response should be streamed or not
# Outputs:    response:    The response from the model
#             Also prints the response in Markdown format
def stream_response_openai(message_input, model_select, stream_choice):
    display_handle = display(Markdown(""), display_id=True)
    if stream_choice and display_handle is not None:
        stream = openai.chat.completions.create(
            model=model_select,
            messages=message_input,
            stream=True
        )
        response = ""
        
        for chunk in stream:
            response += chunk.choices[0].delta.content or ''
            response = response.replace("```","").replace("markdown","")
            update_display(Markdown(response), display_id=display_handle.display_id)

        return response
    
    # If not streaming or no display id, just return the response directly
    response = openai.chat.completions.create(
        model=model_select,
        messages=message_input
    )
    return response.choices[0].message.content

# Define the system prompt
system_prompt = "You are an expert in acoustic signal processing. \
    Answer the following question in detail in markdown."

# Here is the user prompt
prompt = """
Explain what one needs to consider when analyzing hydrophone data of an ultrasonic acoustic pulse with an unknown acoustic working frequency.
Assume that the acoustic pulse is a lithotripsy shock wave delivered in a degassed water tank maintained at 37.5°C +/- 0.5°C.
Assume that the transducer and the hydrophone are maintained in a fixed position relative to each other for all data collection.
Also assume that the hydrophone was provided with a sensitivity curve that has Sensitivity (dB re. 1V/µPa) as a function of discrete frequencies in steps of 0.5MHz.
Do not assume that the sensitivity curve provided is continuous.
First, describe how one would determine the acoustic working frequency of the acoustic pulse.
Then, describe how one would determine the acoustic pressure of the acoustic pulse:
- Include a discussion on necessary sampling frequency and the Nyquist frequency
- Include a discussion about the benefits and limitations to using a fixed sensitivity value at the acoustic working frequency versus interpolating and applying the sensitivity curve to the entire signal
- Explain in detail how one would interpolate and apply the sensitivity curve to the entire signal
- Include a discussion about signal processing techniques that can be used to improve the accuracy of the acoustic pressure measurement, including but not limited to filtering and windowing
- Include an example script in Python that demonstrates how to perform the acoustic pressure measurement using the sensitivity curve and signal processing techniques. Assume inputs are a hydrophone signal in V, a sampling frequency in Hz, and a sensitivity curve
Finally, suggest a method to validate the accuracy of the acoustic pressure measurement.
"""

# Create message    
message = create_message(system_prompt, prompt)

# Get gpt-4o-mini to answer, with streaming
gpt_response = stream_response_openai(message, MODEL_GPT, True)
# Save markdown response to .md file
with open("hydrophone_analysis_gpt.md", "w", encoding="utf-8") as file:
    file.write(gpt_response)

# Get Llama 3.2 to answer
# Note: This assumes that Ollama is running and the model is available
# llama_response = stream_response_openai(message, MODEL_LLAMA, True)
# Save markdown response to .md file
# with open("hydrophone_analysis_llama.md", "w", encoding="utf-8") as file:
#     file.write(llama_response)