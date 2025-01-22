# imports
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import requests
from bs4 import BeautifulSoup


# Initialization
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
MODEL = "gpt-4o-mini"
openai = OpenAI()

# Some websites need you to use proper headers when fetching them:
headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

# Global system message 
system_message = "You are a helpful assistant that is helping a biomedical engineer \
    understand the intellectual property space for their new invention. \
        When asked, you provide short summaries in layman's terms of the key claims \
            in patents related to their inquiries. Always be accurate. If you don't \
                know the answer, you can say 'I'm not sure'."

class GooglePatent:

    def __init__(self, url):
        """
        Create this GooglePatent object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.find('meta', {'name': 'DC.title'})['content'] if soup.find('meta', {'name': 'DC.title'}) else "No title found"

        self.text = ""
        
        # Debugging: find all sections
        sections = soup.find_all('section')

        # Look for an abstract section if it exists. If it does, add that to the text next
        abstract = soup.find('section', {'itemprop': 'abstract'})
        if abstract:
            self.text = self.text + "\n" + abstract.get_text(separator="\n", strip=True)

        # Look for Claims if they exist. If they do, add them to the text next
        claims = soup.find('section', {'itemprop': 'claims'})
        if claims:
            self.text = self.text + "\n" + claims.get_text(separator="\n", strip=True)

        # Look for Description if it exists. If it does, add that to the text next
        description = soup.find('section', {'itemprop': 'description'})
        if description:
            self.text = self.text + "\n" + description.get_text(separator="\n", strip=True)

        # If no text was found to this point, add all text from sections
        if not self.text: 
            for section in soup.find_all('section'):
                self.text += section.get_text(separator="\n", strip=True) + "\n"
        
        # Look at Patent Citations if they exist. If they do, add them to a citation list
        self.citations = []
        families = soup.find('section', {'itemprop': 'family'})
        # Within the families section, expect to find <tr itemprop="backwardReferencesFamily" itemscope="" repeat="">
        for citation in families.find_all('tr', {'itemprop': 'backwardReferencesFamily'}):
            # Grab just the itemprop="publicationNumber"
            citation_number = citation.find('span', {'itemprop': 'publicationNumber'}).get_text(separator="\n", strip=True)
            self.citations.append(citation_number)

# chat function
def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response = handle_tool_call(message)
        messages.append(message)
        for res in response:
            messages.append(res)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
    
    return response.choices[0].message.content

# Function to handle tool calls
def handle_tool_call(message):
    responses = []
    for tool_call in message.tool_calls:
        arguments = json.loads(tool_call.function.arguments)
        patent_numbers = arguments.get("patent_numbers")
        patent_details = get_patent_info(patent_numbers)
        patent_details_dictionary = {}
        for patent_detail in patent_details:
            if isinstance(patent_detail, dict):
                patent_details_dictionary[patent_detail["title"]] = patent_detail

        response = {
            "role": "tool",
            "content": json.dumps(patent_details_dictionary),
            "tool_call_id": tool_call.id
        }
        responses.append(response)

    return responses
    
# Function to get the patent information for one or more patent numbers
def get_patent_info(patent_numbers):
    if isinstance(patent_numbers, str):
        patent_numbers = [patent_numbers]
    
    results = []
    for patent_number in patent_numbers:
        url = get_google_patent_url(patent_number)
        print(f"Tool get_patent_info called with url: {url}")
        patent = GooglePatent(url)
        results.append({
            "title": patent.title,
            "text": patent.text,
            "citations": patent.citations
        })
    return results

# Function to get google patent url from patent number
def get_google_patent_url(patent_number):
    return f"https://patents.google.com/patent/{patent_number}/en"

# Give open_ai the information required to describe the get_patent_info function
get_patent_info_function = {
    "name": "get_patent_info",
    "description": "This function takes one or more patent numbers and returns the title, text, and citations of the patents. Call this whenever you need to get the information of one or more patents.",
    "parameters": {
        "type": "object",
        "properties": {
            "patent_numbers": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "The patent numbers to get the information for."
            },
        },
        "required": ["patent_numbers"],
        "additionalProperties": False
    },
    "returns": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the patent."
                },
                "text": {
                    "type": "string",
                    "description": "The text content of the patent."
                },
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "A list of citation numbers related to the patent."
                }
            },
            "required": ["title", "text", "citations"]
        }
    }
}

# Define the list of tools
tools = [{"type": "function", "function": get_patent_info_function}]

# Gradio Interface
gr.ChatInterface(fn=chat, type="messages").launch(inbrowser=True)

# Test the function
# originating_patent_number = "WO2024102896A2"