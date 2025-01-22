# Day 1 Project
# imports

import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI
import PyPDF2

# Load environment variables in a file called .env

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# Check the key

if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not api_key.startswith("sk-proj-"):
    print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif api_key.strip() != api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("API key found and looks good so far!")

openai = OpenAI()

# A class to represent a Webpage
# If you're not familiar with Classes, check out the "Intermediate Python" notebook

# Some websites need you to use proper headers when fetching them:
headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)


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
        
        # Look for inventors if they exist. If they do, add them to the text before the rest of the text
        inventors = soup.find('meta', {'name': 'DC.creator'})
        if inventors:
            self.text = f"Inventors: {inventors['content']}\n" + self.text
        
        # Debugging: find all sections
        # sections = soup.find_all('section')

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

# A function that writes a User Prompt that asks for summaries of websites:
def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\nThe contents of this website is as follows; \
please provide a short summary of this website in markdown. \
If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt

# A function that writes a User Prompt that asks for summaries of patents:
def user_prompt_for_patent(patent):
    user_prompt = "You are looking at a patent document."
    user_prompt += "\nThe contents of this patent document are as follows; \
        please provide a short summary of this document in markdown. \
        If it includes the listed inventors, a description section, or any claims, then summarize these too.\n\n"
    user_prompt += patent.text
    return user_prompt

# Function to create the messages for the OpenAI API
# Inputs: system prompt (string) 
#         document (either a Website or PatentPDF object)
# Output: a list of dictionaries formatted for the OpenAI API
def messages_for(system_prompt, document):
    if isinstance(document, Website):
        user_prompt = user_prompt_for(document)
    elif isinstance(document, GooglePatent):
        user_prompt = user_prompt_for_patent(document)
    else:
        raise ValueError("Unsupported document type")
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# And now: call the OpenAI API. You will get very familiar with this!
def summarize(url):
    patent = GooglePatent(url)
    response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages_for(
            "You are an assistant that analyzes the contents of a patent and provides a short summary, " +
            "ignoring text that might be navigation related. Respond in markdown.", patent)
    )
    return response.choices[0].message.content

def save_summary_to_markdown(url, filename):
    summary = summarize(url)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(summary)

# Example usage
patent_url = "https://patents.google.com/patent/WO2024102896A2/en"
output_filename = "patent_summary.md"

save_summary_to_markdown(patent_url, "patent_summary.md")

