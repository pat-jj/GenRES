import requests
import os

with open('../run_models/openai_api.key', 'r') as f:
    OPENAI_API_KEY = f.read().strip()
    
# Function to get embeddings from OpenAI API
def get_embedding(text):
    """
    Get the embedding for a given piece of text using the OpenAI API.
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }
    data = {
        'input': text,
        'model': 'text-embedding-ada-002'
    }
    response = requests.post('https://api.openai.com/v1/embeddings', headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['data'][0]['embedding']
    else:
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
