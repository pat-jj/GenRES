import requests
import os
import json

with open('../run_models/openai_api.key', 'r') as f:
    OPENAI_API_KEY = f.read().strip()
    
    
def embedding_retriever(term):
    # Set up the API endpoint URL and request headers
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    # Set up the request payload with the text string to embed and the model to use
    payload = {
        "input": term,
        "model": "text-embedding-ada-002"
    }

    # Send the request and retrieve the response
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Extract the text embeddings from the response JSON
    embedding = response.json()["data"][0]['embedding']

    return embedding