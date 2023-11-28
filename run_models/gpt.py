
from openai import OpenAI

with open('openai_api.key', 'r') as f:
    api_key = f.read().strip()
    
    
def gpt_instruct(model, prompt):
    client = OpenAI(api_key=api_key)

    response = client.completions.create(
    model=model,
    prompt=prompt,
    )
    
    return response.choices[0].text


def gpt_chat(model, prompt):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant to extract relationships (triples) from the text."},
        {"role": "user", "content": prompt}
    ]
    )
    
    return response.choices[0].message.content
