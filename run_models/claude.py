from claude_api import Client
import os

def claude_init():
    with open('claude_cookie.key', 'r') as f:
        cookie = f.read().strip()
        
    client = Client(cookie)
    return client
    


def claude_chat(client, prompt):
    conversation_id = client.create_new_chat()['uuid']
    response = client.send_message(prompt, conversation_id)
    deleted = client.delete_conversation(conversation_id)
    if not deleted:
        print("Failed to delete conversation")
    return response