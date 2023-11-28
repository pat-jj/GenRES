from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm 

device = 'cuda'

def model_name_wrapper(model_name_raw):
    model_name = ''
    if model_name_raw == 'mistral':
        model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
        
    elif model_name_raw == 'llama-7b':
        model_name = 'huggyllama/llama-7b'
        
    elif model_name_raw == 'llama-65b':
        model_name = 'huggyllama/llama-65b'
        
    elif model_name_raw == 'llama-2-7b':
        model_name = 'NousResearch/Llama-2-7b-chat-hf'
        
    elif model_name_raw == 'llama-2-70b':
        model_name = 'NousResearch/Llama-2-70b-chat-hf'
    
    
    return model_name
        

def llama_model_init(model_name, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    model.to(device)
    return tokenizer, model
    
    
def llama_model_inference(tokenizer, model, text, prompt):
    prompt = prompt.replace('$TEXT$', text)
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)

    # Tokenize the text to get the number of tokens
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)  # Number of tokens in the input text

    # Set max_new_tokens to twice the number of tokens in the text
    max_new_tokens = 8 * num_tokens

    generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = tokenizer.batch_decode(generated_ids)
    
    return decoded[0]