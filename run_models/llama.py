from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm 
import torch

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
    
    elif model_name_raw == 'vicuna-1.5-7b':
        model_name = 'lmsys/vicuna-7b-v1.5'

    elif model_name_raw == 'vicuna-1.5-13b':
        model_name = 'lmsys/vicuna-13b-v1.5'

    elif model_name_raw == 'vicuna-1.3-33b':
        model_name = 'lmsys/vicuna-33b-v1.3'
    
    return model_name
        

def llama_model_init(model_name, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir,
        device_map='auto',
        torch_dtype=torch.float16
        )
    return tokenizer, model

def vicuna_model_inference(tokenizer, model, text, prompt, device='cuda'):
    prompt = prompt.replace('$TEXT$', text)
    
    system_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    
    message = system_prompt + 'USER:' + prompt + '\nASSISTANT:'

    encodeds = tokenizer(message, return_tensors="pt").input_ids
    
    model_inputs = encodeds.to(device)

    # Tokenize the text to get the number of tokens
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)  # Number of tokens in the input text

    # Set max_new_tokens to twice the number of tokens in the text
    max_new_tokens = 8 * num_tokens

    generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = tokenizer.batch_decode(generated_ids)
    
    return decoded[0]    
    
def llama_model_inference(tokenizer, model, text, prompt, device='cuda'):
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