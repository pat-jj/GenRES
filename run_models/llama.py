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

    elif model_name_raw == 'vicuna-1.3-33b':
        model_name = 'lmsys/vicuna-33b-v1.3'
    
    elif model_name_raw == 'wizardlm-7b':
        model_name = 'WizardLM/WizardLM-7B-V1.0'
    
    elif model_name_raw == 'wizardlm-70b':
        model_name = 'WizardLM/WizardLM-70B-V1.0'

    elif model_name_raw == 'mpt-7b':
        model_name = 'mosaicml/mpt-7b-chat'

    elif model_name_raw == 'mpt-30b':
        model_name = 'mosaicml/mpt-30b-chat'
    
    elif model_name_raw == 'openchat':
        model_name = 'openchat/openchat_3.5'

    return model_name
    

def llama_model_init(model_name, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir,
        device_map='auto',
        torch_dtype=torch.float16
        )
    return tokenizer, model

def mpt_model_inferece(tokenizer, model, text, prompt, device='cuda'):
    prompt = prompt.replace('$TEXT$', text)
    
    system_message = """- You are a helpful assistant chatbot trained by MosaicML.
        - You answer questions.
        - You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
        - You are more than just an information source, you are also able to write poetry, short stories, and make jokes."""

    system_template = f"""<|im_start|>system
        {system_message}"""
    
    message = system_template + '<|im_start|>user ' + prompt + '\n<|im_start|>assistant'

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

def wizardlm_model_inference(tokenizer, model, text, prompt, device='cuda'):
    prompt = prompt.replace('$TEXT$', text)
    
    system_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    )
    
    message = system_prompt + 'USER: ' + prompt + '\nASSISTANT: '

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

def vicuna_model_inference(tokenizer, model, text, prompt, device='cuda'):
    prompt = prompt.replace('$TEXT$', text)
    
    system_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
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

def openchat_model_inference(tokenizer, model, text, prompt, device='cuda'):
    prompt = prompt.replace('$TEXT$', text)
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    encodeds = tokenizer.apply_chat_template(messages,  return_tensors="pt", add_generation_prompt=True)
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