from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm 

device = 'cuda'

def model_name_wrapper(model_name_raw):
    model_name = ''
    if model_name_raw == 'mistral':
        model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    
    return model_name
        

def llama_model_init(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/data/pj20/.cache')
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='/data/pj20/.cache')
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
    max_new_tokens = 4 * num_tokens

    generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = tokenizer.batch_decode(generated_ids)
    
    return decoded[0]