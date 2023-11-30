from transformers import AutoTokenizer, OPTForCausalLM
from tqdm import tqdm 
import torch

device = 'cuda'
        

def galactica_model_init(model_name, cache_dir):
    if model_name == 'galactica-6.7b':
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b", cache_dir=cache_dir)
        model = OPTForCausalLM.from_pretrained("facebook/galactica-6.7b", device_map="auto", torch_dtype=torch.float16, cache_dir=cache_dir)
        model.to(device)
    elif model_name == 'galactica-30b':
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", cache_dir=cache_dir)
        model = OPTForCausalLM.from_pretrained("facebook/galactica-30b", device_map="auto", torch_dtype=torch.float16, cache_dir=cache_dir)
        model.to(device)
    return tokenizer, model
    
    
def galactica_model_inference(tokenizer, model, text, prompt):
    prompt = prompt.replace('$TEXT$', text)
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Tokenize the text to get the number of tokens
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)  # Number of tokens in the input text

    # Set max_new_tokens to twice the number of tokens in the text
    max_new_tokens = 4 * num_tokens

    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = tokenizer.batch_decode(generated_ids)
    
    return decoded[0]