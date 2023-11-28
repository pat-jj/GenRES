from transformers import AutoTokenizer, OPTForCausalLM
from tqdm import tqdm 

device = 'cuda'
        

def galactica_model_init(cache_dir):
    tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b", cache_dir=cache_dir)
    model = OPTForCausalLM.from_pretrained("facebook/galactica-6.7b", device_map="auto", cache_dir=cache_dir)
    model.to(device)
    return tokenizer, model
    
    
def galactica_model_inference(tokenizer, model, text, prompt):
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