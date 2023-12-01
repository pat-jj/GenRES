from transformers import AutoTokenizer, OPTForCausalLM
from tqdm import tqdm 
import torch

device = 'cuda'
        

def galactica_model_init(model_name, cache_dir):
    if model_name == 'galactica-6.7b':
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b", cache_dir=cache_dir)
        model = OPTForCausalLM.from_pretrained("facebook/galactica-6.7b", device_map="auto", torch_dtype=torch.float16, cache_dir=cache_dir)
    elif model_name == 'galactica-30b':
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", cache_dir=cache_dir)
        model = OPTForCausalLM.from_pretrained("facebook/galactica-30b", device_map="auto", torch_dtype=torch.float16, cache_dir=cache_dir)
    return tokenizer, model
    
    
def galactica_model_inference(tokenizer, model, text, prompt):
    # Tokenize the text to get the number of tokens
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)  # Number of tokens in the input text

    # If input tokens are more than 2000, skip the inference
    if num_tokens > 2000:
        return "The input text is too long to process."

    # Replace placeholder with actual text
    prompt = prompt.replace('$TEXT$', text)
    prompt = 'Question: ' + prompt + '\n\nAnswer:'

    # Convert the prompt to model input format
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Set max_new_tokens to twice the number of tokens in the text
    max_new_tokens = 8 * num_tokens

    # Generate the output tokens
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = tokenizer.batch_decode(generated_ids)

    # Find the position of the generated answer in the decoded text
    answer_start = decoded[0].find('Answer:') + len('Answer:')
    answer_end = len(decoded[0])
    
    # Extract and return the generated answer
    generated_answer = decoded[0][answer_start:answer_end].strip()

    return generated_answer