from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
from tqdm import tqdm 
import torch

device = 'cuda'

def model_name_wrapper(model_name_raw):
    model_name = ''
    if model_name_raw == 'mistral':
        model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    
    return model_name
        

def model_init(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/data/pj20/.cache')
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='/data/pj20/.cache')
    model.to(device)
    return tokenizer, model
    
    
def model_inference(tokenizer, model, text, prompt):
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


def post_processing(decoded):
    relation_str = decoded.split('[/INST]')[-1].replace('</s>', '').strip()
    print(relation_str)
    # relation_arr = json.loads(relation_str)
    return relation_str

    
def run_model(args, tokenizer, model, dataset_file, prompt_file):
    results = {}
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    source_texts = list(dataset.keys())
    for i in tqdm(range(len(source_texts))):
        try:
            source_text = source_texts[i]
            decoded = model_inference(tokenizer, model, source_text, prompt)
            relation_str = post_processing(decoded)
            results[source_text] = relation_str
            if i % 20  == 0:
                with open(f'./results/nyt10m_{args.model_name}.json', 'w') as f:
                    json.dump(results, f, indent=6)
        except:
            print(f'error occured at {i}')
            continue
        
    return results

    
def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mistral')
    parser.add_argument('--prompt_file', type=str, default='./prompts/prompt_mistral.txt')
    parser.add_argument('--dataset_file', type=str, default='./datasets/nyt10m.json')
    parser.add_argument('--output_file', type=str, default='./results/nyt10m_mistral.json')

    args = parser.parse_args()
    return args

def main():
    args = construct_args()
    model_name = model_name_wrapper(args.model_name)
    tokenizer, model = model_init(model_name)
    results = run_model(args, tokenizer, model, args.dataset_file, args.prompt_file)
    with open(f'./results/nyt10m_{args.model_name}.json', 'w') as f:
        json.dump(results, f, indent=6)
    
    
if __name__ == '__main__':
    main()



