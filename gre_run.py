from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
from tqdm import tqdm 

device = 'cuda'

def model_name_wrapper(model_name_raw):
    model_name = ''
    if model_name_raw == 'mistral':
        model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    
    
    return model_name
        

def model_init(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model
    
    
def model_inference(tokenizer, model, text, prompt):
    prompt = prompt.replace('TEXT', text)
    
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=4096, do_sample=False)
    decoded = tokenizer.batch_decode(generated_ids)
    
    return decoded[0]


def post_processing(decoded):
    relation_str = decoded.split('[/INST]')[-1].replace('</s>', '').strip()
    relation_arr = json.loads(relation_str)
    return relation_arr

    
def run_model(tokenizer, model, dataset_file, prompt_file):
    results = {}
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    for source_text in tqdm(dataset.keys()):
        decoded = model_inference(tokenizer, model, source_text, prompt)
        relation_arr = post_processing(decoded)
        results[source_text] = relation_arr
        
    return results

    
    
def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mistral')
    parser.add_argument('--prompt_file', type=str, default='./prompts/prompt_mistral.txt')
    parser.add_argument('--dataset_file', type=str, default='./datasets/nyt10.json')


    args = parser.parse_args()
    return args

def main():
    args = construct_args()
    model_name = model_name_wrapper(args.model_name)
    tokenizer, model = model_init(model_name)
    results = run_model(tokenizer, model, args.dataset_file, args.prompt_file)
    with open('./results/nyt10_mistral.json', 'w') as f:
        json.dump(results, f)
    
    
    
if __name__ == '__main__':
    main()



