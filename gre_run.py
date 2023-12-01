import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
from tqdm import tqdm 
from run_models.llama import model_name_wrapper, llama_model_init, llama_model_inference, vicuna_model_inference, \
    wizardlm_model_inference, mpt_model_inferece, openchat_model_inference
from run_models.galactica import galactica_model_init, galactica_model_inference

try:
    from run_models.gpt import gpt_instruct, gpt_chat
    # from run_models.claude import claude_init, claude_chat
except:
    print('gpt or claude not installed')

# def post_processing(model, generation):
#     if model == 'mistral':
#         relation_str = generation.split('[/INST]')[-1].replace('</s>', '').strip()
#         print(relation_str)
#         return relation_str


def llama_run_model(args, tokenizer, model, dataset_file, prompt_file, output_file):
    results = {}
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    source_texts = list(dataset.keys())
    for i in tqdm(range(len(source_texts))):
        try:
            source_text = source_texts[i]
            if 'vicuna' in args.model_name:
                generation = vicuna_model_inference(tokenizer, model, source_text, prompt, device=args.device)
            elif 'wizardlm' in args.model_name:
                generation = wizardlm_model_inference(tokenizer, model, source_text, prompt, device=args.device)
            elif 'mpt' in args.model_name:
                generation = mpt_model_inferece(tokenizer, model, source_text, prompt, device=args.device)
            elif 'openchat' in args.model_name:
                generation = openchat_model_inference(tokenizer, model, source_text, prompt, device=args.device)
            else:
                generation = llama_model_inference(tokenizer, model, source_text, prompt, device=args.device)
            # relation_str = post_processing(args.model_name, generation)
            results[source_text] = generation
            if i % 20  == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=6)
        except:
            print(f'error occured at {i}', )
            continue
        
    return results

def gpt_run_model(args, dataset_file, prompt_file, output_file):
    results = {}
    with open(prompt_file, 'r') as f:
        prompt_ = f.read()
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
        
    if 'gpt-4' in args.model_name:
        gpt_func = gpt_chat
    else:
        gpt_func = gpt_instruct
    
    source_texts = list(dataset.keys())
    for i in tqdm(range(len(source_texts))):
        try:
            source_text = source_texts[i]
            prompt = prompt_.replace('$TEXT$', source_text)
            generation = gpt_func(args.model_name, prompt)
            # relation_str = post_processing(args.model_name, generation)
            results[source_text] = generation
            if i % 20  == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=6)
        except:
            print(f'error occured at {i}')
            continue
        
    return results


def claude_run_model(args, dataset_file, prompt_file, output_file):
    results = {}
    with open(prompt_file, 'r') as f:
        prompt_ = f.read()
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
        
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
        
    client = claude_init()
    
    source_texts = list(dataset.keys())
    for i in tqdm(range(len(source_texts))):
        if source_texts[i] in results:
            continue
        try:
            source_text = source_texts[i]
            prompt = prompt_.replace('$TEXT$', source_text)
            generation = claude_chat(client, prompt)
            # relation_str = post_processing(args.model_name, generation)
            results[source_text] = generation
            # save every iteration
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=6)
        except:
            print(f'error occured at {i}')
            continue
        
    return results


def galactica_run_model(args, tokenizer, model, dataset_file, prompt_file, output_file):
    results = {}
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    source_texts = list(dataset.keys())
    for i in tqdm(range(len(source_texts))):
        try:
            source_text = source_texts[i]
            generation = galactica_model_inference(tokenizer, model, source_text, prompt)
            # relation_str = post_processing(args.model_name, generation)
            results[source_text] = generation
            if i % 20  == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=6)
        except:
            print(f'error occured at {i}')
            continue
        
    return results

    
def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_family', type=str, default='llama')
    parser.add_argument('--model_name', type=str, default='mistral')
    parser.add_argument('--model_cache_dir', type=str, default='/data/pj20/.cache')
    parser.add_argument('--prompt', type=str, default='general_bag')
    parser.add_argument('--dataset', type=str, default='nyt10m')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp_id', type=str, default='1')
    
    args = parser.parse_args()
    return args

def main():
    args = construct_args()
    dataset_file = f'./datasets/{args.dataset}.json'
    prompt_file = f'./prompts/{args.prompt}.txt'
    output_file = f'./results/{args.dataset}_{args.model_name}_{args.exp_id}.json'
    
    if args.model_family == 'llama':
        model_name = model_name_wrapper(args.model_name)
        tokenizer, model = llama_model_init(model_name, args.model_cache_dir)
        results = llama_run_model(args, tokenizer, model, dataset_file, prompt_file, output_file)
    
    elif args.model_family == 'gpt':
        results = gpt_run_model(args, dataset_file, prompt_file, output_file)
        
    elif args.model_family == 'others' and args.model_name == 'claude':
        results = claude_run_model(args, dataset_file, prompt_file, output_file)
    
    elif args.model_family == 'others' and 'galactica' in args.model_name:
        tokenizer, model = galactica_model_init(args.model_name, args.model_cache_dir)
        results = galactica_run_model(args, tokenizer, model, dataset_file, prompt_file, output_file)
    
        
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=6)
    
    
if __name__ == '__main__':
    main()



