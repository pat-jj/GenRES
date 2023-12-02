import argparse
from collections import defaultdict
import json
import os
from tqdm import tqdm
from openai import OpenAI
import numpy as np
from multiprocessing import Pool

with open('../run_models/openai_api.key', 'r') as f:
    api_key = f.read().strip()
    
with open('../prompts/granularity_checker.txt', 'r') as f:
    granularity_checker_prompt = f.read().strip()
    

def gpt_instruct(prompt):
    client = OpenAI(api_key=api_key)

    while True:
        try:
            response = client.completions.create(
                model='gpt-3.5-turbo-instruct',
                prompt=prompt,
                max_tokens=800,
                temperature=0.3,
            )
            return response.choices[0].text
        except Exception as e:
            print(f"Error in gpt_instruct: {e}. Retrying...")


def calculate_granularity_score(data_to_evaluate):
    # Store the results and scores
    granularity_results = defaultdict(dict)
    granularity_scores = []
    
    for text, triples in tqdm(data_to_evaluate.items(), desc="Evaluating triples"):
        gs_triples = []
        if len(triples) == 0:
            granularity_scores.append(0)
            continue
        for triple in triples:
            # print(triple)
            # Construct the prompt with the current triple
            prompt = granularity_checker_prompt.replace('$TRIPLE$', json.dumps(triple))
            
            # Get the model's response
            response = gpt_instruct(prompt)
            # print(response)
            
            # Parse the response to find the number of sub-triples
            # Assuming the response format is consistent with the provided examples
            # This needs to be adjusted based on the actual response format you get from the model
            num_sub_triples = response.count('[')  # Counting each sub-triple format
            # print(num_sub_triples)
            
            # Apply the exponential decay to the number of sub-triples
            gs_triple = np.exp(-num_sub_triples)
            # print(gs_triple)
            
            gs_triples.append(gs_triple)
            
            # Store the results and the score
            granularity_results[text][json.dumps(triple)] = response
            
        granularity_score = np.mean(gs_triples)
        
        granularity_scores.append(granularity_score)
    
    # Calculate the overall granularity score for the data
    avg_granularity_score = np.mean(granularity_scores)
    
    return avg_granularity_score, granularity_results


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', type=str, default='false')
    parser.add_argument('--model_name', type=str, default='mistral')
    parser.add_argument('--dataset', type=str, default='nyt10m')
    parser.add_argument('--exp_id', type=str, default='1')
    
    args = parser.parse_args()
    return args


def main():
    args = construct_args()
    
    if args.all == 'true':
        all_scores = defaultdict(dict)
        
        model_names = [
            'vicuna-1.5-7b',
            'vicuna-1.3-33b', 
            'llama-2-7b',
            'llama-2-70b',
            'wizardlm-70b',
            'text-davinci-003',
            'gpt-3.5-turbo-instruct',
            'gpt-3.5-turbo-1106',
            'gpt-4',
            'gpt-4-1106-preview',
            'mistral',
            'zephyr-7b-beta',
            'galactica-30b',
            'openchat'
            ]
        
        dataset_names = [
            'cdr_rand_200',
            'docred_rand_200',
            'nyt10m_rand_500',
            'wiki20m_rand_500',
            'tacred_rand_800',
            'wiki80_rand_800',
        ]
        
        if os.path.exists(f'./results/GS.json'):
            with open(f'./results/GS.json', 'r') as f:
                all_scores = json.load(f)
            
        for model_name in model_names:
            for dataset_name in dataset_names:
                if model_name in all_scores[dataset_name]:
                    continue
                try:
                    file_to_evaluate = f'../processed_results/{dataset_name}_{model_name}_{args.exp_id}.json'
                    with open(file_to_evaluate, 'r') as f:
                        data_to_evaluate = json.load(f)
                    
                    print(f"Calculating GS score for model {model_name} on dataset {dataset_name}...")
                    GS_score, results = calculate_granularity_score(data_to_evaluate)
                    try:
                        with open(f'./granularity/{dataset_name}_{model_name}_{args.exp_id}.json', 'w') as f:
                            json.dump(results, f, indent=6)
                    except Exception as e:
                        print(f"Error saving results for model {model_name} on dataset {dataset_name}: {e}")
                    print(f"GS score for model {model_name} on dataset {dataset_name}: {GS_score}")
                    
                    all_scores[dataset_name][model_name] = GS_score
                except Exception as e:
                    print(f"Error calculating GS score for model {model_name} on dataset {dataset_name}: {e}")
                    continue
                
                with open(f'./results/GS.json', 'w') as f:
                    json.dump(all_scores, f, indent=6)
            
    else:
        file_to_evaluate = f'../processed_results/{args.dataset}_{args.model_name}_{args.exp_id}.json'
        with open(file_to_evaluate, 'r') as f:
            data_to_evaluate = json.load(f)
        
        print(f"Calculating GS score for model {args.model_name} on dataset {args.dataset}...")
        GS_score, results = calculate_granularity_score(data_to_evaluate)
        try:
            with open(f'./factualness/{args.dataset}_{args.model_name}_{args.exp_id}.json', 'w') as f:
                json.dump(results, f, indent=6)
        except Exception as e:
            print(f"Error saving results for model {model_name} on dataset {args.dataset}: {e}")
        print(f"GS score for model {args.model_name} on dataset {args.dataset}: {GS_score}")
    
if __name__ == '__main__':
    main()