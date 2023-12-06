import argparse
from collections import defaultdict
import json
import os
from tqdm import tqdm
from openai import OpenAI
import time

with open('../run_models/openai_api.key', 'r') as f:
    api_key = f.read().strip()
    
with open('../prompts/fact_checker.txt', 'r') as f:
    fact_checker_prompt = f.read().strip()
    

def gpt_instruct(prompt):
    client = OpenAI(api_key=api_key)

    while True:
        try:
            response = client.completions.create(
                model='gpt-3.5-turbo-instruct',
                prompt=prompt,
                max_tokens=10,
                temperature=0.3,
            )
            return response.choices[0].text
        except Exception as e:
            print(f"Error in gpt_instruct: {e}. Retrying...")
            #delay
            time.sleep(50)


def calculate_factualness_score(data_to_evaluate):
    # Store the results
    results = {}

    for source_text, triples_list in tqdm(data_to_evaluate.items()):
        # Store the factualness results for each triple
        factualness_results = []
        
        for triple in triples_list:
            # Replace placeholders with actual source text and triple
            prompt = fact_checker_prompt.replace('$TEXT$', source_text).replace('$TRIPLE$', json.dumps(triple))
            # Get the factualness result from the GPT model
            result = gpt_instruct(prompt).strip().lower()
            
            # Check if the result is 'true' or 'false' and convert it to a boolean
            is_factual = True if result == 'true' else False
            factualness_results.append(is_factual)
        
        # Calculate the factualness score
        factualness_score = sum(factualness_results) / len(triples_list) if triples_list else 0
        # Store the score and the individual results
        results[source_text] = {
            'score': factualness_score,
            'results': factualness_results
        }
        
    avg_factualness_score = sum([result['score'] for result in results.values()]) / len(results) if results else 0
    
    # Return the dictionary of scores and results
    return avg_factualness_score, results


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
            # 'vicuna-1.5-7b',
            # 'vicuna-1.3-33b', 
            # 'llama-2-7b',
            # 'llama-2-70b',
            # 'wizardlm-70b',
            # 'text-davinci-003',
            # 'gpt-3.5-turbo-instruct',
            # 'gpt-3.5-turbo-1106',
            # 'gpt-4',
            # 'gpt-4-1106-preview',
            # 'mistral',
            # 'zephyr-7b-beta',
            # 'galactica-30b',
            # 'openchat',
            'gpt-3.5_closed',
            'gpt-3.5_semi',
            ]
        
        dataset_names = [
            # 'cdr_rand_200',
            # 'docred_rand_200',
            'nyt10m_rand_500',
            # 'wiki20m_rand_500',
            # 'tacred_rand_800',
            # 'wiki80_rand_800',
        ]
        
        if os.path.exists(f'./results/FS.json'):
            with open(f'./results/FS.json', 'r') as f:
                all_scores = json.load(f)
            
        for model_name in model_names:
            for dataset_name in dataset_names:
                if model_name in all_scores[dataset_name]:
                    continue
                try:
                    file_to_evaluate = f'../processed_results/{dataset_name}_{model_name}_{args.exp_id}.json'
                    with open(file_to_evaluate, 'r') as f:
                        data_to_evaluate = json.load(f)
                    
                    print(f"Calculating FS score for model {model_name} on dataset {dataset_name}...")
                    fs_score, results = calculate_factualness_score(data_to_evaluate)
                    try:
                        with open(f'./factualness/{dataset_name}_{model_name}_{args.exp_id}.json', 'w') as f:
                            json.dump(results, f, indent=6)
                    except Exception as e:
                        print(f"Error saving results for model {model_name} on dataset {dataset_name}: {e}")
                    print(f"FS score for model {model_name} on dataset {dataset_name}: {fs_score}")
                    
                    all_scores[dataset_name][model_name] = fs_score
                except Exception as e:
                    print(f"Error calculating FS score for model {model_name} on dataset {dataset_name}: {e}")
                    continue
                
                with open(f'./results/FS.json', 'w') as f:
                    json.dump(all_scores, f, indent=6)
            
    else:
        file_to_evaluate = f'../processed_results/{args.dataset}_{args.model_name}_{args.exp_id}.json'
        with open(file_to_evaluate, 'r') as f:
            data_to_evaluate = json.load(f)
        
        print(f"Calculating FS score for model {args.model_name} on dataset {args.dataset}...")
        fs_score, results = calculate_factualness_score(data_to_evaluate)
        try:
            with open(f'./factualness/{args.dataset}_{args.model_name}_{args.exp_id}.json', 'w') as f:
                json.dump(results, f, indent=6)
        except Exception as e:
            print(f"Error saving results for model {model_name} on dataset {args.dataset}: {e}")
        print(f"FS score for model {args.model_name} on dataset {args.dataset}: {fs_score}")
    
if __name__ == '__main__':
    main()