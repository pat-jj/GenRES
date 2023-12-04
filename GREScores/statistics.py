import argparse
from collections import defaultdict
import json
from nltk.tokenize import word_tokenize
import numpy as np

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
        
        num_triples_all = defaultdict(dict)
        average_num_tokens_per_triple_all = defaultdict(dict)
        
        for model_name in model_names:
            for dataset_name in dataset_names:
                print(f'Processing {model_name} on {dataset_name}')
                num_triples = []
                average_num_tokens_per_triple = []

                file_to_evaluate = f'../processed_results/{dataset_name}_{model_name}_{args.exp_id}.json'
                with open(file_to_evaluate, 'r') as f:
                    data_to_evaluate = json.load(f)
                
                for text, triple_list in data_to_evaluate.items():
                    num_triples.append(len(triple_list))
                    tokens_per_triple = []
                    for triple in triple_list:
                        num_tokens = 0
                        head, relation, tail = triple
                        if type(head) == str and type(relation) == str and type(tail) == str:
                            num_tokens += len(word_tokenize(head))
                            num_tokens += len(word_tokenize(relation))
                            num_tokens += len(word_tokenize(tail))
                            tokens_per_triple.append(num_tokens)
                        else:
                            tokens_per_triple.append(0)
                        
                    average_num_tokens_per_triple.append(sum(tokens_per_triple) / len(tokens_per_triple) if len(tokens_per_triple) > 0 else 0)
                
                num_triples_all[dataset_name][model_name] = np.mean(num_triples)
                average_num_tokens_per_triple_all[dataset_name][model_name] = np.mean(average_num_tokens_per_triple)
                
                
        with open(f'./results/average_num_triples.json', 'w') as f:
            json.dump(num_triples_all, f, indent=4)
        
        with open(f'./results/average_num_tokens_per_triple.json', 'w') as f:
            json.dump(average_num_tokens_per_triple_all, f, indent=4)
                
    
if __name__ == '__main__':
    main()