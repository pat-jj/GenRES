from collections import defaultdict
import json
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
from factualness import calculate_factualness_score
from topical import calculate_ts_score
from uniqueness import calculate_uniqueness_score
from granularity import calculate_granularity_score
from completeness import calculate_completeness_score
import os



def main():
    datasets = ['cdr', 'docred', 'nyt10m', 'wiki20m', 'tacred', 'wiki80']    
    if os.path.exists('./results/ground_truth.json'):
        results = json.load(open('./results/ground_truth.json', 'r'))
    else:
        results = defaultdict(dict)
    
    for dataset_name in datasets:
        print(f'Processing {dataset_name} ...')
        
        file_to_evaluate = f'../datasets/processed/{dataset_name}_processed.json'
        with open(file_to_evaluate, 'r') as f:
            data_to_evaluate = json.load(f)
            
        if dataset_name not in results:
            results[dataset_name] = {}
        
        if 'avg_num_triples' not in results[dataset_name] or 'avg_num_tokens_per_triple' not in results[dataset_name]:
            print('Computing stats ...')
            num_triples = []
            average_num_tokens_per_triple = []
            
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
            
            results[dataset_name]['avg_num_triples'] = np.mean(num_triples)
            results[dataset_name]['avg_num_tokens_per_triple'] = np.mean(average_num_tokens_per_triple)
            json.dump(results, open('./results/ground_truth.json', 'w'), indent=4)
            
        
        if 'TS' not in results[dataset_name]:
            print('Computing TS ...')
            dictionary = pickle.load(open(f'./topical_process/{dataset_name}_dictionary.pkl', 'rb'))
            lda_model = pickle.load(open(f'./topical_process/{dataset_name}_lda.pkl', 'rb'))
            
            TS = calculate_ts_score(data_to_evaluate, dictionary, lda_model)
            results[dataset_name]['TS'] = TS
            json.dump(results, open('./results/ground_truth.json', 'w'), indent=4)
        
        if 'US' not in results[dataset_name]:
            print('Computing US ...')
            US = calculate_uniqueness_score(data_to_evaluate)
            results[dataset_name]['US'] = US
            json.dump(results, open('./results/ground_truth.json', 'w'), indent=4)
        
        if 'FS' not in results[dataset_name]:
            print('Computing FS ...')
            FS, _ = calculate_factualness_score(data_to_evaluate=data_to_evaluate)
            results[dataset_name]['FS'] = FS
            json.dump(results, open('./results/ground_truth.json', 'w'), indent=4)
            
        if 'GS' not in results[dataset_name]:
            print('Computing GS ...')
            GS, _ = calculate_granularity_score(data_to_evaluate)
            results[dataset_name]['GS'] = GS
            json.dump(results, open('./results/ground_truth.json', 'w'), indent=4)
            
        if 'CS' not in results[dataset_name]:
            print('Computing CS ...')
            CS, _ = calculate_completeness_score(data_to_evaluate, dataset_name)
            results[dataset_name]['CS'] = CS
            json.dump(results, open('./results/ground_truth.json', 'w'), indent=4)

        print(f'Finished processing {dataset_name} ...')
        
        
if __name__ == '__main__':
    main()