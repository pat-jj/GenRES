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

def add_scores(original, new_scores, attribute_name):
    for text in original.keys():
        if text in new_scores:
            original[text][attribute_name] = new_scores[text]
    return original

def main():
    datasets = [
        # 'cdr', 
        # 'docred', 
        'nyt10m', 
        # 'wiki20m', 
        # 'tacred', 
        # 'wiki80'
        ]    

    # file_name = 'nyt10m_rand_500_gpt-4-1106-preview_1.json'
    # save_name = 'nyt10m_gpt4_1106.json'
    
    # file_name = 'nyt10m_rand_500_gpt-3.5_semi_1.json'
    # save_name = 'nyt10m_semi.json'
    
    file_name = 'nyt10m_rand_500_gpt-3.5_closed_1.json'
    save_name = 'nyt10m_closed.json'
    
    if os.path.exists(f'./evals/{save_name}'):
        results = json.load(open(f'./evals/{save_name}', 'r'))
    else:
        results = defaultdict(dict)
        
    gt = json.load(open('../datasets/processed/nyt10m_processed.json', 'r'))
    for text in gt.keys():
        results[text]['ground_truth'] = gt[text]
    
    for dataset_name in datasets:
        print(f'Processing {dataset_name} ...')
        
        file_to_evaluate = f'../processed_results/{file_name}'
        with open(file_to_evaluate, 'r') as f:
            data_to_evaluate = json.load(f)
        
        for text in data_to_evaluate.keys():
            results[text]['extracted_triples'] = data_to_evaluate[text]
            
        print('Computing TS ...')
        dictionary = pickle.load(open(f'./topical_process/{dataset_name}_dictionary.pkl', 'rb'))
        lda_model = pickle.load(open(f'./topical_process/{dataset_name}_lda.pkl', 'rb'))
        
        _, TS_all = calculate_ts_score(data_to_evaluate, dictionary, lda_model, output_all_scores=True)
        results = add_scores(results, TS_all, 'TS')
        json.dump(results, open(f'./evals/{save_name}', 'w'), indent=4)
        
        print('Computing US ...')
        _, US_all = calculate_uniqueness_score(data_to_evaluate, output_all_scores=True)
        results = add_scores(results, US_all, 'US')
        json.dump(results, open(f'./evals/{save_name}', 'w'), indent=4)
        
        print('Computing FS ...')
        FS_all = {}
        with open(f'./factualness/{file_name}', 'r') as f:
            FS_details = json.load(f)
        for text in FS_details.keys():
            FS_all[text] = FS_details[text]['score']
        results = add_scores(results, FS_all, 'FS')
        json.dump(results, open(f'./evals/{save_name}', 'w'), indent=4)
            
        print('Computing GS ...')
        GS_all = json.load(open(f'./granularity/{file_name}', 'r'))
        results = add_scores(results, GS_all, 'GS')
        json.dump(results, open(f'./evals/{save_name}', 'w'), indent=4)
            
        print('Computing CS ...')
        CS_all = json.load(open(f'./completeness/details/{file_name[:-7]}.json', 'r'))
        results = add_scores(results, CS_all, 'CS')
        json.dump(results, open(f'./evals/{save_name}', 'w'), indent=4)

        print(f'Finished processing {dataset_name} ...')
        
        
if __name__ == '__main__':
    main()