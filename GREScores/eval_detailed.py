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
    model_names = [
        # 'vicuna-1.5-7b',
        # 'vicuna-1.3-33b', 
        # 'llama-2-7b',
        'llama-2-70b',
        # 'wizardlm-70b',
        # 'text-davinci-003',
        # 'gpt-3.5-turbo-instruct',
        # 'gpt-3.5-turbo-1106',
        # 'gpt-4',
        'gpt-4-1106-preview',
        # 'mistral',
        # 'zephyr-7b-beta',
        # 'galactica-30b',
        'openchat',
        # 'gpt-3.5_closed',
        # 'gpt-3.5_semi',
        'groundtruth'
        ]
    
    dataset_names = [
        # 'cdr_rand_200',
        # 'docred_rand_200',
        # 'nyt10m_rand_500',
        # 'wiki20m_rand_500',
        # 'tacred_rand_800',
        # 'wiki80_rand_800',
        'wiki20m_rand_100',
    ]
    
    # seeds = [54, 64, 74, 84]
    seeds=[1]
        
    results = defaultdict(dict)
    
    for dataset_name in dataset_names:
        ds = dataset_name.split('_')[0]
        dictionary = pickle.load(open(f'./topical_process/{ds}_dictionary.pkl', 'rb'))
        lda_model = pickle.load(open(f'./topical_process/{ds}_lda.pkl', 'rb'))
        
        for model_name in model_names:
            print(f'Processing {dataset_name} ...')            
            file_to_evaluate = f'../processed_results/{dataset_name}_{model_name}_1.json'
            with open(file_to_evaluate, 'r') as f:
                data_to_evaluate = json.load(f)
                             
            print('Computing TS ...')
            TS, text2TS = calculate_ts_score(data_to_evaluate, dictionary, lda_model, output_all_scores=True)
            
            print('Computing US ...')
            US, text2US = calculate_uniqueness_score(data_to_evaluate, output_all_scores=True)
            
            print('Computing FS ...')
            FS, details = calculate_factualness_score(data_to_evaluate=data_to_evaluate, dataset_name=dataset_name, model_name=model_name, seed=1)
            text2FS = {}
            for text, res in details.items():
                text2FS[text] = res['score']
                
            print('Computing GS ...')
            GS, _, text2GS = calculate_granularity_score(data_to_evaluate, dataset_name=dataset_name, model_name=model_name, seed=1, output_all_scores=True)
                
            print('Computing CS ...')
            CS, _, text2CS = calculate_completeness_score(data_to_evaluate, ds, output_all_scores=True)
                
                
            print("Gathering data ...")
            for text in data_to_evaluate.keys():
                results[text]['TS'] = text2TS[text]
                results[text]['US'] = text2US[text]
                results[text]['FS'] = text2FS[text]
                try:
                    results[text]['GS'] = text2GS[text]
                except:
                    results[text]['GS'] = 1
                try:
                    results[text]['CS'] = text2CS[text]
                except:
                    results[text]['CS'] = 0


            print(f'Finished processing {dataset_name} - {model_name} ...')
            
            with open(f'./results_new/{dataset_name}_{model_name}_detailed.json', 'w') as f:
                json.dump(results, f, indent=4)
            
        
if __name__ == '__main__':
    main()