import argparse
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.matutils import kullback_leibler
from collections import defaultdict
import math
import nltk
import json
import pickle
import os
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [word for word in words if word not in stop_words and word.isalnum()]

def calculate_ts_score(data, dictionary, lda, output_all_scores=False):
    all_ts_scores = {}
    for source_text in data.keys():  
        triples = data[source_text]
        triples_str = ''
        for triple in triples:
            triples_str += f"{triple[0]} {triple[1]} {triple[2]} ."
        processed_source = preprocess(source_text)
        processed_triples = preprocess(triples_str)
        source_corpus = dictionary.doc2bow(processed_source)
        triples_corpus = dictionary.doc2bow(processed_triples)

        source_dist = lda.get_document_topics(source_corpus, minimum_probability=0)
        triples_dist = lda.get_document_topics(triples_corpus, minimum_probability=0)

        ts_score = math.exp(-kullback_leibler(source_dist, triples_dist))
        all_ts_scores[source_text] = ts_score
    
    average_ts_score = sum(all_ts_scores.values()) / len(all_ts_scores)
    
    if output_all_scores:
        return average_ts_score, all_ts_scores
    
    return average_ts_score
    

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
        
        for model_name in model_names:
            for dataset_name in dataset_names:
                for seed in seeds:
                    try:
                        file_to_evaluate = f'../processed_results/{dataset_name}_{model_name}_{seed}.json'
                        with open(file_to_evaluate, 'r') as f:
                            data_to_evaluate = json.load(f)
                        dataset = dataset_name.split('_')[0]
                        dictionary = pickle.load(open(f'./topical_process/{dataset}_dictionary.pkl', 'rb'))
                        lda_model = pickle.load(open(f'./topical_process/{dataset}_lda.pkl', 'rb'))
                        
                        print(f"Calculating TS score for model {model_name} on dataset {dataset}...")
                        ts_score = calculate_ts_score(data_to_evaluate, dictionary, lda_model)
                        print(f"TS score for model {model_name} on dataset {dataset}: {ts_score}")
                        
                        all_scores[dataset_name][f'{model_name}-{seed}'] = ts_score
                    except:
                        continue
                    
                with open(f'./results_new/TS.json', 'w') as f:
                    json.dump(all_scores, f, indent=6)
            
    else:
        file_to_evaluate = f'../processed_results/{args.dataset}_{args.model_name}_{args.exp_id}.json'
        with open(file_to_evaluate, 'r') as f:
            data_to_evaluate = json.load(f)
        dataset = args.dataset.split('_')[0]
        dictionary = pickle.load(open(f'./topical_process/{dataset}_dictionary.pkl', 'rb'))
        lda_model = pickle.load(open(f'./topical_process/{dataset}_lda.pkl', 'rb'))
        
        print(f"Calculating TS score for model {args.model_name} on dataset {args.dataset}...")
        ts_score = calculate_ts_score(data_to_evaluate, dictionary, lda_model)
        print(f"TS score for model {args.model_name} on dataset {args.dataset}: {ts_score}")
    

if __name__ == '__main__':
    main()
    
    
    