import argparse
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.matutils import kullback_leibler
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

def calculate_ts_score(data, dictionary, lda):
    all_ts_scores = []
    for source_text in tqdm(data.keys()):  
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

        ts_score = kullback_leibler(source_dist, triples_dist)
        all_ts_scores.append(ts_score)
    
    
    average_ts_score = sum(all_ts_scores) / len(all_ts_scores)
    return average_ts_score
    

def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mistral')
    parser.add_argument('--dataset', type=str, default='nyt10m')
    parser.add_argument('--exp_id', type=str, default='1')
    
    args = parser.parse_args()
    return args

def main():
    args = construct_args()
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
    
    
    