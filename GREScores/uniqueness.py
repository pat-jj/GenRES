import json
import os
import argparse
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from openai_emb import embedding_retriever
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import concurrent


print("Loading element embeddings...")
with open('/data/pj20/gre_element_embedding_dict.json', 'r') as f:
    ELE_EMB_DICT = json.load(f)
    
    
def get_triple_embedding(triple):
    entity_emb = np.add(ELE_EMB_DICT[triple[0]], ELE_EMB_DICT[triple[2]])
    triple_emb = np.add(entity_emb, ELE_EMB_DICT[triple[1]])
    return triple_emb.tolist()


def calculate_uniqueness(vectors, phi=0.95):
    """Calculate the Uniqueness Score using cosine similarity and a threshold."""
    similarity_matrix = cosine_similarity(vectors)
    np.fill_diagonal(similarity_matrix, 1)  # Ignore self-similarity
    
    # Count pairs with cosine similarity smaller than the threshold
    count_smaller_than_phi = np.sum(similarity_matrix < phi)
    
    total_pairs = len(vectors) * (len(vectors) - 1)
    return count_smaller_than_phi / total_pairs if total_pairs > 0 else 1


def calculate_uniqueness_for_text(triples):
    """Calculate the uniqueness score for a batch of texts."""
    vectors = []
    for triple in triples:
        try:
            vectors.append(get_triple_embedding(triple))
        except:
            continue
    
    try:
        return calculate_uniqueness(np.array(vectors))
    except:
        return 1


def calculate_uniqueness_score(data_to_evaluate, output_all_scores=False):
    """Calculate the Uniqueness Score for a dataset using multi-threading."""
    scores = {}
    num_threads = 16

    def process_triples(triples):
        # if no triples, return 0
        if len(triples) == 0:
            return 1
        return calculate_uniqueness_for_text(triples)

    # with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     tasks = [executor.submit(process_triples, triples) for _, triples in data_to_evaluate.items()]
    #     for future in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks)):
    #         scores.append(future.result())
    
    for text, triples in tqdm(data_to_evaluate.items()):
        scores[text] = process_triples(triples)

    avg_score = np.mean(list(scores.values()))
    if output_all_scores:
        return avg_score, scores
    return avg_score


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
            ]
        
        dataset_names = [
            # 'cdr_rand_200',
            # 'docred_rand_200',
            # 'nyt10m_rand_500',
            'wiki20m_rand_500',
            # 'tacred_rand_800',
            # 'wiki80_rand_800',
        ]
        
        seeds = [54, 64, 74, 84]
        
        if os.path.exists(f'./results/US.json'):
            with open(f'./results/US.json', 'r') as f:
                all_scores = json.load(f)
            
        for model_name in model_names:
            for dataset_name in dataset_names:
                for seed in seeds:
                    if f'{model_name}-{seed}' in all_scores[dataset_name]:
                        continue
                    
                    file_to_evaluate = f'../processed_results/{dataset_name}_{model_name}_{seed}.json'
                    
                    try:
                        with open(file_to_evaluate, 'r') as f:
                            data_to_evaluate = json.load(f)
                    except:
                        continue
                    
                    print(f"Calculating US score for model {model_name} on dataset {dataset_name}...")
                    us_score = calculate_uniqueness_score(data_to_evaluate)
                    print(f"US score for model {model_name} on dataset {dataset_name}: {us_score}")
                    
                    all_scores[dataset_name][f'{model_name}-{seed}'] = us_score

                    
                    with open(f'./results/US.json', 'w') as f:
                        json.dump(all_scores, f, indent=6)
                    
    else:
        file_to_evaluate = f'../processed_results/{args.dataset}_{args.model_name}_{args.exp_id}.json'
        with open(file_to_evaluate, 'r') as f:
            data_to_evaluate = json.load(f)
        
        print(f"Calculating US score for model {args.model_name} on dataset {args.dataset}...")
        us_score = calculate_uniqueness_score(data_to_evaluate)
        print(f"US score for model {args.model_name} on dataset {args.dataset}: {us_score}")
    
if __name__ == '__main__':
    main()
