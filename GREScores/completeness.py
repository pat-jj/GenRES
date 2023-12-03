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

num_workers = 16

# File path for storing embeddings
embedding_file_path = '/data/pj20/grescore/triple_embeddings.json'

def load_embeddings():
    """Load embeddings from the JSON file."""
    if os.path.exists(embedding_file_path):
        with open(embedding_file_path, 'r') as file:
            return json.load(file)
    else:
        return {}

def save_embeddings(embeddings):
    """Save embeddings to the JSON file."""
    with open(embedding_file_path, 'w') as file:
        json.dump(embeddings, file)

def get_triple_embedding(triple, embeddings):
    """Get the embedding for a triple, using the API if not already in the file."""
    triple_str = ' '.join(triple)
    if triple_str not in embeddings:
        embeddings[triple_str] = embedding_retriever(triple_str)
    return embeddings[triple_str]


def load_ground_truth(dataset_name):
    """Load ground truth data from JSON file."""
    gt_file_path = f"./datasets_text2triple_string/{dataset_name}.json"
    if os.path.exists(gt_file_path):
        with open(gt_file_path, 'r') as file:
            return json.load(file)
    else:
        return {}

def calculate_completeness_score(data_to_evaluate, ground_truth, threshold=0.5):
    """Calculate the Completeness Score for a dataset."""
    completeness_scores = []
    embeddings = load_embeddings()  # Load existing embeddings

    # Pre-compute embeddings for ground truth triples
    gt_embeddings = {gt: get_triple_embedding(gt, embeddings) for gt in ground_truth}

    for text, triples in data_to_evaluate.items():
        matches = 0
        for triple in triples:
            triple_embedding = get_triple_embedding(triple, embeddings)
            # Find the best match from the ground truth
            best_match_score = max(cosine_similarity([triple_embedding], list(gt_embeddings.values())))
            if best_match_score >= threshold:
                matches += 1

        # Compute completeness score for this text
        if ground_truth:
            completeness_scores.append(matches / len(ground_truth))
    
    # Save all embeddings after processing
    save_embeddings(embeddings)

    # Calculate the overall average completeness score
    avg_completeness_score = np.mean(completeness_scores) if completeness_scores else 0
    return avg_completeness_score


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
        
        if os.path.exists(f'./results/US.json'):
            with open(f'./results/US.json', 'r') as f:
                all_scores = json.load(f)
            
        for model_name in model_names:
            for dataset_name in dataset_names:
                if model_name in all_scores[dataset_name]:
                    continue
                try:
                    file_to_evaluate = f'../processed_results/{dataset_name}_{model_name}_{args.exp_id}.json'
                    with open(file_to_evaluate, 'r') as f:
                        data_to_evaluate = json.load(f)
                    
                    print(f"Calculating US score for model {model_name} on dataset {dataset_name}...")
                    us_score = calculate_uniqueness_score(data_to_evaluate)
                    print(f"US score for model {model_name} on dataset {dataset_name}: {us_score}")
                    
                    all_scores[dataset_name][model_name] = us_score
                except Exception as e:
                    print(f"Error calculating US score for model {model_name} on dataset {dataset_name}: {e}")
                    continue
                
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
