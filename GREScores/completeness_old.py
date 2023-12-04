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
embedding_file_path = '/data/pj20/grescore/triple_string_embeddings.json'

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


def get_triple_string_embedding(triple, dataset, embeddings):
    """Get the embedding for a triple, using the API if not already in the file."""
    term_1, relation, term_2 = triple
    if dataset == 'cdr':
        triple_str = f"The relation between \"{term_1}\" and \"{term_2}\""
    else:
        triple_str = f"The relation between \"{term_1}\" and \"{term_2}\" is \"{relation}\""
        
    if triple_str not in embeddings:
        return triple_str, embedding_retriever(triple_str)
    else:
        return triple_str, embeddings[triple_str]


def load_ground_truth_tristr2emb():
    """Load ground truth data from JSON file."""
    gt_tristr2emb_path = f"/data/pj20/triple_string_embeddings.json"
    if os.path.exists(gt_tristr2emb_path):
        with open(gt_tristr2emb_path, 'r') as file:
            return json.load(file)
    else:
        return {}


def load_ground_truth_text2tristrs(dataset_name):
    """Load ground truth data from JSON file."""
    gt_text2tristrs_path = f"./completeness/datasets_text2triple_string/{dataset_name}.json"
    if os.path.exists(gt_text2tristrs_path):
        with open(gt_text2tristrs_path, 'r') as file:
            return json.load(file)
    else:
        return {}

import threading
def calculate_completeness_score(data_to_evaluate, dataset, gt_tristr2emb, threshold=0.90):
    embeddings = load_embeddings()  # Load existing embeddings
    embeddings_lock = threading.Lock()
    text2tristrs = load_ground_truth_text2tristrs(dataset)
    completeness_scores = []
    scores_details = defaultdict(dict)

    for text, triples in tqdm(data_to_evaluate.items()):
        matches = 0
        gt_embeddings = {}
        
        if text not in text2tristrs:
            completeness_scores.append(1)
            continue

        if len(triples) == 0:
            completeness_scores.append(0)
            continue
        
        if len(text2tristrs[text]) == 0:
            completeness_scores.append(1)
            continue
        
        for tristr in text2tristrs[text]:
            gt_embeddings[tristr] = gt_tristr2emb[tristr]

        extracted_embeddings = []
        extracted_triples = []
        
        # Parallel embedding retrieval
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_triple = {executor.submit(get_triple_string_embedding, triple, dataset, embeddings): triple for triple in triples}
            for future in concurrent.futures.as_completed(future_to_triple):
                try:
                    triple_str, triple_embedding = future.result()
                    with embeddings_lock:
                        embeddings[triple_str] = triple_embedding  # Safely update embeddings

                    extracted_embeddings.append(triple_embedding)
                    extracted_triples.append(triple_str)

                except Exception as exc:
                    print(f'Error processing: {exc}')
        

        # Recall calculation
        gt_recalls = {tristr: 0 for tristr in gt_embeddings}
        for gt_tristr, gt_embedding in gt_embeddings.items():
            similarity_scores = cosine_similarity([gt_embedding], extracted_embeddings)
            best_match_score = np.max(similarity_scores)
            if best_match_score >= threshold:
                gt_recalls[gt_tristr] = 1

            # Store details
            scores_details[text][gt_tristr] = similarity_scores.tolist()[0]

        # Compute completeness score for this text
        completeness_scores.append(sum(gt_recalls.values()) / len(gt_recalls) if len(gt_recalls) > 0 else 0)

    # Save all embeddings after processing
    with embeddings_lock:
        save_embeddings(embeddings)

    avg_completeness_score = np.mean(completeness_scores) if completeness_scores else 0
    return avg_completeness_score, scores_details




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
        
        if os.path.exists(f'./results/CS.json'):
            with open(f'./results/CS.json', 'r') as f:
                all_scores = json.load(f)
                
        gt_tristr2emb = load_ground_truth_tristr2emb()
            
        for model_name in model_names:
            for dataset_name in dataset_names:
                if model_name in all_scores[dataset_name]:
                    continue
                try:
                    file_to_evaluate = f'../processed_results/{dataset_name}_{model_name}_{args.exp_id}.json'
                    with open(file_to_evaluate, 'r') as f:
                        data_to_evaluate = json.load(f)
                    
                    print(f"Calculating CS score for model {model_name} on dataset {dataset_name}...")
                    CS_score, details = calculate_completeness_score(data_to_evaluate, dataset_name.split('_')[0], gt_tristr2emb)
                    print(f"CS score for model {model_name} on dataset {dataset_name}: {CS_score}")
                    
                    all_scores[dataset_name][model_name] = CS_score
                    
                except Exception as e:
                    print(f"Error calculating CS score for model {model_name} on dataset {dataset_name}: {e}")
                    continue
                
                with open(f'./completeness/details/{dataset_name}_{model_name}.json', 'w') as f:
                    json.dump(details, f, indent=6)
                
                with open(f'./results/CS.json', 'w') as f:
                    json.dump(all_scores, f, indent=6)
            
    else:
        file_to_evaluate = f'../processed_results/{args.dataset}_{args.model_name}_{args.exp_id}.json'
        with open(file_to_evaluate, 'r') as f:
            data_to_evaluate = json.load(f)
        
        print(f"Calculating CS score for model {args.model_name} on dataset {args.dataset}...")
        CS_score, details = calculate_completeness_score(data_to_evaluate, args.dataset.split('_')[0], gt_tristr2emb)
        print(f"CS score for model {args.model_name} on dataset {args.dataset}: {CS_score}")
    
if __name__ == '__main__':
    main()
