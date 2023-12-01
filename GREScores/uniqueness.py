import json
import os
import argparse
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai_emb import get_embedding

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
    triple_str = ' '.join(triple)  # Convert the triple to a string
    if triple_str not in embeddings:
        embeddings[triple_str] = get_embedding(triple_str)  # Replace with your actual API call
        save_embeddings(embeddings)  # Save updated embeddings
    return embeddings[triple_str]

def calculate_uniqueness(triples):
    """Calculate the Uniqueness Score for a list of triples."""
    embeddings = load_embeddings()
    vectors = [get_triple_embedding(triple, embeddings) for triple in triples]
    n = len(vectors)
    score = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
            score += (1 - sim)
    total_pairs = n * (n - 1) / 2
    return score / total_pairs if total_pairs > 0 else 0


def calculate_uniqueness_score(data_to_evaluate):
    """Calculate the Uniqueness Score for a dataset."""
    scores = []
    for source_text, triples in data_to_evaluate.items():
        score = calculate_uniqueness(triples)
        scores.append(score)
    
    avg_score = np.mean(scores)
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
            # ... your model names
        ]
        
        dataset_names = [
            # ... your dataset names
        ]
        
        for model_name in model_names:
            for dataset_name in dataset_names:
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
