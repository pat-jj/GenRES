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

print("Loading Ground Truth triple embeddings...")
gt_triple_emb_store = {}
gt_relation_emb_store = {}
for dataset in ['cdr', 'docred', 'nyt10m', 'wiki20m', 'tacred', 'wiki80']:
    with open(f'../datasets/processed/{dataset}_processed.json', 'r') as f:
            gt_text_triples = json.load(f)
    
    for text in gt_text_triples.keys():
        gt_triple_list = gt_text_triples[text]
        for triple in gt_triple_list:
            triple_str = str(triple)
            entity_emb = np.add(ELE_EMB_DICT[triple[0]], ELE_EMB_DICT[triple[2]])
            triple_emb = np.add(np.array(entity_emb), np.array(ELE_EMB_DICT[triple[1]]))
            # emb_ = np.concatenate([ELE_EMB_DICT[triple[0]], ELE_EMB_DICT[triple[1]]])
            # triple_emb = np.concatenate([emb_, ELE_EMB_DICT[triple[2]]])
            gt_triple_emb_store[triple_str] = triple_emb.tolist()
            gt_relation_emb_store[triple_str] = ELE_EMB_DICT[triple[1]]
            

import threading
def calculate_completeness_score(data_to_evaluate, dataset, model_name=None, threshold=0.95):
        
    completeness_scores = []
    scores_details = defaultdict(dict)
    with open(f'../datasets/processed/{dataset}_processed.json', 'r') as f:
            gt_text_triples = json.load(f)

    for text, triples in tqdm(data_to_evaluate.items()):
        
        if len(triples) == 0:
            completeness_scores.append(0)
            continue
        
        if text not in gt_text_triples.keys():
            continue
            
        if len(gt_text_triples[text]) == 0:
            completeness_scores.append(1)
            continue
        
        # if type(triples[0][0]) == list:
        #     triples = triples[0]
        # else:
        #     triples = triples
            
        gt_triples = gt_text_triples[text]
        gt_embeddings = {str(triple): gt_triple_emb_store[str(triple)] for triple in gt_triples}
        # Recall calculation
        gt_recalls = {gt_triple: 0 for gt_triple in gt_embeddings.keys()}
        
        extracted_triple_embeddings = []
        extracted_relation_embeddings = []
        for triple in triples:
            try:
                entity_emb = np.add(ELE_EMB_DICT[triple[0]], ELE_EMB_DICT[triple[2]])
                triple_emb = np.add(entity_emb, ELE_EMB_DICT[triple[1]])
                # emb_ = np.concatenate([ELE_EMB_DICT[triple[0]], ELE_EMB_DICT[triple[1]]])
                # triple_emb = np.concatenate([emb_, ELE_EMB_DICT[triple[2]]])
                extracted_triple_embeddings.append(triple_emb.tolist())
                extracted_relation_embeddings.append(ELE_EMB_DICT[triple[1]])
            except:
                continue
        if len(extracted_triple_embeddings) == 0:
            continue
        for gt_triple, gt_embedding in gt_embeddings.items():
            similarity_scores = cosine_similarity([gt_embedding], extracted_triple_embeddings)
            best_match_score = np.max(similarity_scores)
            best_match_index = np.argmax(similarity_scores)
            if best_match_score >= threshold:
                if model_name == 'gpt-3.5_closed':
                    extracted_relation_emb = extracted_relation_embeddings[best_match_index]
                    relation_similarity_score = cosine_similarity([gt_relation_emb_store[gt_triple]], [extracted_relation_emb])
                    if relation_similarity_score >= threshold:
                        gt_recalls[gt_triple] = 1
                else:
                    gt_recalls[gt_triple] = 1

            # Store details
            scores_details[text][gt_triple] = similarity_scores.tolist()[0]

        # Compute completeness score for this text
        completeness_scores.append(sum(gt_recalls.values()) / len(gt_recalls) if len(gt_recalls) > 0 else 0)
        

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
            'openchat',
            'gpt-3.5_closed',
            'gpt-3.5_semi',
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
        
        # if os.path.exists(f'./results/CS.json'):
        #     with open(f'./results/CS.json', 'r') as f:
        #         all_scores = json.load(f)
                            
        for model_name in model_names:
            for dataset_name in dataset_names:
                for seed in seeds:
                    if f'{model_name}-{seed}' in all_scores[dataset_name]:
                        continue
                    # try:
                    file_to_evaluate = f'../processed_results/{dataset_name}_{model_name}_{seed}.json'
                    try:
                        with open(file_to_evaluate, 'r') as f:
                            data_to_evaluate = json.load(f)
                    except:
                        continue
                    print(f"Calculating CS score for model {model_name} on dataset {dataset_name}...")
                    CS_score, details = calculate_completeness_score(data_to_evaluate, dataset_name.split('_')[0], model_name)
                    print(f"CS score for model {model_name} on dataset {dataset_name}: {CS_score}")
                    
                    all_scores[dataset_name][f'{model_name}-{seed}'] = CS_score
                        
                    # except Exception as e:
                    #     print(f"Error calculating CS score for model {model_name} on dataset {dataset_name}: {e}")
                    #     continue
                    
                    with open(f'./completeness/details/{dataset_name}_{model_name}.json', 'w') as f:
                        json.dump(details, f, indent=6)
                    
                    with open(f'./results_new/CS.json', 'w') as f:
                        json.dump(all_scores, f, indent=6)
            
    else:
        file_to_evaluate = f'../processed_results/{args.dataset}_{args.model_name}_{args.exp_id}.json'
        with open(file_to_evaluate, 'r') as f:
            data_to_evaluate = json.load(f)
        
        print(f"Calculating CS score for model {args.model_name} on dataset {args.dataset}...")
        CS_score, details = calculate_completeness_score(data_to_evaluate, args.dataset.split('_')[0])
        print(f"CS score for model {args.model_name} on dataset {args.dataset}: {CS_score}")
    
if __name__ == '__main__':
    main()
