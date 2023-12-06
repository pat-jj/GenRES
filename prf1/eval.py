import json
from collections import defaultdict

def calculate_prf1_score(data_to_evaluate, dataset):
        
    precision = []
    recall = []
    f1 = []
    
    scores_details = defaultdict(dict)
    with open(f'../datasets/processed/{dataset}_processed.json', 'r') as f:
            gt_text_triples = json.load(f)
            
    for text, triples in data_to_evaluate.items():
        try:
            gt_triples = gt_text_triples[text]
        except:
            continue
        try:
            pred_triple_str = [str(triple).lower().replace('_', ' ') for triple in triples]
            gt_triple_str = [str(triple).lower().replace('_', ' ') for triple in gt_triples]
            p, r, f = calculate_metrics(pred_triple_str, gt_triple_str)
            precision.append(p)
            recall.append(r)
            f1.append(f)
        except:
            continue
    
    return sum(precision)/len(precision), sum(recall)/len(recall), sum(f1)/len(f1)


def calculate_metrics(predicted_relations, ground_truth_relations):
    prec, rec = 0, 0

    # Count correct predictions
    for pred in predicted_relations:
        if pred in ground_truth_relations:
            prec += 1

    for gt in ground_truth_relations:
        if gt in predicted_relations:
            rec += 1
    
    precision = prec / len(predicted_relations)
    recall = rec / len(ground_truth_relations)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1



model_names = [
    'gpt-3.5_closed',
    'gpt-3.5_semi',
    'gpt-3.5-turbo-instruct',
    ]

dataset_names = [
    'cdr_rand_200',
    'nyt10m_rand_500',
]

for model_name in model_names:
    for dataset_name in dataset_names:
        file_to_evaluate = f'../processed_results/{dataset_name}_{model_name}_1.json'
        with open(file_to_evaluate, 'r') as f:
            data_to_evaluate = json.load(f)
        
        precision, recall, f1 = calculate_prf1_score(data_to_evaluate, dataset_name.split('_')[0])

        print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1} for {model_name} on {dataset_name}")
