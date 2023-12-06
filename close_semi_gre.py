import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
from tqdm import tqdm 
from collections import defaultdict
from run_models.llama import model_name_wrapper, llama_model_init, llama_model_inference, vicuna_model_inference, \
    wizardlm_model_inference, mpt_model_inferece, openchat_model_inference, zephyr_model_inferece, chatglm3_model_inference
from run_models.galactica import galactica_model_init, galactica_model_inference

try:
    from run_models.gpt import gpt_instruct, gpt_chat
    # from run_models.claude import claude_init, claude_chat
except:
    print('gpt or claude not installed')
    

def get_rel_ent_set(dataset):
    if dataset == "cdr":
        relation_set = ["induced by", "not induced by"]
        entity_type_set = ["chemical", "disease"]
    elif dataset == "nyt10m":
        relation_set =['administrative_divisions',
                        'advisors',
                        'capital',
                        'children',
                        'company',
                        'contains',
                        'country',
                        'county_seat',
                        'ethnicity',
                        'featured_film_locations',
                        'founders',
                        'geographic_distribution',
                        'location',
                        'locations',
                        'majorshareholders',
                        'nationality',
                        'neighborhood_of',
                        'place_founded',
                        'place_lived',
                        'place_of_birth',
                        'place_of_burial',
                        'place_of_death',
                        'religion'
                        ]
        entity_type_set =  ['administrative_division',
                            'business',
                            'company',
                            'country',
                            'deceasedperson',
                            'ethnicity',
                            'event',
                            'film',
                            'location',
                            'neighborhood',
                            'people',
                            'person',
                            'region',
                            'time',
                            'us_county'
                            ]
        
    return relation_set, entity_type_set



def gpt_run_model(args, dataset_file, prompt_file, output_file):
    results = defaultdict(list)
    
    dataset_name = args.dataset.split('_')[0]
    relation_set, entity_type_set = get_rel_ent_set(dataset_name)
    
    with open(prompt_file, 'r') as f:
        prompt_ = f.read()
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
        
    gpt_func = gpt_instruct
    
    if args.type == 'closed':
        source_texts = list(dataset.keys())
        for i in tqdm(range(len(source_texts))):
            source_text = source_texts[i]
            triples = dataset[source_text]
            for triple in triples:
                subject_, object_ = triple[0], triple[2]
                prompt = prompt_.replace('$TEXT$', source_text)
                prompt = prompt.replace('$RELATION_SET$', str(relation_set))
                prompt = prompt.replace('$SUBJECT$', subject_)
                prompt = prompt.replace('$OBJECT$', object_)
                generation = gpt_func(args.model_name, prompt)
                results[source_text].append([subject_, generation, object_])
        
            if i % 20  == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=6)
        
    else:
        source_texts = list(dataset.keys())
        for i in tqdm(range(len(source_texts))):
            # try:
            source_text = source_texts[i]
            prompt = prompt_.replace('$TEXT$', source_text)
            prompt = prompt.replace('$RELATION_SET$', str(relation_set))
            prompt = prompt.replace('$ENTITY_TYPE_SET$', str(entity_type_set))
            generation = gpt_func(args.model_name, prompt)
            # relation_str = post_processing(args.model_name, generation)
            results[source_text] = generation
            if i % 20  == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=6)
        # except:
        #     print(f'error occured at {i}')
        #     continue
        
    return results


    
def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nyt10m')
    parser.add_argument('--type', type=str, default='open')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-instruct')
    parser.add_argument('--exp_id', type=str, default='1')
    
    args = parser.parse_args()
    return args

def main():
    args = construct_args()
    
    if args.type == 'closed':
        prompt_file = './prompts/close_gre.txt'
    elif args.type == 'semi':
        prompt_file = './prompts/semi_open_gre.txt'
    
    dataset_file = f'./datasets/processed/{args.dataset}_processed.json'
    output_file = f'./results/{args.dataset}_gpt-3.5_{args.type}_{args.exp_id}.json'
    
    
    results = gpt_run_model(args, dataset_file, prompt_file, output_file)
    
        
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=6)
    
    
if __name__ == '__main__':
    main()



