import json
from openai_emb import embedding_retriever
from collections import defaultdict
from tqdm import tqdm
import threading
import os 

if not os.path.exists('/data/pj20/gre_element_embedding_dict.json'):
    element_embedding_dict = defaultdict(list)
else:
    with open('/data/pj20/gre_element_embedding_dict.json', 'r') as f:
        element_embedding_dict = json.load(f)
    
element_set = set()

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


for model_name in model_names:
    for dataset_name in dataset_names:
        file = f'../processed_results/{dataset_name}_{model_name}_1.json'
        with open(file, 'r') as f:
            data = json.load(f)
        for text in data.keys():
            triple_list = data[text]
            for triple in triple_list:
                if type(triple[0]) == list:
                    for triple_ in triple:
                        for element in triple_:
                            element_set.add(element)
                else:
                    for element in triple:
                        element_set.add(element)                  
                

for dataset_name in dataset_names:
    dataset = dataset_name.split('_')[0]
    file = f'./processed/{dataset}_processed.json'
    with open(file, 'r') as f:
        data = json.load(f)
        for text in data.keys():
            triple_list = data[text]
            for triple in triple_list:
                for element in triple:
                    element_set.add(element)
                

print("There are {} elements in total.".format(len(element_set)))
            

def process_element(elements, element_embedding_dict, lock, progress_bar, counter_lock, counter, file_path):
    for element in elements:
        if element not in element_embedding_dict:
            embedding = embedding_retriever(element)
            with lock:
                element_embedding_dict[element] = embedding
                with counter_lock:
                    counter[0] += 1
                    if counter[0] % 20000 == 0:
                        with open(file_path, 'w') as f:
                            json.dump(element_embedding_dict, f, indent=4)
            with progress_bar_lock:
                progress_bar.update(1)

# Splitting the element set into chunks for each thread
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Number of threads
num_threads = 32
element_list = list(element_set)
element_chunks = list(chunks(element_list, len(element_list) // num_threads))

# Dictionary to hold embeddings (shared resource)
lock = threading.Lock()

# Creating and starting threads
total_elements = len(element_set)
progress_bar = tqdm(total=total_elements, desc="Processing Elements")
progress_bar_lock = threading.Lock()  # Lock for thread-safe updates of the progress bar

# Creating and starting threads
counter = [0]  # Use a list to create a mutable integer that can be shared across threads
counter_lock = threading.Lock()  # Lock for thread-safe updates of the counter
file_path = '/data/pj20/gre_element_embedding_dict.json'

# Creating and starting threads
threads = []
for i in range(num_threads):
    thread = threading.Thread(target=process_element, args=(element_chunks[i], element_embedding_dict, lock, progress_bar, counter_lock, counter, file_path))
    threads.append(thread)
    thread.start()

# Waiting for all threads to complete
for thread in threads:
    thread.join()
    
progress_bar.close() 
    
with open('/data/pj20/gre_element_embedding_dict.json', 'w') as f:
    json.dump(element_embedding_dict, f, indent=4)