# GenRES: Evaluate Generative Relation Extraction
This repository is the official implementation of [GenRES: Rethinking Evaluation for Generative Relation Extraction in the Era of Large Language Models](https://arxiv.org/abs/2402.10744). 

**We will release the pip installer of GenRES soon!**
## Requirements and Installation

## Get Started
Run Vicuna
```[bash]
bash scripts/llama_run_script.sh wiki20m_rand_500 vicuna-1.5-7b general_bag
bash scripts/llama_run_script.sh wiki20m_rand_500 vicuna-1.3-33b general_bag
```

Run GPT
```[bash]
bash scripts/gpt_run.sh docred_rand_200 gpt-3.5-turbo-instruct general_bag
bash scripts/gpt_run.sh docred_rand_200 gpt-4 general_bag
bash scripts/gpt_run.sh docred_rand_200 text-davinci-003 general_bag
```

Run Claude
```[bash]
bash scripts/others_run.sh cdr_rand_200 claude bio_bag
bash scripts/others_run.sh docred_rand_200 claude general_bag
```

Run Galactica
```[bash]
bash scripts/others_run.sh cdr_rand_200 galactica-6.7b bio_bag
bash scripts/others_run.sh docred_rand_200 galactica-30b general_bag
```


Post-processing
```[bash]
post_process.ipynb
```
