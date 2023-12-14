#!/bin/bash

# Define the model names and datasets in arrays
model_names=("gpt-3.5-turbo-1106")
seeds=(54 64 74 84)

datasets=("wiki20m_rand_500")

# Loop over each dataset and model and call the script with the parameters
for dataset in "${datasets[@]}"; do
  for model_name in "${model_names[@]}"; do
    for seed in "${seeds[@]}"; do
      python3 gre_run.py \
        --model_family gpt \
        --model_name "$model_name" \
        --dataset "$dataset" \
        --prompt general_bag \
        --seed "$seed"
    done
  done
done
