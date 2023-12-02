#!/bin/bash

# Define the model names and datasets in arrays
# model_names=("gpt-3.5-turbo-instruct" "text-davinci-003" "gpt-4")
model_names=("gpt-3.5-turbo-1106")

datasets=("cdr_rand_200")

# Loop over each dataset and model and call the script with the parameters
for dataset in "${datasets[@]}"; do
  for model_name in "${model_names[@]}"; do
    python3 gre_run.py \
      --model_family gpt \
      --model_name "$model_name" \
      --dataset "$dataset" \
      --prompt bio_bag \
      --exp_id 1
  done
done
