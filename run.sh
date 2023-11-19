export CUDA_VISIBLE_DEVICES=4

python3 gre_run.py \
    --model_name mistral \
    --prompt_file ./prompts/prompt_mistral.txt  \
    --dataset_file ./datasets/nyt10m.json