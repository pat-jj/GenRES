export CUDA_VISIBLE_DEVICES=6

# Read arguments
DATASET=$1
MODEL_NAME=$2
PROMPT_DEMO=$3
seeds=(54 64 74 84)

# Loop over seeds and run the Python script
for seed in "${seeds[@]}"; do
    python3 gre_run.py \
        --model_family llama \
        --model_name "$MODEL_NAME" \
        --model_cache_dir /data/pj20/.cache \
        --dataset "$DATASET" \
        --prompt "$PROMPT_DEMO" \
        --seed $seed \
        --exp_id 1
done