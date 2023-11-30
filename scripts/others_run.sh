DATASET=$1
MODEL_NAME=$2
PROMPT_DEMO=$3


CUDA_VISIBLE_DEVICES=6 python3 gre_run.py \
    --model_family others \
    --model_cache_dir /data/pj20/.cache \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --prompt  $PROMPT_DEMO \
    --exp_id 1 \