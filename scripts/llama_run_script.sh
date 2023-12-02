# export CUDA_VISIBLE_DEVICES=0,5,6
DATASET=$1
MODEL_NAME=$2
PROMPT_DEMO=$3

CUDA_VISIBLE_DEVICES=5 python3 gre_run.py \
    --model_family llama \
    --model_name $MODEL_NAME \
    --model_cache_dir /data/pj20/.cache \
    --dataset $DATASET \
    --prompt $PROMPT_DEMO \
    --exp_id 1 \

# python3 gre_run.py \
#     --model_family llama \
#     --model_name llama-2-70b \
#     --model_cache_dir /data/pj20/.cache \
#     --dataset wiki20m_rand_500 \
#     --prompt general_bag \
#     --exp_id 1 \


# CUDA_VISIBLE_DEVICES=1 python3 gre_run.py \
#     --model_family llama \
#     --model_name llama-2-7b \
#     --model_cache_dir /data/pj20/.cache \
#     --dataset wiki20m_rand_500 \
#     --prompt general_bag \
#     --exp_id 1 \


# CUDA_VISIBLE_DEVICES=2 python3 gre_run.py \
#     --model_family llama \
#     --model_name llama-2-7b \
#     --model_cache_dir /data/pj20/.cache \
#     --dataset tacred_rand_800 \
#     --prompt general_sent \
#     --exp_id 1 \