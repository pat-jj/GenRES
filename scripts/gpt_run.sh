DATASET=$1
MODEL_NAME=$2
PROMPT_DEMO=$3


python3 gre_run.py \
    --model_family gpt \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --prompt  $PROMPT_DEMO \
    --exp_id 1 \

# python3 gre_run.py \
#     --model_family gpt \
#     --model_name text-davinci-003 \
#     --dataset nyt10m_rand_500 \
#     --prompt general_bag \
#     --exp_id 1 \

# python3 gre_run.py \
#     --model_family gpt \
#     --model_name gpt-4 \
#     --dataset nyt10m_rand_500 \
#     --prompt general_bag \
#     --exp_id 1 \