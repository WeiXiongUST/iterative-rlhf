#!/bin/bash

base_dir="/home/xiongwei/fix_vllm/rsf_mistral_07"
sft_model="HuggingFaceH4/mistral-7b-sft-beta"

mkdir $base_dir

i=1
next_i=$((i + 1))
model_dir="${base_dir}/ratio_095_c52_model1_lr_2e6_2epoch"
mkdir ${model_dir}
mkdir ${model_dir}/data
infer_dir=HuggingFaceH4/zephyr-7b-beta
#weqweasdas/ratio_095_c52_model1_lr_2e6_2epoch
#python ./gen_data_vllm.py --tokenizer ${infer_dir} --dataset_name_or_path ${i} --output_dir ${model_dir}/data/gen_data.json --K 1 --max_new_tokens 2048 

read -p "按回车键继续..."
#python ./new_transform.py --dataset_path ${model_dir}/data/gen_data --my_idx ${i} --output_dir ${model_dir}/gen_data
#accelerate launch ./get_reward.py --dataset_name_or_path ${model_dir}/data/gen_data.json --output_dir ${model_dir}/data --record_dir ${base_dir}/reward_record.txt 
#accelerate launch ./get_kl.py --model_name_or_path ${model_dir} --dataset_name_or_path ${model_dir}/data/data_with_rewards.json --output_dir ${model_dir}/data --record_dir ${base_dir}/reward_record.txt 
python write_config.py  ${sft_model} ${base_dir}/model${next_i} ./config_full_mistral.yaml ${model_dir}/data/top1_data.json

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ./deepspeed_zero3.yaml ./weighted_sft.py ./config_full_mistral.yaml




