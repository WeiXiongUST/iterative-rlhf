#!/bin/bash

base_dir="/home/xiongwei/rsf_gemma_2b_058"
sft_model="HuggingFaceH4/mistral-7b-sft-beta"

mkdir $base_dir

i=2
next_i=$((i + 1))
model_dir="${base_dir}/model${i}"
mkdir ${model_dir}
mkdir ${model_dir}/data
for (( j=1; j<=8; j++ )); do   
  python ./gen_data_vllm.py --tokenizer ${sft_model} --dataset_name_or_path ${i} --output_dir ${model_dir}/data/gen_data${j}.json --K 1 --max_new_tokens 2048 --seed ${j}
done

read -p "按回车键继续..."
python ./transform.py --dataset_path ${model_dir}/data/gen_data --my_idx ${i}
accelerate launch ./get_reward.py --dataset_name_or_path ${model_dir}/data/gen_data.json --output_dir ${model_dir}/data --record_dir ${base_dir}/reward_record.txt 
#accelerate launch /home/xiongwei/rsf/get_kl.py --model_name_or_path weqweasdas/rsf_plus_gemma2b_iter1 --dataset_name_or_path ${model_dir}/data/data_with_rewards.json --output_dir ${model_dir}/data --record_dir ${base_dir}/reward_record.txt 
python write_config.py  ${sft_model} ${base_dir}/model${next_i} ./config_full_mistral.yaml ${base_dir}/model${i}/data/top1_data.json

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ./deepspeed_zero3.yaml ./weighted_sft.py ./config_full_mistral.yaml




