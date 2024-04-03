# iterative-rlhf

We alternate among the following steps:
- generate data;
- compute reward;
- filter reward;
- training

Step 1: modify the base_dir in run_rsf_mistral.sh and set i = 0

Step 2: bash run_api_gen.sh in one terminal and bash run_rsf_mistral.sh to run the following line

```sh
python ./gen_data_vllm.py --tokenizer ${sft_model} --dataset_name_or_path ${i} --output_dir ${model_dir}/data/gen_data.json --K 8 --max_new_tokens 2048 
```

Step 3: bash run_rsf_mistral.sh to run the following line
```sh
accelerate launch ./get_reward.py --dataset_name_or_path ${model_dir}/data/gen_data.json --output_dir ${model_dir}/data --record_dir ${base_dir}/reward_record.txt 
```

Step 4: python filter_samples_reward_kl.py to get the samples

Step 5: training bash run_rsf_mistral.sh to run the following line

```sh
python write_config.py  ${base_dir}/model${i} ${base_dir}/model${next_i} ./config_full_mistral.yaml ${base_dir}/model${i}/data/top1_data.json

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ./deepspeed_zero3.yaml ./weighted_sft.py ./config_full_mistral.yaml
```
