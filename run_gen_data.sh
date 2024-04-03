base_dir="/home/cyeab/rsf_mistral_baseline"
sft_model="HuggingFaceH4/mistral-7b-sft-beta"

mkdir $base_dir

i=2
next_i=$((i + 1))
#model_dir="${base_dir}/model${i}"
model_dir="${base_dir}/model${i}"
mkdir ${model_dir}
mkdir ${model_dir}/data
my_world_size=8

CUDA_VISIBLE_DEVICES=0 python new_vllm.py --model_name_or_path ${model_dir} --dataset_name_or_path ${i} --output_dir ${model_dir}/data --share 0 --my_world_size ${my_world_size} &
CUDA_VISIBLE_DEVICES=1 python new_vllm.py --model_name_or_path ${model_dir} --dataset_name_or_path ${i} --output_dir ${model_dir}/data --share 1 --my_world_size ${my_world_size} &
CUDA_VISIBLE_DEVICES=2 python new_vllm.py --model_name_or_path ${model_dir} --dataset_name_or_path ${i} --output_dir ${model_dir}/data --share 2 --my_world_size ${my_world_size} &
CUDA_VISIBLE_DEVICES=3 python new_vllm.py --model_name_or_path ${model_dir} --dataset_name_or_path ${i} --output_dir ${model_dir}/data --share 3 --my_world_size ${my_world_size} &
CUDA_VISIBLE_DEVICES=4 python new_vllm.py --model_name_or_path ${model_dir} --dataset_name_or_path ${i} --output_dir ${model_dir}/data --share 4 --my_world_size ${my_world_size} &
CUDA_VISIBLE_DEVICES=5 python new_vllm.py --model_name_or_path ${model_dir} --dataset_name_or_path ${i} --output_dir ${model_dir}/data --share 5 --my_world_size ${my_world_size} &
CUDA_VISIBLE_DEVICES=6 python new_vllm.py --model_name_or_path ${model_dir} --dataset_name_or_path ${i} --output_dir ${model_dir}/data --share 6 --my_world_size ${my_world_size} &
CUDA_VISIBLE_DEVICES=7 python new_vllm.py --model_name_or_path ${model_dir} --dataset_name_or_path ${i} --output_dir ${model_dir}/data --share 7 --my_world_size ${my_world_size} &


