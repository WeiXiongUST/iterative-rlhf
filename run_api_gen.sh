base_dir="/home/yechenlu/rsf_mistral_length_penalty_005_07ratio"
sft_model="HuggingFaceH4/mistral-7b-sft-beta"

mkdir $base_dir

i=1
next_i=$((i + 1))
#model_dir="${base_dir}/model${i}"
#model_dir="${base_dir}/model${i}"
#mkdir ${model_dir}
#mkdir ${model_dir}/data
#model_dir=/home/yechenlu/rsf_mistral_058/model1/checkpoint-198
#model_dir=weqweasdas/ratio_095_c52_model1_lr_2e6_2epoch
model_dir="/home/yechenlu/rsf_mistral_length_penalty_005_07ratio/model1"
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.api_server \
    --model ${model_dir} \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --host 127.0.0.1 --tensor-parallel-size 1 \
    --port 8000 \
& CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server \
    --model ${model_dir} \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --host 127.0.0.1 --tensor-parallel-size 1 \
    --port 8001 \
& CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.api_server \
    --model ${model_dir} \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --host 127.0.0.1 --tensor-parallel-size 1 \
    --port 8002 \
& CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.api_server \
    --model ${model_dir} \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --host 127.0.0.1 --tensor-parallel-size 1 \
    --port 8003 \

    
