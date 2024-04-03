CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --host 127.0.0.1 --tensor-parallel-size 1 \
    --port 8000 \
& CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --host 127.0.0.1 --tensor-parallel-size 1 \
    --port 8001 \
& CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --host 127.0.0.1 --tensor-parallel-size 1 \
    --port 8002 \
& CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --host 127.0.0.1 --tensor-parallel-size 1 \
    --port 8003 \
& CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --host 127.0.0.1 --tensor-parallel-size 1 \
    --port 8004 \
& CUDA_VISIBLE_DEVICES=5 python -m vllm.entrypoints.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --host 127.0.0.1 --tensor-parallel-size 1 \
    --port 8005 \
& CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --host 127.0.0.1 --tensor-parallel-size 1 \
    --port 8006 \
& CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --host 127.0.0.1 --tensor-parallel-size 1 \
    --port 8007 \

