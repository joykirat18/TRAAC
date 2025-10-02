
export HUGGINGFACE_TOKEN='None'
export HF_TOKEN='None'
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --port 8005 \
  --tensor-parallel-size 2