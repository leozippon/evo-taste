#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server with runtime multi-LoRA loading, in the foreground.
# Usage: CUDA_VISIBLE_DEVICES=<one gpu> scripts/dev/serve_vllm.sh <hf-model-id> <port> <kv-cache-size, e.g. 12G> [extra vllm flags]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
: "${CUDA_VISIBLE_DEVICES:?pin exactly one GPU with CUDA_VISIBLE_DEVICES}"
model=$1 port=$2 kv=$3
shift 3
# Enables POST /v1/load_lora_adapter and /v1/unload_lora_adapter.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# --kv-cache-memory-bytes: a fixed KV budget instead of --gpu-memory-utilization, whose sizing measures free memory
#   and is corrupted when another process (a co-located trainer, other users) allocates during vLLM's startup.
# --generation-config vllm: do not inherit the checkpoint's sampling defaults (Qwen3: top_k=20, top_p=0.95),
#   so a request with only temperature=1.0 samples from the full softmax that the trainer scores.
# --enable-prompt-tokens-details: usage.prompt_tokens_details.cached_tokens reports prefix-cache hits per request.
exec vllm serve "$model" --host 127.0.0.1 --port "$port" \
  --kv-cache-memory-bytes "$kv" --max-model-len 8192 \
  --generation-config vllm --enable-prompt-tokens-details \
  --enable-lora --max-lora-rank 64 --max-loras 8 --max-cpu-loras 64 \
  "$@"
