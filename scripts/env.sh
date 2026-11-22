# Source from any directory before every run:  source scripts/env.sh
# Activates the uv venv, loads .env (HF token, HF mirror endpoint, GitHub token) and points all caches at /Data2.
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
set -a
source "$_root/.env" || { set +a; return 1; }
set +a
# The home disk is nearly full: model/dataset and package caches are shared on /Data2 outside the repo.
export HF_HOME=/Data2/lzp/hf-home
export UV_CACHE_DIR=/Data2/lzp/uv-cache
# Xet-backed downloads stall through the mirror; plain HTTP downloads work.
export HF_HUB_DISABLE_XET=1
export VLLM_NO_USAGE_STATS=1
# Some reasoning-gym generators iterate Python sets of strings, so a task instance would depend on the per-process
# hash seed. evotaste.tasks refuses to import under any other value; child processes inherit it from here.
export PYTHONHASHSEED=0
# FlashInfer JIT-compiles kernels on vLLM's first start. Build them with the CUDA 13 toolkit that matches torch (cu130)
# and the driver (13.2); the nvcc first on the default PATH is CUDA 12.8 and links a second CUDA runtime.
export CUDA_HOME=/usr/local/cuda-13.2
export PATH=$CUDA_HOME/bin:$PATH
source "$_root/.venv/bin/activate" || return 1
unset _root
