#!/usr/bin/env bash
# Run one final evaluation as shards spread over whatever GPUs have room, and wait for all of them.
#
#   scripts/run_eval.sh runs/<name> <suite> <tag> <adapters> <per-family> <samples> <per-source> <shards>
#
# An evaluation shard only needs a vLLM server (about 19 GiB with a 12 GiB KV cache), so it can share a card
# with another job. Shards are placed one per card on cards with enough free memory that do not already host
# one of our workers or shards; shards that find no card wait. A shard that dies is started again (records are
# written at the end of a shard, so it simply redoes its part). Exits 0 when every shard has finished cleanly.
# Run from the repository root after `source scripts/env.sh`.
set -uo pipefail

run=$1 suite=$2 tag=$3 adapters=$4 per_family=$5 samples=$6 per_source=$7 shards=$8
name=$(basename "$run")
dir="logs/$name"
log="$dir/eval-$suite-$tag.log"
need_free_mib=${NEED_FREE_MIB:-21000}
mkdir -p "$dir"
declare -A pid_of gpu_of done_ok tries

say() { echo "$(date '+%F %T') $*" | tee -a "$log"; }

busy_gpus() {  # cards hosting one of our workers or evaluation shards
  for pid in $(pgrep -u "$USER" -f "^python -m evotaste\.(worker|evaluate) run "); do
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'
  done
}

pick_gpu() {
  local busy; busy=" $(busy_gpus | tr '\n' ' ') "
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | while IFS=', ' read -r g free; do
    case "$busy" in *" $g "*) continue ;; esac
    if [ "$free" -ge "$need_free_mib" ]; then echo "$g"; return; fi
  done
}

launch() {
  local i=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu nohup python -m evotaste.evaluate run "$run" --suite "$suite" --adapters "$adapters" \
    --per-family "$per_family" --samples "$samples" --per-source "$per_source" --shard "$i/$shards" --tag "$tag" \
    --kv-cache 12G --gpu-memory-utilization 0.42 > /dev/null 2>> "$dir/eval-stderr-gpu$gpu.log" &
  pid_of[$i]=$!; gpu_of[$i]=$gpu; tries[$i]=$(( ${tries[$i]:-0} + 1 ))
  say "shard $i/$shards on gpu $gpu pid ${pid_of[$i]} (attempt ${tries[$i]})"
}

say "evaluation $suite tag $tag: $shards shards, per-family $per_family x $samples, per-source $per_source"
while true; do
  remaining=0
  for i in $(seq 0 $((shards - 1))); do
    [ -n "${done_ok[$i]:-}" ] && continue
    remaining=$((remaining + 1))
    pid=${pid_of[$i]:-}
    if [ -n "$pid" ]; then
      if kill -0 "$pid" 2>/dev/null; then continue; fi
      if wait "$pid"; then
        done_ok[$i]=1; remaining=$((remaining - 1)); say "shard $i finished"; continue
      fi
      say "shard $i died (attempt ${tries[$i]})"
      if [ "${tries[$i]}" -ge 6 ]; then say "shard $i failed too often; giving up"; exit 1; fi
      pid_of[$i]=""
    fi
    gpu=$(pick_gpu)
    [ -z "$gpu" ] && continue
    launch "$i" "$gpu"
    sleep 30
  done
  [ "$remaining" -eq 0 ] && break
  sleep 30
done
say "all $shards shards finished"
