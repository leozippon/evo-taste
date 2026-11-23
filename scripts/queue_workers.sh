#!/usr/bin/env bash
# Start one worker per member list as GPUs become free, one list per GPU.
#
#   scripts/queue_workers.sh runs/<name> m1,m2,m3 m4,m5 ...
#
# A GPU counts as free when no process holds memory on it (below 500 MiB) and this script has not just given it
# away. The machine is shared and another user's job can claim the same card in the same minute, so one minute
# after each launch the card is checked again: if a process of another user is on it, the worker is withdrawn
# (it has not trained yet) and the list waits for the next free card. Workers are detached and resumable:
# restarting a list after a crash is the same command. Run from the repository root after `source scripts/env.sh`.
set -euo pipefail

run=$1; shift
name=$(basename "$run")
mkdir -p "logs/$name"
log="logs/$name/queue.log"
claimed=""

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | while IFS=', ' read -r idx used; do
    case " $claimed " in *" $idx "*) continue ;; esac
    if [ "$used" -lt 500 ]; then echo "$idx"; return; fi
  done
}

foreign_on_gpu() {  # prints the owner of the first process on GPU $1 that is not ours
  local uuid
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1 == i {print $2}')
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1 == u {print $2}' |
    while read -r pid; do
      owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
      if [ -n "$owner" ] && [ "$owner" != "$USER" ]; then echo "$owner"; return; fi
    done
}

for members in "$@"; do
  while true; do
    gpu=$(free_gpu || true)
    if [ -z "$gpu" ]; then sleep 60; continue; fi
    CUDA_VISIBLE_DEVICES=$gpu nohup python -m evotaste.worker run "$run" --members "$members" > /dev/null 2>> "logs/$name/stderr-gpu$gpu.log" &
    pid=$!
    sleep 60
    owner=$(foreign_on_gpu "$gpu" || true)
    if [ -n "$owner" ]; then
      # The worker defers SIGTERM while its server starts, so stop the server first.
      pkill -TERM -P "$pid" 2>/dev/null || true
      sleep 15
      kill -TERM "$pid" 2>/dev/null || true
      echo "$(date '+%F %T') gpu $gpu withdrawn (shared with $owner) members $members" | tee -a "$log"
      sleep 120
      continue
    fi
    echo "$(date '+%F %T') gpu $gpu pid $pid members $members" | tee -a "$log"
    claimed="$claimed $gpu"
    break
  done
done
