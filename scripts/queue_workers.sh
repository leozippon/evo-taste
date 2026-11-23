#!/usr/bin/env bash
# Start one worker per member list as GPUs become free, one list per GPU.
#
#   scripts/queue_workers.sh runs/<name> m1,m2,m3 m4,m5 ...
#
# A GPU counts as free when no process holds memory on it (below 500 MiB) and this script has not just given it
# away; the machine is shared, so the check is repeated for every launch. Workers are detached and resumable:
# restarting a list after a crash is the same command. Run from the repository root after `source scripts/env.sh`.
set -euo pipefail

run=$1; shift
name=$(basename "$run")
mkdir -p "logs/$name"
claimed=""

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | while IFS=', ' read -r idx used; do
    case " $claimed " in *" $idx "*) continue ;; esac
    if [ "$used" -lt 500 ]; then echo "$idx"; return; fi
  done
}

for members in "$@"; do
  while true; do
    gpu=$(free_gpu || true)
    [ -n "$gpu" ] && break
    sleep 60
  done
  CUDA_VISIBLE_DEVICES=$gpu nohup python -m evotaste.worker run "$run" --members "$members" > /dev/null 2>&1 &
  echo "$(date '+%F %T') gpu $gpu pid $! members $members" | tee -a "logs/$name/queue.log"
  claimed="$claimed $gpu"
  sleep 5
done
