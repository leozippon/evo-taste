#!/usr/bin/env bash
# Keep the workers of a run alive on a shared machine.
#
#   scripts/supervise_run.sh runs/<name>
#
# Reads three small files under logs/<name>/ on every pass, so they can be edited while it runs:
#   lists          one comma-separated member list per line; each list is one worker
#   gpus           GPU indices this run may use, space-separated
#   worker-args    extra arguments for `worker run` (optional), e.g. --checkpoint-every 5
#   need-free-mib  optional; free memory a card must have before a worker is started on it (default 40000)
#   upgrade-after  optional; workers started before this file was last touched are stopped right after their
#                  next checkpoint, and so restart on the current code and arguments without losing steps
#
# A list whose members are not all finished and that has no live worker is started on an allowed GPU with enough
# free memory. Other users' jobs land on our cards without warning and kill a worker by exhausting memory; the
# worker is resumable, so the supervisor simply starts it again, elsewhere if need be.
# Run from the repository root after `source scripts/env.sh`. Stop with SIGTERM; workers keep running.
set -uo pipefail

run=$1
name=$(basename "$run")
dir="logs/$name"
log="$dir/supervisor.log"
declare -A seen_ckpt

say() { echo "$(date '+%F %T') $*" | tee -a "$log"; }

worker_pid() { pgrep -u "$USER" -f "^python -m evotaste.worker run $run --members $1( |\$)" | head -1; }

finished() {  # all members of the list are done (or their slot is exhausted)
  python3 - "$run" "$1" <<'EOF'
import json, os, sys
run, members = sys.argv[1], sys.argv[2].split(",")
def done(m):
    # follow replacements: the slot is finished when its current lineage is done or culled without a successor
    cur = m
    while True:
        p = os.path.join(run, "members", cur, "state.json")
        if not os.path.exists(p):
            return False
        s = json.load(open(p))
        if s["status"] == "done":
            return True
        if s["status"] == "culled":
            if not s.get("replaced_by"):
                return True
            cur = s["replaced_by"]
            continue
        return False
sys.exit(0 if all(done(m) for m in members) else 1)
EOF
}

min_ckpt() {
  python3 - "$run" "$1" <<'EOF'
import json, os, sys
run, members = sys.argv[1], sys.argv[2].split(",")
steps = []
for m in members:
    p = os.path.join(run, "members", m, "state.json")
    steps.append(json.load(open(p))["step"] if os.path.exists(p) else -1)
print(min(steps))
EOF
}

my_gpus() {  # cards that already host one of our workers
  for pid in $(pgrep -u "$USER" -f "^python -m evotaste.worker run "); do
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'
  done
}

pick_gpu() {
  local busy; busy=" $(my_gpus | tr '\n' ' ') "
  for g in $(cat "$dir/gpus"); do
    case "$busy" in *" $g "*) continue ;; esac
    need_free_mib=$(cat "$dir/need-free-mib" 2>/dev/null || echo 40000)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    if [ -n "$free" ] && [ "$free" -ge "$need_free_mib" ]; then echo "$g"; return; fi
  done
}

stop_worker() {  # the worker defers SIGTERM while its server starts, so fall back to stopping the server
  local pid=$1
  kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 24); do kill -0 "$pid" 2>/dev/null || return 0; sleep 5; done
  pkill -TERM -P "$pid" 2>/dev/null; sleep 15
  kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
}

say "supervisor started for $run"
while true; do
  while read -r members; do
    [ -z "$members" ] && continue
    pid=$(worker_pid "$members")
    if [ -n "$pid" ]; then
      if [ -f "$dir/upgrade-after" ]; then
        started=$(stat -c %Y "/proc/$pid" 2>/dev/null || echo 0)
        flagged=$(stat -c %Y "$dir/upgrade-after")
        ck=$(min_ckpt "$members")
        key="$pid"
        if [ "$started" -lt "$flagged" ]; then
          if [ -n "${seen_ckpt[$key]:-}" ] && [ "$ck" -gt "${seen_ckpt[$key]}" ]; then
            say "stopping pid $pid at checkpoint $ck for upgrade: $members"
            stop_worker "$pid"
            sleep 5
            pid=""  # restart this list at once, before a waiting list can take its card
          else
            seen_ckpt[$key]=$ck
          fi
        fi
      fi
      [ -n "$pid" ] && continue
    fi
    finished "$members" && continue
    gpu=$(pick_gpu)
    if [ -z "$gpu" ]; then continue; fi
    args=$(cat "$dir/worker-args" 2>/dev/null || true)
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu nohup python -m evotaste.worker run "$run" --members "$members" $args \
      > /dev/null 2>> "$dir/stderr-gpu$gpu.log" &
    say "started pid $! on gpu $gpu ($args): $members"
    sleep 20
  done < "$dir/lists"
  sleep 60
done
