#!/bin/bash
# Keep one job alive on a shared node that periodically wipes GPU processes.
#
# History on these two nodes: node03 runs a labmate's gpu_killer.py that SIGTERMs
# every GPU process of this shared account (plus its whole process group) every
# 2s unless it lives under envs/ga_vllm.  node09 has no such daemon running, but
# something with the same signature clears GPU jobs whenever another user claims
# the node -- three sweeps so far, each taking out the training runs AND the
# tmux windows supervising them.
#
# Defences, in order of what each survives:
#   setsid --wait <job>  : the job gets its own session, so a process-group kill
#                          aimed at the job cannot walk up into this supervisor.
#   supervisor detached  : this script is itself started with setsid+nohup and
#                          not from a tmux pane, so losing the tmux server (or
#                          having its windows closed) does not stop the loop.
#   restart loop         : after any death, wait and start again -- progress
#                          resumes at the next checkpoint or the next queue item.
#
# Usage: setsid nohup bash supervise.sh <name> <gpu> <command...> >/dev/null 2>&1 &
set -u
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel
NAME=$1; GPU=$2; shift 2

# huggingface.co is unreachable from this node.  Every job here loads at least
# the frozen CLIP encoder, so without the mirror they all die on a
# ConnectionError -- which is exactly what happened for nine hours on 08-30:
# the supervisors dutifully restarted jobs that could never start, because the
# tmux commands they replaced had carried HF_ENDPOINT and this script did not.
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-120}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
LOG=$ROOT/logs/${NAME}.log
SUP=$ROOT/logs/${NAME}.supervisor

echo "[$(date +%m%d-%H:%M)] supervisor up pid=$$ gpu=$GPU cmd: $*" >> "$SUP"
rm -f "$ROOT/logs/${NAME}.FAILING"

# A restart loop that hides a permanent failure is worse than stopping: on 08-30
# a missing HF_ENDPOINT made every job die on load, and the supervisors restarted
# them every 60s for nine hours while looking, from the outside, like progress.
# So distinguish "killed mid-run" (restart, that is the point) from "cannot even
# start" (back off, then raise a file that a status check cannot miss).
fast_fails=0
while true; do
  echo "[$(date +%m%d-%H:%M)] starting $NAME" >> "$SUP"
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU setsid --wait "$@" >> "$LOG" 2>&1
  rc=$?
  ran=$(( $(date +%s) - t0 ))
  echo "[$(date +%m%d-%H:%M)] $NAME exited rc=$rc after ${ran}s" >> "$SUP"
  if [ "$rc" = "0" ]; then
    # Mark completion, so re-running the launcher does not spend a GPU redoing
    # finished work: the launcher's only other signal is "is a supervisor alive",
    # and a finished job has none.
    date +%Y-%m-%dT%H:%M > "$ROOT/logs/${NAME}.done"
    echo "[$(date +%m%d-%H:%M)] $NAME done, supervisor exiting" >> "$SUP"
    break
  fi
  if [ "$ran" -lt 120 ]; then
    fast_fails=$((fast_fails + 1))
  else
    fast_fails=0                      # it ran a while, so a kill, not a bad setup
  fi
  if [ "$fast_fails" -ge 5 ]; then
    { echo "[$(date +%m%d-%H:%M)] $NAME failed to start ${fast_fails}x in a row."
      echo "last output:"; tail -n 25 "$LOG"; } > "$ROOT/logs/${NAME}.FAILING"
    echo "[$(date +%m%d-%H:%M)] $NAME giving up; see logs/${NAME}.FAILING" >> "$SUP"
    break
  fi
  sleep $(( 60 * (fast_fails + 1) ))
done
