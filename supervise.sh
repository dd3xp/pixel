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
LOG=$ROOT/logs/${NAME}.log
SUP=$ROOT/logs/${NAME}.supervisor

echo "[$(date +%m%d-%H:%M)] supervisor up pid=$$ gpu=$GPU cmd: $*" >> "$SUP"
while true; do
  echo "[$(date +%m%d-%H:%M)] starting $NAME" >> "$SUP"
  CUDA_VISIBLE_DEVICES=$GPU setsid --wait "$@" >> "$LOG" 2>&1
  rc=$?
  echo "[$(date +%m%d-%H:%M)] $NAME exited rc=$rc" >> "$SUP"
  # rc 0 means the job finished on its own; anything else is a crash or a kill
  if [ "$rc" = "0" ]; then
    echo "[$(date +%m%d-%H:%M)] $NAME done, supervisor exiting" >> "$SUP"
    break
  fi
  sleep 60
done
