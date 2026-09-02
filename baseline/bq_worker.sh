#!/bin/bash
# SD-piXL 10k baseline worker.  Pulls jobs from queue.txt, one at a time.
#
# Failure modes this has had to survive, in the order we met them:
#   * the disk hit 99%, so a run finished its 10001 steps and then could not
#     write final_argmax.png            -> verify the output, retry once
#   * hf-mirror read timeouts killed runs seconds after launch
#                                       -> long HF timeouts
#   * something on the node SIGTERMs every job we start, repeatedly
#                                       -> heartbeat locks, so a killed job's
#                                          claim is reclaimable; the caller
#                                          wraps this in a restart loop
set -u
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel
GPU=$1
Q=$ROOT/baseline/queue.txt
RES=$ROOT/baseline/results
LOCK=$ROOT/baseline/locks
STALE=600                      # a claim whose heartbeat is this old is dead
mkdir -p "$RES" "$LOCK"
export PYTHONNOUSERSITE=1   # node03: a torch in the shared ~/.local shadows the env torch (torchvision::nms missing)
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ETAG_TIMEOUT=120 HF_HUB_DOWNLOAD_TIMEOUT=120
# CUDA_VISIBLE_DEVICES is set by supervise.sh, which also gates on free memory;
# exporting it again here would re-index inside an already-restricted view.
# $GPU is kept as a label for logs and temp files.

log() { echo "[$(date +%m%d-%H:%M) gpu$GPU] $*"; }

HB_PID=""
start_beat() {                 # prove this claim is still alive
  ( while true; do touch "$1/beat" 2>/dev/null; sleep 60; done ) &
  HB_PID=$!
}
stop_beat() { [ -n "$HB_PID" ] && kill "$HB_PID" 2>/dev/null; HB_PID=""; }
trap 'stop_beat' EXIT

claim() {                      # $1=lockdir -> 0 if we now own it
  mkdir "$1" 2>/dev/null && { touch "$1/beat"; return 0; }
  local beat="$1/beat"
  [ -f "$beat" ] || return 1
  local age=$(( $(date +%s) - $(stat -c %Y "$beat" 2>/dev/null || echo 0) ))
  if [ "$age" -gt "$STALE" ]; then
    log "stealing stale claim $(basename "$1") (heartbeat ${age}s old)"
    touch "$beat"; return 0
  fi
  return 1
}

run_one() {                    # size idx prompt -> 0 if the png landed in $RES
  local size=$1 idx=$2 prompt=$3
  local before=$(date +%s)
  cd $ROOT/../SD-piXL || return 1
  setsid --wait python main.py -c $ROOT/baseline/sdpixl_db32_10k.yaml \
      --size=$size,$size -pt "$prompt" > /tmp/bq_${GPU}_${size}_${idx}.out 2>&1
  # Take the directory from the run's own output rather than guessing by
  # timestamp: with several workers on one queue, "newest matching directory"
  # can belong to somebody else's run.
  local d
  d=$(grep -aoE 'Configuration written to [^ ]+/config\.yaml' \
        /tmp/bq_${GPU}_${size}_${idx}.out 2>/dev/null | tail -1 \
        | sed 's|Configuration written to ||; s|/config\.yaml$||')
  if [ -z "$d" ] || [ ! -d "$d" ]; then      # fall back to the old heuristic
    d=$(find $ROOT/baseline/sdpixl_db32_10k -maxdepth 1 -type d -name "*im${size}x${size}*" \
          -newermt "@$before" 2>/dev/null | sort | tail -1)
  fi
  if [ -n "$d" ] && [ -f "$d/final_argmax.png" ]; then
    cp "$d/final_argmax.png" "$RES/10k_s${size}_p${idx}.png"; return 0
  fi
  log "  no output; tail of python log:"; tail -3 /tmp/bq_${GPU}_${size}_${idx}.out | cut -c1-120
  return 1
}

while true; do
  claimed=""
  while IFS='|' read -r size idx prompt; do
    [ -z "${size:-}" ] && continue
    [ -f "$RES/10k_s${size}_p${idx}.png" ] && continue
    if claim "$LOCK/${size}_${idx}"; then claimed="$size|$idx|$prompt"; break; fi
  done < "$Q"
  [ -z "$claimed" ] && { log "queue drained or fully claimed"; break; }

  IFS='|' read -r size idx prompt <<< "$claimed"
  avail=$(df --output=avail -BG /mnt/data | tail -1 | tr -dc 0-9)
  if [ "$avail" -lt 30 ]; then
    log "HOLD: only ${avail}G free, not starting s${size} p${idx}"
    rm -rf "$LOCK/${size}_${idx}"; sleep 600; continue
  fi
  start_beat "$LOCK/${size}_${idx}"
  log "start s${size} p${idx}: $prompt (${avail}G free)"
  if run_one "$size" "$idx" "$prompt"; then
    log "OK    s${size} p${idx}"
  else
    log "RETRY s${size} p${idx}"
    run_one "$size" "$idx" "$prompt" && log "OK    s${size} p${idx} (2nd)" \
      || { log "FAIL  s${size} p${idx} after 2 attempts"; rm -rf "$LOCK/${size}_${idx}"; }
  fi
  stop_beat
done
