#!/bin/bash
# Start every long-running job under its own detached supervisor and return.
#
# Launching these one-by-one over ssh hung the connection: a backgrounded child
# that still holds the session's stdout keeps ssh open, and the later launches
# never ran.  Doing it from a script on the node, with every child's fds closed
# explicitly, keeps the ssh call short and makes the set of jobs one atomic
# thing to read and re-run.
set -u
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel
cd "$ROOT" || exit 1
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python

start() {   # name gpu command...
  local name=$1
  if pgrep -f "supervise.sh $name " > /dev/null; then
    echo "already running: $name"
    return
  fi
  if [ -f "$ROOT/logs/${name}.done" ]; then
    echo "already finished: $name  (rm logs/${name}.done to redo)"
    return
  fi
  setsid nohup bash "$ROOT/supervise.sh" "$@" </dev/null >/dev/null 2>&1 &
  disown
  echo "started: $name (gpu $2)"
}

# --- resolution-conditioning ablation: ladder vs per-resolution specialists ---
start abl_ladder 7 "$PY" src/v6/train_v7.py --steps 20000 --sample_every 5000 \
      --only_buckets 12,16,20,24 --init workdir/abl_ladder/model_latest.pt \
      --out workdir/abl_ladder
start abl_s12 4 "$PY" src/v6/train_v7.py --steps 20000 --sample_every 5000 \
      --only_buckets 12 --init workdir/abl_s12/model_latest.pt --out workdir/abl_s12
start abl_s16 5 "$PY" src/v6/train_v7.py --steps 20000 --sample_every 5000 \
      --only_buckets 16 --init workdir/abl_s16/model_latest.pt --out workdir/abl_s16

# --- SD-piXL 10k baseline queue (24 remaining of 32) ---
start bq0 0 bash "$ROOT/baseline/bq_worker.sh" 0
start bq1 1 bash "$ROOT/baseline/bq_worker.sh" 1

# --- sentinel: names whoever SIGTERMs our GPU jobs (worked on node03) ---
start gsent 2 "$PY" src/v6/sentinel_gpu.py

# --- which machines are attached to this shared account, sampled continuously.
# The sentinel only works if the sender is still alive when the signal is
# handled, and it cannot see a SIGKILL at all; this does not depend on either.
# NEED_MB=0 because it never touches a GPU.
NEED_MB=0 start connlog 0 bash "$ROOT/connlog.sh"

# --- how much can the text-only model already draw? 100 everyday objects ---
start coverage16 6 "$PY" src/v6/coverage.py --size 16 --out runs_out/coverage16

# --- v11: fine-tune v7c on real sprites PLUS the SDXL pseudo-labels.
# Weighting is the whole question and I do not know the right value, so two runs
# differ only in it: too little and the new vocabulary is never learned, too much
# and the lower-quality pseudo-labels drag down the pixel-art craft that the real
# sprites supply.  Both must be checked against the original 8 prompts, not just
# on new objects -- a vocabulary gain paid for with quality loss is not a win.
start v11_r15 4 "$PY" src/v6/train_v7.py --steps 8000 --sample_every 2000 \
      --init workdir/v7c_bow/model_latest.pt \
      --extra data/pseudo,data/pseudo.csv,15 --out workdir/v11_r15
start v11_r45 6 "$PY" src/v6/train_v7.py --steps 8000 --sample_every 2000 \
      --init workdir/v7c_bow/model_latest.pt \
      --extra data/pseudo,data/pseudo.csv,45 --out workdir/v11_r45

# --- pseudo-labels for distillation: SDXL has the object vocabulary our corpus
# lacks, and the coverage run showed the gap is vocabulary, not rendering.
# Reuses the downscale-baseline pipeline, which already produces 12/16/20/24.
start distill 7 "$PY" src/v6/baseline_downscale.py \
      --prompts prompts/vocab_distill.txt --out runs_out/distill_v1 --n 4

sleep 3
echo "--- supervisors now running ---"
pgrep -af "supervise.sh " | sed 's/.*supervise.sh /supervise.sh /' | cut -c1-60
