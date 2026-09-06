#!/bin/bash
# Energy-score stochastic denoiser probe + paired control, matched protocol.
# Usage: bash baseline/run_probe_es.sh <gpu> <init_ckpt> <suffix> [steps]
#   pilot (contaminated, mechanism check only): bash baseline/run_probe_es.sh 2 workdir/v7_lowres/model_latest.pt _pilot 10000
#   clean:                                       bash baseline/run_probe_es.sh 2 workdir/v7h/model_latest.pt "" 20000
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
GPU=${1:-2}; INIT=$2; SUF=${3:-}; STEPS=${4:-20000}
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU
case "$INIT" in *v7h*) until [ -f logs/v7h.done ]; do sleep 120; done;; esac
for V in es ctrl; do
  NAME=probe_es${SUF}_$V
  if [ ! -f workdir/$NAME/.done ]; then
    if [ $V = es ]; then X=""; else X="--mse --m 1 --no_xi --bs_scale 1.0"; fi
    echo "[$(date +%m%d-%H:%M)] TRAIN $NAME $X"
    $P src/v6/train_es.py --init $INIT --steps $STEPS --out workdir/$NAME $X && touch workdir/$NAME/.done
  fi
  echo "[$(date +%m%d-%H:%M)] EVAL $NAME"
  bash baseline/eval_matched.sh $NAME $GPU
done
$P src/v6/fd_decomp.py --size 16 --gen runs_out/probe_es${SUF}_es_matched_eval/s16 runs_out/probe_es${SUF}_ctrl_matched_eval/s16
echo PROBE_ES${SUF}_DONE
