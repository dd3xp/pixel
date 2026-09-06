#!/bin/bash
# matched-prompt protocol sweep (held-out captions, n=1, 3000): baselines + guidance variants
set -u
GPU=${1:-2}
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"
run() { local TAG=$1; local CK=$2; shift 2; CKPT=$CK EXTRA="$*" bash baseline/eval_matched.sh "$TAG" "$GPU" 2>&1 | grep -E "FAIR FD|unique|DONE|Error|Traceback"; }
run v7_lowres            workdir/v7_lowres/model_latest.pt
run probe_tv_cfg7        workdir/probe_tv/model_latest.pt --cfg 7
run probe_tv_autog2      workdir/probe_tv/model_latest.pt --guide_ckpt workdir/v7_lowres/model_latest.pt --cfg 2
run probe_tv_cfg10       workdir/probe_tv/model_latest.pt --cfg 10
export PYTHONNOUSERSITE=1 HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$GPU
/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python src/v6/fd_decomp.py --size 16 --gen runs_out/probe_tv_matched_eval/s16 runs_out/probe_tv_matched_q16/s16 runs_out/v7_lowres_matched_eval/s16 runs_out/probe_tv_cfg7_matched_eval/s16 runs_out/probe_tv_autog2_matched_eval/s16 runs_out/probe_tv_cfg10_matched_eval/s16 2>&1 | grep -E "fd=|Error|Traceback"
echo DIAG_MATCHED_DONE
