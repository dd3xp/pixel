#!/bin/bash
# cycle 5 zero-training diagnostics on probe_tv: how much of the FD gap is guidance/sampler?
# Each point: 3304 samples @16 -> fair FD (raw) -> +q16.  Usage: bash baseline/diag_sampler.sh <gpu>
set -u
GPU=${1:-3}
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"
run() {  # run <tag> <extra sample_e args...>
    local TAG=$1; shift
    CKPT=workdir/probe_tv/model_latest.pt EXTRA="$*" bash baseline/eval_probe.sh "$TAG" "$GPU" 2>&1 | grep -E "FAIR FD|DONE|Error|error" | tail -n 3
    bash baseline/q16_eval.sh "$TAG" "$GPU" 2>&1 | grep -E "FAIR FD|unique|DONE|Error|error"
}
run probe_tv_cfg2   --cfg 2
run probe_tv_cfg1p5 --cfg 1.5
run probe_tv_cfg7   --cfg 7
run probe_tv_ddim50 --sampler ddim --steps 50
run probe_tv_autog2 --guide_ckpt workdir/v7_lowres/model_latest.pt --cfg 2
run probe_tv_autog4 --guide_ckpt workdir/v7_lowres/model_latest.pt --cfg 4
echo DIAG_SAMPLER_DONE
