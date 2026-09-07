#!/bin/bash
# probe_cg: coarse-belief guidance.  Fine-tune v7h 20k steps with train_coarse.py (half the samples
# trained as their 2x2 block-averaged "coarse view" under a second label set), then evaluate under
# the matched protocol: fine-only sanity (cfg4), coarse-guided w in {1.5,2,3}, coarse + bucket:12,
# and the paired zero-training references on the SAME fine-tuned weights.
# Usage: setsid nohup bash supervise.sh probe_cg 3 bash baseline/run_probe_cg.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-3}
OUT=${OUT:-workdir/probe_cg}
STEPS=${STEPS:-20000}
mkdir -p runs_out logs
echo "[$(date +%m%d-%H:%M)] STAGE train probe_cg steps=$STEPS"
$P src/v6/train_coarse.py --init workdir/v7h/model_latest.pt --steps $STEPS --out $OUT ${CG_ARGS:-} || { echo CG_TRAIN_FAIL; exit 1; }
CK=$OUT/model_latest.pt
run() {  # <tag> <extra sample_e args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_q16/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched.sh $tag $GPU && touch runs_out/${tag}_matched_q16/.done
}
N=$(basename $OUT)
run ${N}_cfg4 --cfg 4
run ${N}_coarse_w2 --cfg 2 --guide_mode coarse
run ${N}_coarse_w1p5 --cfg 1.5 --guide_mode coarse
run ${N}_coarse_w3 --cfg 3 --guide_mode coarse
run ${N}_bk12_w2 --cfg 2 --guide_mode bucket:12
run ${N}_autog10k_w1p5 --cfg 1.5 --guide_ckpt workdir/v7h/model_step010000.pt
run ${N}_coarse_s5k_w1p5 --cfg 1.5 --guide_mode coarse --guide_ckpt $OUT/model_step005000.pt
echo "[$(date +%m%d-%H:%M)] fd_decomp"
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/${N}_*_matched_eval/s16)
echo CG_DONE
