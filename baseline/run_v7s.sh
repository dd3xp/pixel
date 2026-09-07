#!/bin/bash
# v7s = CLEAN second model for the paper (the existing second model v7_lowres is memorisation-contaminated):
# v7h recipe (same data, same clean hold-out exclusion) but a narrower UNet (--width 96: blocks 96/192/384,
# ~0.56x params), a different seed, 60k steps, EMA snapshots every 5k.  Then the four main 16 px rows under the
# matched protocol: bare CFG w4 / autoguidance (own step10k) w1.5 / bucket:12 w2 / composed (step10k + bucket:12) w1.5.
# Launch: setsid nohup bash supervise.sh v7s 3 bash baseline/run_v7s.sh >/dev/null 2>&1 &   (from ssh, < /dev/null)
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
STEPS=${STEPS:-60000}
OUT=${OUT:-workdir/v7s}
GPU=${CUDA_VISIBLE_DEVICES:-3}
mkdir -p runs_out logs
EX=runs_out/holdout_exclude.txt
[ -s $EX ] || { echo "missing $EX (run_v7h.sh creates it)"; exit 1; }
if [ ! -f $OUT/model_latest.pt ] || [ ! -f $OUT/.train_done ]; then
  echo "[$(date +%m%d-%H:%M)] STAGE train v7s steps=$STEPS width=96 seed=1"
  $P src/v6/train_v7.py --steps $STEPS --out $OUT --exclude $EX --snap_every 5000 --width 96 --seed 1 || { echo V7S_TRAIN_FAIL; exit 1; }
  touch $OUT/.train_done
fi
CK=$OUT/model_latest.pt
WK=$OUT/model_step010000.pt
run() {  # <tag> <extra sample_e args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_q16/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched.sh $tag $GPU && touch runs_out/${tag}_matched_q16/.done
}
run v7s_cfg4            --cfg 4
run v7s_gbk12_w2        --cfg 2 --guide_mode bucket:12
run v7s_autog10k_w1p5   --cfg 1.5 --guide_ckpt $WK
run v7s_gbk12s10k_w1p5  --cfg 1.5 --guide_mode bucket:12 --guide_ckpt $WK
run v7s_gbk20_w2        --cfg 2 --guide_mode bucket:20
echo "[$(date +%m%d-%H:%M)] fd_decomp"
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/v7s_*_matched_eval/s16)
echo V7S_DONE
