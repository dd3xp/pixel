#!/bin/bash
# 1b (GFT-form internalisation) on a v8 model.  The 09-16/17 data rebuild changed the target distribution,
# so every method number has to be re-established on it: on the native protocol v8f2's plain CFG (123.10)
# already beats the old model's best stack (1b+ICG 131.18), and the open question is whether the guidance
# still adds on top once the targets are real pixel art.
# Recipe is unchanged from run_gft_v7r.sh (10k steps, beta ~ U(0.4,1), 25% plain, reference = an early
# snapshot queried under the LOWER bucket); only init/snapshot/eval tags differ.
# Launch: NEED_MB=34000 setsid nohup bash supervise.sh gft_v8f2 6 env BASE=workdir/v8f2 TAG=v8f2 bash baseline/run_gft_v8.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-6}
BASE=${BASE:-workdir/v8f2}; TAG=${TAG:-v8f2}
SNAP=${SNAP:-$BASE/model_step020000.pt}          # the "weak" reference model: this run's own early EMA
OUT=workdir/gft_$TAG
if [ ! -f $OUT/.trained ]; then
  echo "[$(date +%m%d-%H:%M)] STAGE train 1b on $TAG (init $BASE, snapshot $SNAP)"
  $P src/v6/train_gft_res.py --steps 10000 --out $OUT --ref snapshot --init $BASE/model_latest.pt \
     --snap $SNAP --csv_suffix _recap || { echo GFT_V8_TRAIN_FAIL; exit 1; }
  touch $OUT/.trained
fi
for spec in "${TAG}_gft_w1p5 16 0" "${TAG}_r20_gft_w1p5 20 0" "${TAG}_r24_gft_w1p5 24 0" "${TAG}_r12_gft_w1p5 12 0"; do
  set -- $spec
  [ -f runs_out/$1_matched_eval/.done ] && continue
  CKPT=$OUT/model_latest.pt EXTRA="--cfg 1 --gft_beta 0.666667 --seed $3 --chunk 500" \
    PROMPTS=runs_out/heldout3000_prompts_recap.txt bash baseline/eval_matched_r.sh $1 $2 $GPU \
    && touch runs_out/$1_matched_eval/.done || echo "GFT_V8_EVAL_FAIL $1"
  CUDA_VISIBLE_DEVICES=$GPU $P src/v6/fd_fair.py --size $2 --native --gen runs_out/$1_matched_eval/s$2 \
    --out runs_out/fair_fd$2_native.json
done
echo GFT_V8_${TAG}_DONE
