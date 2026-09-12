#!/bin/bash
# Candidate 1b: GFT-form internaliser of the composed cross-resolution guidance (train_gft_res.py), on v7h so the
# numbers compare with 7.53 (composed, 2 NFE + stored snapshot) / 8.52 (label, 2 NFE) / 12.49 (best CFG, 2 NFE).
# CRSC's T0/T1 both said STOP, so this is the design doc's prescribed fallback.
# Bars at ONE forward pass (--cfg 1 --gft_beta 1/w): keep as a row if <= 8.5; INTERNALISER claim if <= 7.8.
# Launch (waits for 36 GB free on GPU2, i.e. after v7r_snap10k):
#   NEED_MB=36000 setsid nohup bash supervise.sh probe_gft 2 bash baseline/run_probe_gft.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
OUT=workdir/probe_gft
if [ ! -f $OUT/.trained ]; then
  echo "[$(date +%m%d-%H:%M)] STAGE train probe_gft (ref = v7h 10k snapshot under the lower label)"
  $P src/v6/train_gft_res.py --steps 10000 --out $OUT --ref snapshot || { echo PROBE_GFT_TRAIN_FAIL; exit 1; }
  touch $OUT/.trained
fi
for w in 1.5 2; do
  b=$($P -c "print(round(1/$w, 6))")
  t=probe_gft_w$(echo $w | tr . p)
  [ -f runs_out/${t}_matched_eval/.done ] && continue
  echo "[$(date +%m%d-%H:%M)] STAGE eval 1 NFE, beta=$b (w=$w)"
  CKPT=$OUT/model_latest.pt EXTRA="--cfg 1 --gft_beta $b" bash baseline/eval_matched_r.sh $t 16 ${CUDA_VISIBLE_DEVICES:-2} \
    && touch runs_out/${t}_matched_eval/.done
done
echo PROBE_GFT_DONE
