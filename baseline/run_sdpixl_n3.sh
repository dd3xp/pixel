#!/bin/bash
# SD-piXL on the recaptioned prompts, node03. The first attempt (tail of run_v7r_evals.sh, 09-14 22:01) lost all 30
# sprites to CUDA OOM within seconds on the crowded GPU7. Now: one sprite at a time, wait for 24 GB free before each
# start, up to 3 tries per sprite, skip finished ones (final_argmax.png). ~9 GPU-h per sprite.
# Launch: NEED_MB=24000 setsid nohup bash supervise.sh sdpixl_n3 7 bash baseline/run_sdpixl_n3.sh >/dev/null 2>&1 &
set -u
ROOT=/mnt/data/kw/RoundSquisheen/pixel
cd $ROOT/SD-piXL
GPU=${CUDA_VISIBLE_DEVICES:-7}
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
OUT=$ROOT/pixel/runs_out/sdpixl_recap
PROMPTS=$ROOT/pixel/runs_out/heldout3000_prompts_recap.txt
mkdir -p $OUT
for i in $(seq ${FIRST:-0} ${LAST:-29}); do
  d=$OUT/$(printf "%05d" $i)
  for try in 1 2 3; do
    find "$d" -name final_argmax.png 2>/dev/null | grep -q . && break
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)" -ge 24000 ]; do sleep 120; done
    p=$(sed -n "$((i + 1))p" $PROMPTS)
    echo "[$(date +%m%d-%H:%M)] start $i try $try :: $p"
    $PY main.py --config config.yaml --seed 0 -pt "$p" --size 16,16 -respath "$d" >> $ROOT/pixel/logs/sdpixl_recap_$i.log 2>&1 \
      || { echo "[$(date +%m%d-%H:%M)] FAILED $i try $try"; sleep 300; }
  done
done
echo SDPIXL_N3_DONE
