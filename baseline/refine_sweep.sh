#!/bin/bash
# The strength dial, measured rather than asserted: two points do not make the trade-off curve the text
# claims.  sigma = 1.0 is also a self-check -- it discards the input entirely, so it should reproduce our
# own samples for these captions (50.8 native FD).
# Usage: bash baseline/refine_sweep.sh <gpu> [system]
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
GPU=${1:-6}; SYS=${2:-gpt}
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROMPTS=runs_out/ext200/v8n/icg2_pal4/prompts.txt
for st in 0.2 0.3 0.5 0.8 1.0; do
  t=refine_${SYS}_s${st/./p}
  d=runs_out/refine/${t}
  if [ ! -f $d/.done ]; then
    CUDA_VISIBLE_DEVICES=$GPU $P src/v6/sample_e.py --ckpt workdir/v8n/model_latest.pt \
      --buckets 12,16,20,24,32,48,64 --sizes 16 --prompts $PROMPTS --n 1 --seed 0 \
      --guide_mode icg --cfg 2 --pal 4 --chunk 200 \
      --init_dir runs_out/ext200/q/${SYS}_q4/s16 --init_strength $st \
      --out $d && touch $d/.done || continue
  fi
  CUDA_VISIBLE_DEVICES=$GPU $P src/v6/fd_fair.py --size 16 --native \
    --gen $d/s16 --out runs_out/fair_fd16_native.json
done
$P src/v6/refine_fidelity.py --init runs_out/ext200/q/${SYS}_q4/s16 \
  --ours runs_out/ext200/v8n/icg2_pal4/s16 \
  --runs runs_out/refine/refine_${SYS}_s0p2/s16 runs_out/refine/refine_${SYS}_s0p3/s16 \
         runs_out/refine/refine_${SYS}_s0p4/s16 runs_out/refine/refine_${SYS}_s0p5/s16 \
         runs_out/refine/refine_${SYS}_s0p6/s16 runs_out/refine/refine_${SYS}_s0p8/s16 \
         runs_out/refine/refine_${SYS}_s1p0/s16
echo REFINE_SWEEP_DONE
