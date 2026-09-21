#!/bin/bash
# The palette size k was chosen on the same held-out FD the paper reports, which the limitations
# section admits.  This removes the objection instead of admitting it: split the 3,000 held-out
# captions into two halves by index parity, choose k on half A, report it on half B.  No new sampling
# is needed -- the runs already exist, only the scoring is redone on disjoint subsets.
# Usage: bash baseline/k_split.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
GPU=${1:-6}
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for k in 3 4 5 6 8; do
  src=runs_out/v8n_icg2_pal${k}_matched_eval/s16
  [ -d "$src" ] || { echo "missing $src"; continue; }
  for h in 0 1; do
    d=runs_out/ksplit/pal${k}_h${h}
    if [ ! -d "$d" ]; then
      mkdir -p $d
      for f in $src/*.png; do
        n=$(basename $f); i=${n%%_*}; i=${i#0}
        [ -z "$i" ] && i=0
        if [ $((10#$i % 2)) -eq $h ]; then ln -sf ../../../$f $d/$n; fi
      done
    fi
    CUDA_VISIBLE_DEVICES=$GPU $P src/v6/fd_fair.py --size 16 --native \
      --gen $d --out runs_out/fair_fd16_native_ksplit.json
  done
done
echo KSPLIT_DONE
