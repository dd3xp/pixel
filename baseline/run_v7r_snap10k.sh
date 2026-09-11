#!/bin/bash
# Weak model for composed guidance on the recaptioned corpus. v7r saves snapshots only every 20k steps, so it has
# no 10k snapshot, which is what the composed reference uses on v7h. train_v7.py has a constant LR (no schedule),
# so a 10k-step run of the exact v7r recipe is the same training state as v7r's step 10k (up to data order).
# Launch: NEED_MB=20000 setsid nohup bash supervise.sh v7r_snap10k 2 bash baseline/run_v7r_snap10k.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
echo "[$(date +%m%d-%H:%M)] STAGE train v7r_snap10k (v7r recipe, 10k steps)"
$P src/v6/train_v7.py --steps 10000 --out workdir/v7r_snap10k --exclude runs_out/holdout_exclude.txt \
   --snap_every 5000 --csv_suffix _recap || { echo V7R_SNAP10K_FAIL; exit 1; }
echo V7R_SNAP10K_DONE
