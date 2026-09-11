#!/bin/bash
# probe_mdm2 (A1b): the discrete family's second and last probe before the whole direction is written off.
# A1 died of ZERO-DOMINATION, not of discreteness: 62% of tokens are 0 because most pixels are transparent, and
# at full masking it predicted zero for 98.5% of visible-pixel RGB (accuracy 0.002). Giving it the true silhouette
# made it worse (466 vs 418), so the anchor was never the problem.
# A1b removes the zero mass from the objective: RGB is supervised only where the ground-truth pixel is opaque
# (invisible RGB is zeroed at render time anyway), and decoding resolves the silhouette first, then fills colour
# only inside it. If this still fails, the discrete family is closed with a clean explanation.
# Bar: v7h best-CFG 12.49, current best 7.53, floor 3.45. External: gpt-image-2 131.63, SDXL 225.11.
# Launch: NEED_MB=20000 setsid nohup bash supervise.sh probe_mdm2 2 bash baseline/run_probe_mdm2.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
STEPS=${STEPS:-40000}
OUT=${OUT:-workdir/probe_mdm2}
echo "[$(date +%m%d-%H:%M)] STAGE train probe_mdm2 (visible-only RGB supervision) steps=$STEPS"
$P src/v6/train_mdm.py --steps $STEPS --out $OUT --exclude runs_out/holdout_exclude.txt \
   --bs_scale 0.15 --snap_every 20000 --visible_only || { echo PROBE_MDM2_TRAIN_FAIL; exit 1; }
echo "[$(date +%m%d-%H:%M)] STAGE eval (alpha-first decoding)"
for cfg in "s32_e2 --steps 32 --edit_rounds 2 --alpha_first" "s32_g --steps 32 --edit_rounds 2 --alpha_first --greedy"; do
  tag=mdm2_$(echo $cfg | cut -d' ' -f1); args=$(echo $cfg | cut -d' ' -f2-)
  [ -f runs_out/${tag}_matched_eval/.done ] && continue
  $P src/v6/sample_mdm.py --ckpt $OUT/model_latest.pt --size 16 --out runs_out/${tag}_matched_eval $args \
    && $P src/v6/fd_fair.py --size 16 --gen runs_out/${tag}_matched_eval/s16 --out runs_out/fair_fd16.json \
    && touch runs_out/${tag}_matched_eval/.done
done
echo PROBE_MDM2_DONE
