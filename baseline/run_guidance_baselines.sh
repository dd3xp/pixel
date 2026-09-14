#!/bin/bash
# External training-free guidance baselines on v7r @16 px (recaptioned prompts, seed 0, 2 NFE each, same as ours).
# Spec: pixel_art_research_20260816/guidance_baselines_spec.md. Compare with v7r best CFG 11.28 (3 seeds),
# label 8.86, composed 8.50, and 1b 7.23 (1 NFE).
#   CDG  (Han et al., CVPR 2026, closest to ours): content tokens -> "", padding kept, replaces CFG; w 1.5/2/3
#   ICG  (Sadat et al., ICLR 2025): Gaussian random condition each step; w 1.5/2
#   ICG-label: uniformly random resolution bucket (random-label control for our lower-bucket rule); w 1.5/2
#   TSG  (Sadat et al.): time-embedding noise, SD-conditional (s3, a0.25, t>=400) w 1.5/2.5/4; DiT-conditional (s2, a1) w 2.5
#   SEG  (Hong, NeurIPS 2024): blurred queries; mid sigma=inf, d1+u1 sigma 1/2, all sigma 1; w 3
#   PAG  (Ahn et al., ECCV 2024), re-run on v7r: mid w 1.5/2
# Launch: NEED_MB=9000 setsid nohup bash supervise.sh guide_base 7 bash baseline/run_guidance_baselines.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-7}
CK=workdir/v7r/model_latest.pt
RUNS=(
  "v7r_cdg_w1p5 --cfg 1.5 --guide_mode cdg" "v7r_cdg_w2 --cfg 2 --guide_mode cdg" "v7r_cdg_w3 --cfg 3 --guide_mode cdg"
  "v7r_icg_w1p5 --cfg 1.5 --guide_mode icg" "v7r_icg_w2 --cfg 2 --guide_mode icg"
  "v7r_icglabel_w1p5 --cfg 1.5 --guide_mode icglabel" "v7r_icglabel_w2 --cfg 2 --guide_mode icglabel"
  "v7r_tsg_sd_w1p5 --cfg 1.5 --guide_mode tsg:3,0.25,400" "v7r_tsg_sd_w2p5 --cfg 2.5 --guide_mode tsg:3,0.25,400"
  "v7r_tsg_sd_w4 --cfg 4 --guide_mode tsg:3,0.25,400" "v7r_tsg_dit_w2p5 --cfg 2.5 --guide_mode tsg:2,1,0"
  "v7r_seg_midinf_w3 --cfg 3 --guide_mode seg:mid:100000" "v7r_seg_d1u1s1_w3 --cfg 3 --guide_mode seg:d1,u1:1"
  "v7r_seg_d1u1s2_w3 --cfg 3 --guide_mode seg:d1,u1:2" "v7r_seg_alls1_w3 --cfg 3 --guide_mode seg:all:1"
  "v7r_pag_mid_w1p5 --cfg 1.5 --guide_mode pag:mid" "v7r_pag_mid_w2 --cfg 2 --guide_mode pag:mid"
)
for r in "${RUNS[@]}"; do
  tag=${r%% *}; args=${r#* }
  [ -f runs_out/${tag}_matched_eval/.done ] && continue
  echo "[$(date +%m%d-%H:%M)] $tag :: $args"
  CKPT=$CK EXTRA="$args --chunk 500" bash baseline/eval_matched_r.sh $tag 16 $GPU \
    && touch runs_out/${tag}_matched_eval/.done || echo "GUIDE_BASE_FAIL $tag"
done
echo GUIDE_BASE_DONE
