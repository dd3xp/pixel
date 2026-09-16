#!/bin/bash
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export PYTHONNOUSERSITE=1 HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=2
/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python tmp_up/score_subset.py --size 16 --n 200 --dirs \
  A_real_floor=runs_out/heldout3000_totensor_s16 \
  B_ours_composed=runs_out/v7h_gbk12s10k_w1p5_matched_eval/s16 \
  C_ours_bestCFG=runs_out/v7h_cfg1p5_matched_eval/s16 \
  D_ours_CFGw4=runs_out/v7h_matched_eval/s16 \
  E_killed_vpred=runs_out/vpred_cfg2_matched_eval/s16 \
  F_killed_srcbucket=runs_out/src_s48_cfg2_matched_eval/s16 \
  G_killed_probe_cc=runs_out/probe_cc_coarse_w1p5_matched_eval/s16 \
  H_killed_probe_cg=runs_out/probe_cg_coarse_w1p5_matched_eval/s16 \
  I_killed_energy=runs_out/probe_es_pilot_hyb300_matched_eval/s16 \
  J_ext_gpt_image_2=runs_out/api_gpt_image_2/s16 \
  K_ext_sdxl=runs_out/sdxl_dn_eval/s16
echo SCORE_ALL_DONE
