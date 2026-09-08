#!/bin/bash
# GPU3 batch after DIAG_V7S2_DONE:
#  (15) wall-clock / peak-memory table for 3000 samples @16 (bare CFG, bucket ref, autoguidance, composed, cfg_text 3-NFE) -> logs/diag_time.txt
#  (6)  24 px guidance-weight sweep (R=24 diag_wsweep20.sh)
# Launch: NEED_MB=8000 setsid nohup bash supervise.sh diag_gpu3b 3 bash baseline/diag_gpu3b.sh 3 >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-3}; export CUDA_VISIBLE_DEVICES=$GPU
H=workdir/v7h/model_latest.pt;  HW=workdir/v7h/model_step010000.pt
head -n 1000 runs_out/heldout3000_prompts.txt > runs_out/time_prompts1000.txt
timeit() {  # <tag> <extra args>   1000 prompts x n=1 @16, chunk 500; time + peak mem via nvidia-smi poll
  local tag=$1; shift
  rm -rf runs_out/_time_$tag
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU; sleep 1; done ) > runs_out/_mem_$tag.txt &
  local mp=$!
  local t0=$(date +%s.%N)
  $P src/v6/sample_e.py --ckpt $H --buckets 12,16,20,24,32,48,64 --sizes 16 --prompts runs_out/time_prompts1000.txt --n 1 --seed 0 \
     --chunk 500 --out runs_out/_time_$tag "$@" > /dev/null 2>&1
  local t1=$(date +%s.%N)
  kill $mp 2>/dev/null
  local peak=$(sort -n runs_out/_mem_$tag.txt | tail -n 1)
  echo "TIME $tag  1000 samples @16  $($P -c "print(f'{$t1-$t0:.1f}')") s  peak_mem_MiB=$peak  :: $*" | tee -a logs/diag_time.txt
  rm -rf runs_out/_time_$tag runs_out/_mem_$tag.txt
}
echo "[$(date +%m%d-%H:%M)] timing on GPU $GPU ($(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU))" | tee -a logs/diag_time.txt
timeit warmup       --cfg 4
timeit cfg4         --cfg 4
timeit bk12_w2      --cfg 2 --guide_mode bucket:12
timeit autog10k     --cfg 1.5 --guide_ckpt $HW
timeit composed     --cfg 1.5 --guide_mode bucket:12 --guide_ckpt $HW
timeit composed_ct  --cfg 1.5 --guide_mode bucket:12 --guide_ckpt $HW --cfg_text 1.5
timeit cfg1         --cfg 1
echo DIAG_TIME_DONE
R=24 bash baseline/diag_wsweep20.sh $GPU
echo DIAG_GPU3B_DONE
