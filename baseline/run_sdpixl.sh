#!/bin/bash
# Run SD-piXL baseline on 8 MC prompts at a given size. Usage: run_sdpixl.sh 16
S=$1
cd /mnt/data/kw/RoundSquisheen/pixel/SD-piXL
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=https://hf-mirror.com
i=0
while IFS= read -r p; do
  i=$((i+1))
  echo "[$(date +%H:%M)] prompt $i/8 size $S: $p"
  python main.py -c /mnt/data/kw/RoundSquisheen/pixel/pixel/baseline/sdpixl_db32_3k.yaml --size=$S,$S -pt "$p" 2>&1 | grep -E 'Error|Traceback|step.*3000|Saved' | tail -2
done < /mnt/data/kw/RoundSquisheen/pixel/pixel/baseline/prompts8.txt
echo ALL_DONE
