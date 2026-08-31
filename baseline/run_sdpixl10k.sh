#!/bin/bash
# SD-piXL baseline at the paper-fair 10k steps. Usage: run_sdpixl10k.sh <size>
S=$1
cd /mnt/data/kw/RoundSquisheen/pixel/SD-piXL
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=https://hf-mirror.com
i=0
while IFS= read -r p; do
  i=$((i+1))
  echo "[$(date +%m%d-%H:%M)] size $S prompt $i/8: $p"
  python main.py -c /mnt/data/kw/RoundSquisheen/pixel/pixel/baseline/sdpixl_db32_10k.yaml --size=$S,$S -pt "$p" 2>&1 | grep -E 'Error|Traceback' | tail -2
done < /mnt/data/kw/RoundSquisheen/pixel/pixel/baseline/prompts8.txt
echo SIZE_${S}_DONE
