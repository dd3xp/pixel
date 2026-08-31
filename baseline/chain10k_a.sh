#!/bin/bash
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
for S in 12 16; do bash baseline/run_sdpixl10k.sh $S; done
echo CHAIN10K_A_DONE
