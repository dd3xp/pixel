#!/bin/bash
# baseline ladder on GPU0: sizes 12, 20, 24 (16 already done)
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
for S in 12 20 24; do bash baseline/run_sdpixl.sh $S; done
echo BASE_CHAIN_DONE
