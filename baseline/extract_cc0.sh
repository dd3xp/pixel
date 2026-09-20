#!/bin/bash
# Cut the CC0 bundle into sprites once every zip is on disk.  extract_oga_v3.py marks each zip with a
# .done file, so this is resumable and a second pass only handles the parts that arrived late.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export PYTHONNOUSERSITE=1
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
while ! grep -q "OGA_CC0_FETCH_FINAL missing=0" runs/oga_cc0_fetch.log 2>/dev/null; do sleep 180; done
$P src/v6/extract_oga_v3.py data/OpenGameArt-CC0 data/oga4_cut_cc0 CC0-1.0
echo CC0_EXTRACT_DONE
find data/oga4_cut_cc0 -name "*.png" | wc -l
