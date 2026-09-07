#!/bin/bash
# wait for the APG sweep on GPU3, then start probe_cc (contrast-deficient trained reference) under supervise.sh
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
until grep -q DIAG_APG_DONE logs/diag_apg.log 2>/dev/null; do sleep 60; done
setsid nohup bash supervise.sh probe_cc 3 env OUT=workdir/probe_cc CG_ARGS="--degrade contrast --f 0.6" bash baseline/run_probe_cg.sh >/dev/null 2>&1 &
echo launched
