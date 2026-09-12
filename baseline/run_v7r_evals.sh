#!/bin/bash
# Post-v7r queue (node03 GPU3): re-establish every headline number on the RECAPTIONED model and prompts.
# Waits for v7r (V7R_DONE, which also runs the 16 px CFG sweep 1.5/2/3/4) and the 10k weak model (v7r_snap10k).
# Per resolution R in 16, 20, 24, 12 (the user's range is 12-24):
#   CFG sweep 1.5/2/2.5/3/4 (seed 0) -> best weight gets seeds 1, 2
#   label reference  bucket:<lower> w2           seeds 0-2   (not at 12: no lower bucket)
#   composed         10k snapshot + bucket:<lower> w1.5  seeds 0-2   (not at 12)
#   autoguidance     10k snapshot, w1.5          seeds 0-2
# Lower bucket as in the paper: 16->12, 20->16, 24->16. Prompts: heldout3000_prompts_recap.txt (eval_matched_r.sh
# picks it for v7r* names). Then SD-piXL on the recaptioned prompts (external baseline, ~9 h per sprite, 2 parallel).
# Launch: NEED_MB=20000 setsid nohup bash supervise.sh v7r_evals 3 bash baseline/run_v7r_evals.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-3}
until grep -q V7R_DONE logs/v7r.log 2>/dev/null; do sleep 300; done
until grep -q V7R_SNAP10K_DONE logs/v7r_snap10k.log 2>/dev/null; do sleep 300; done
CK=workdir/v7r/model_latest.pt
SNAP=workdir/v7r_snap10k/model_latest.pt
echo "[$(date +%m%d-%H:%M)] v7r and its 10k weak model ready; starting evals on GPU $GPU"

run() {  # <tag> <R> <sample_e args...>
  local tag=$1 R=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return 0
  echo "[$(date +%m%d-%H:%M)] RUN $tag @${R}px :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched_r.sh $tag $R $GPU && touch runs_out/${tag}_matched_eval/.done \
    || echo "EVAL_FAIL $tag"
}
best_w() {  # <R> <prefix> -> best CFG weight on seed 0, e.g. 1.5
  $P - "$1" "$2" <<'EOF'
import json, re, sys
R, pre = sys.argv[1], sys.argv[2]
d = json.load(open(f"runs_out/fair_fd{R}.json"))
c = {}
for k, v in d.items():
    m = re.fullmatch(rf"runs_out/{re.escape(pre)}_cfg([0-9p]+)_matched_eval/s{R}", k)
    if m:
        c[m.group(1).replace("p", ".")] = v
print(min(c, key=c.get) if c else "2")
EOF
}
lower() { case $1 in 16) echo 12 ;; 20) echo 16 ;; 24) echo 16 ;; *) echo "" ;; esac; }

for R in 16 20 24 12; do
  pre=v7r_r$R; [ $R = 16 ] && pre=v7r          # run_v7r.sh already wrote v7r_cfg{1p5,2,3,4} at 16 px
  for w in 1.5 2 2.5 3 4; do run ${pre}_cfg$(echo $w | tr . p) $R --cfg $w; done
  bw=$(best_w $R $pre); echo "best CFG @${R}px = $bw"
  for s in 1 2; do run ${pre}_cfg$(echo $bw | tr . p)_seed$s $R --cfg $bw --seed $s; done
  L=$(lower $R)
  for s in 0 1 2; do
    sfx=""; [ $s != 0 ] && sfx=_seed$s
    if [ -n "$L" ]; then
      run ${pre}_bk${L}_w2$sfx $R --cfg 2 --guide_mode bucket:$L --seed $s
      run ${pre}_stk${L}_w1p5$sfx $R --cfg 1.5 --guide_ckpt $SNAP --guide_mode bucket:$L --seed $s
    fi
    run ${pre}_autog10k_w1p5$sfx $R --cfg 1.5 --guide_ckpt $SNAP --seed $s
  done
done
echo "[$(date +%m%d-%H:%M)] V7R_EVALS_DONE"

# external baseline on the same recaptioned prompts: SD-piXL, 16 px, sprites 0-29, two at a time on this GPU
bash baseline/run_sdpixl_recap.sh 0 29 $GPU 2
echo V7R_QUEUE_ALL_DONE
