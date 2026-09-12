#!/bin/bash
# SDXL generality pilot on node03 GPU2 (node09 is no longer used, user 09-12). Chained after probe_gft (1b).
# 1) COCO-30k val2014 shard 0 from hf-mirror -> data/coco30k/{captions_shard0.txt, real512/}; parquet deleted after.
# 2) scoring models: facebook/dinov2-large, openai/clip-vit-large-patch14.
# 3) configs in priority order (the key comparison first), 1000 prompts each, each scored right after it finishes:
#    FD-DINOv2 (ViT-L/14) vs real rows 1000-2999, CLIP-L score; floor = real rows 0-999.
#    Verdict: cfglabel (ours on top of CFG) below cfg5 AND negos512 at similar CLIP -> generality holds, scale up.
# Launch: NEED_MB=30000 setsid nohup bash supervise.sh sdxl_n3 2 bash baseline/run_sdxl_n3.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_ENDPOINT=https://hf-mirror.com
GPU=${CUDA_VISIBLE_DEVICES:-2}
until grep -q "PROBE_GFT_DONE\|PROBE_GFT_TRAIN_FAIL" logs/probe_gft.log 2>/dev/null; do sleep 300; done
echo "[$(date +%m%d-%H:%M)] probe_gft finished; SDXL pilot on GPU $GPU"

if [ ! -f data/coco30k/captions_shard0.txt ]; then
  mkdir -p data/coco30k
  $P - <<'EOF' || { echo SDXL_N3_DATA_FAIL; exit 1; }
import io
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from PIL import Image
f = hf_hub_download("sayakpaul/coco-30-val-2014", "data/train-00000-of-00010-45de7542ea7caa89.parquet",
                    repo_type="dataset", local_dir="data/coco30k")
rows = pq.read_table(f).to_pylist()
out = Path("data/coco30k/real512"); out.mkdir(exist_ok=True)
caps = []
for i, r in enumerate(rows):
    caps.append(r["caption"].replace("\n", " ").strip())
    im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
    w, h = im.size; s = min(w, h)
    im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((512, 512), Image.BICUBIC).save(out / f"{i:05d}.png")
open("data/coco30k/captions_shard0.txt", "w", encoding="utf-8").write("\n".join(caps) + "\n")
Path(f).unlink()
print(len(caps), "captions")
EOF
fi
$P -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/dinov2-large', allow_patterns=['*.json', 'model.safetensors'])
snapshot_download('openai/clip-vit-large-patch14', allow_patterns=['*.json', '*.txt', 'model.safetensors'])
print('SCORING_MODELS_OK')" || { echo SDXL_N3_MODELS_FAIL; exit 1; }
export HF_HUB_OFFLINE=1

SC=runs_out/sdxl_pilot/scores.json
mkdir -p runs_out/sdxl_pilot
grep -q '"real_floor"' $SC 2>/dev/null || CUDA_VISIBLE_DEVICES=$GPU $P src/sdxl_gen/metrics.py --gen data/coco30k/real512 \
    --real_as_gen --tag real_floor --out $SC
CFGS=("cfg5 --mode cfg --w 5" "cfglabel512_l1 --mode cfglabel --w 5 --lam 1 --low 512"
      "negos512 --mode negos --w 5 --low 512" "cfglabel512_l0p5 --mode cfglabel --w 5 --lam 0.5 --low 512"
      "cfg3 --mode cfg --w 3" "cfglabel256_l1 --mode cfglabel --w 5 --lam 1 --low 256"
      "label512_w3 --mode label --w 3 --low 512" "cfg7 --mode cfg --w 7"
      "negos256 --mode negos --w 5 --low 256" "label512_w2 --mode label --w 2 --low 512")
for c in "${CFGS[@]}"; do
  tag=${c%% *}; args=${c#* }
  grep -q "\"$tag\"" $SC 2>/dev/null && continue
  echo "[$(date +%m%d-%H:%M)] $tag :: $args"
  CUDA_VISIBLE_DEVICES=$GPU $P src/sdxl_gen/sdxl_sizeguide.py --prompts data/coco30k/captions_shard0.txt \
      --out runs_out/sdxl_pilot/$tag --end 1000 --bs 4 $args || { echo "SDXL_GEN_FAIL $tag"; continue; }
  CUDA_VISIBLE_DEVICES=$GPU $P src/sdxl_gen/metrics.py --gen runs_out/sdxl_pilot/$tag --tag $tag --out $SC \
      || echo "SDXL_SCORE_FAIL $tag"
done
echo SDXL_N3_DONE
