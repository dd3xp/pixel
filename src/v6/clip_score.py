"""Text-faithfulness of saved matched-protocol samples: CLIP ViT-B/32 image-text cosine (x100).

Guidance must not trade prompt alignment for FD, so every FD row in the paper gets a CLIP score on the
SAME saved samples (runs_out/<tag>_matched_eval/s<R>, file <prompt_idx>_<k>.png) against the caption it was
sampled from (runs_out/heldout3000_prompts.txt, line = prompt_idx).  Real held-out sprites
(runs_out/heldout3000_totensor_s<R>/<k:05d>.png, same order) give the reference value.

R px RGBA is composited on mid-grey and nearest-upsampled to 224 before the CLIP preprocessor, identical for
every set, so only relative values matter.  Reports mean/sd of 100*cos and the fraction of samples whose own
caption is the best match among 100 random captions (retrieval@1/100, chance 1 %).

Usage:
  python src/v6/clip_score.py --size 16 --dirs real=runs_out/heldout3000_totensor_s16 \
      cfg4=runs_out/v7h_matched_eval/s16 bk12=runs_out/v7h_gbk12_w2_matched_eval/s16 [--out runs_out/clip16.json]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_set(d, n_prompts):
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    idx = []
    for f in files:
        m = re.match(r"(\d+)(?:_\d+)?\.png$", os.path.basename(f))
        idx.append(int(m.group(1)))
    keep = [(f, i) for f, i in zip(files, idx) if i < n_prompts]
    return [f for f, _ in keep], [i for _, i in keep]


def to_clip_img(f, side=224, bg=128):
    im = Image.open(f).convert("RGBA")
    a = im.split()[3]
    canvas = Image.new("RGB", im.size, (bg, bg, bg))
    canvas.paste(im.convert("RGB"), mask=a)
    return canvas.resize((side, side), Image.NEAREST)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--dirs", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--prompts", default="runs_out/heldout3000_prompts.txt")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--out", default=None, help="json to append results into")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = [l.rstrip("\n") for l in open(args.prompts, encoding="utf-8")]
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # text features once
    T = []
    for s in range(0, len(prompts), args.bs):
        t = proc(text=prompts[s:s + args.bs], return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
        T.append(torch.nn.functional.normalize(model.get_text_features(**t), dim=-1))
    T = torch.cat(T)

    rng = np.random.default_rng(0)
    res = {}
    for spec in args.dirs:
        name, d = spec.split("=", 1)
        files, idx = load_set(d, len(prompts))
        I = []
        for s in range(0, len(files), args.bs):
            ims = [to_clip_img(f) for f in files[s:s + args.bs]]
            x = proc(images=ims, return_tensors="pt").to(dev)
            I.append(torch.nn.functional.normalize(model.get_image_features(**x), dim=-1))
        I = torch.cat(I)
        Tm = T[torch.tensor(idx, device=dev)]
        cos = (I * Tm).sum(-1).cpu().numpy()
        # retrieval@1 among own caption + 99 random distractors
        hits = 0
        for k in range(len(idx)):
            others = rng.choice(len(prompts), 99, replace=False)
            cand = torch.cat([Tm[k:k + 1], T[torch.tensor(others, device=dev)]])
            hits += int((I[k:k + 1] @ cand.T).argmax().item() == 0)
        r1 = hits / max(1, len(idx))
        res[name] = {"n": len(idx), "clip_mean": float(100 * cos.mean()), "clip_sd": float(100 * cos.std()),
                     "r1_100": r1}
        print(f"CLIP@{args.size} {name:36s} n={len(idx)} 100cos {100 * cos.mean():6.2f} ± {100 * cos.std():5.2f}"
              f"  R@1/100 {100 * r1:5.1f}%", flush=True)
    if args.out:
        old = json.load(open(args.out)) if os.path.exists(args.out) else {}
        old.update({f"s{args.size}:{k}": v for k, v in res.items()})
        json.dump(old, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
