"""Scores for the SDXL generality test: FD-DINOv2 (ViT-L/14 CLS, Stein et al. 2023 protocol: 224 px bicubic,
ImageNet normalisation) against real COCO val2014 images, and CLIP score (ViT-L/14, 100*cos) against the captions.

Every config is scored against the same real reference set (default: all 3000 images of shard 0), so FD values are
comparable across configs even though n = 1000 biases them upward. Results are appended to <out json> by tag.
Usage: python src/sdxl_gen/metrics.py --gen runs_out/sdxl_pilot/cfg5 --tag cfg5 --out runs_out/sdxl_pilot/scores.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from transformers import AutoModel, CLIPModel, CLIPProcessor

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(files):
    return [Image.open(f).convert("RGB") for f in files]


@torch.no_grad()
def dino_feats(model, ims, dev, bs=64):
    out = []
    for i in range(0, len(ims), bs):
        x = torch.stack([torch.from_numpy(np.asarray(im.resize((224, 224), Image.BICUBIC))).permute(2, 0, 1)
                         for im in ims[i:i + bs]]).float() / 255
        x = ((x - MEAN) / STD).to(dev)
        out.append(model(pixel_values=x).pooler_output.float().cpu())
    return torch.cat(out).numpy().astype(np.float64)


def fd(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs, _ = linalg.sqrtm(s1 @ s2, disp=False)
    cs = cs.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs))


@torch.no_grad()
def clip_score(model, proc, ims, caps, dev, bs=64):
    s = []
    for i in range(0, len(ims), bs):
        inp = proc(text=caps[i:i + bs], images=ims[i:i + bs], return_tensors="pt", padding=True, truncation=True).to(dev)
        o = model(**inp)
        ie = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
        te = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
        s.append((100 * (ie * te).sum(-1)).cpu())
    return float(torch.cat(s).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real", default="data/coco30k/real512")
    ap.add_argument("--captions", default="data/coco30k/captions_shard0.txt")
    ap.add_argument("--ref_range", type=int, nargs=2, default=[1000, 3000],
                    help="real images used as the FD reference; disjoint from the 1000 prompt rows 0-999 so that the "
                         "real rows 0-999, scored as if generated (--gen data/coco30k/real512 --real_as_gen), give "
                         "the floor")
    ap.add_argument("--real_as_gen", action="store_true", help="score real rows 0-999 (the floor)")
    a = ap.parse_args()
    dev = "cuda"
    caps = [l.rstrip("\n") for l in open(a.captions, encoding="utf-8")]
    gf = sorted(Path(a.gen).glob("*.png"))
    if a.real_as_gen:
        gf = gf[:1000]
    idx = [int(f.stem) for f in gf]
    gims = load(gf)
    dino = AutoModel.from_pretrained("facebook/dinov2-large").to(dev).eval()
    lo, hi = a.ref_range
    cache = Path(a.real) / f"_dinov2L_{lo}_{hi}.npy"
    if cache.exists():
        fr = np.load(cache)
    else:
        fr = dino_feats(dino, load(sorted(Path(a.real).glob("*.png"))[lo:hi]), dev)
        np.save(cache, fr)
    fg = dino_feats(dino, gims, dev)
    del dino
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    res = dict(n=len(gims), fd_dinov2=fd(fg, fr), clip=clip_score(clip, proc, gims, [caps[i] for i in idx], dev))
    p = Path(a.out)
    allr = json.loads(p.read_text()) if p.exists() else {}
    allr[a.tag] = res
    p.write_text(json.dumps(allr, indent=2))
    print(f"{a.tag}: n={res['n']} FD-DINOv2={res['fd_dinov2']:.2f} CLIP={res['clip']:.2f}", flush=True)


if __name__ == "__main__":
    main()
