"""FD-DINOv2: Frechet distance in DINOv2-small feature space.

Rationale: Inception-FID is unreliable on tiny non-natural sprites (Inception
trained on 299px photos). DINOv2 features are self-supervised, not color-
dominated, and the recognized best FID backbone. We compute, at each target
resolution R, FD-DINOv2 between a method's R x R outputs and REAL sprites
downsampled to R x R -- both then upscaled NEAREST to 224 through an identical
pipeline. Lower = closer to the real-sprite distribution at that resolution.
"""
import sys, os, glob, random
import numpy as np
import torch
from PIL import Image
from scipy import linalg
from transformers import AutoModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_model = None
def model():
    global _model
    if _model is None:
        _model = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
    return _model

def load_rgb_at_R(path, R):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")
    if im.size != (R, R):
        im = im.resize((R, R), Image.BOX)           # real sprites -> R x R
    im = im.resize((224, 224), Image.NEAREST)         # identical upscale for all
    return np.asarray(im, dtype=np.float32) / 255.0

@torch.no_grad()
def embed(paths, R, bs=128):
    feats = []
    for i in range(0, len(paths), bs):
        arr = np.stack([load_rgb_at_R(p, R) for p in paths[i:i+bs]])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2)
        x = (x - MEAN) / STD
        out = model()(pixel_values=x.to(DEV)).last_hidden_state[:, 0]  # CLS
        feats.append(out.cpu().numpy())
    return np.concatenate(feats)

def stats(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)

def frechet(mu1, s1, mu2, s2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))

def fd(gen_paths, real_paths, R):
    mg, sg = stats(embed(gen_paths, R))
    mr, sr = stats(embed(real_paths, R))
    return frechet(mg, sg, mr, sr)
