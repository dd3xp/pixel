"""Is the model broken, or the decoder? Feed it REAL sprites with a known mask and look at what it predicts."""
import sys, torch, numpy as np
sys.path.insert(0, "src/v6")
from train_mdm import MDM, to_ids, MASK, CH, V, smooth_target
from train_v7 import NativeSprites, BUCKETS, embed
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F

dev = "cuda"
m = MDM(512, 12, max_pix=32*32).to(dev).eval()
m.load_state_dict(torch.load("workdir/probe_mdm/model_latest.pt", map_location=dev))
tokz = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
txt = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()

ds = NativeSprites([("data/oga_clean", "data/oga_captions.csv", 1)], [])
idx = [i for i, b in enumerate(ds.bucket_of) if BUCKETS[b] == 16][:8]
xs, ts = [], []
for i in idx:
    x, t, b = ds[i]
    xs.append(x); ts.append(t)
x = torch.stack(xs).to(dev)
cond = embed(ts, tokz, txt, dev)
lab = torch.full((len(idx),), BUCKETS.index(16), device=dev, dtype=torch.long)
ids = torch.stack([to_ids(xi) for xi in x]).to(dev)
L = ids.shape[1]
pix = torch.arange(L, device=dev) // CH
ch = torch.arange(L, device=dev) % CH

print("ground-truth id stats: mean %.1f  std %.1f  min %d  max %d" % (
    ids.float().mean(), ids.float().std(), ids.min(), ids.max()))
for frac in (0.1, 0.5, 0.9):
    t = torch.full((len(idx),), frac, device=dev)
    mask = torch.rand(len(idx), L, device=dev) < frac
    inp = torch.where(mask, torch.full_like(ids, MASK), ids)
    with torch.no_grad():
        lg = m(inp, pix, ch, t, cond, lab)
    pred = lg.argmax(-1)
    acc = (pred[mask] == ids[mask]).float().mean().item()
    near = ((pred[mask] - ids[mask]).abs() <= 8).float().mean().item()
    print("mask=%.1f  argmax mean %.1f std %.1f | exact %.3f | within8 %.3f" % (
        frac, pred[mask].float().mean(), pred[mask].float().std(), acc, near))
