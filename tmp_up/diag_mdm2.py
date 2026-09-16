"""Was the 80% token accuracy real, or just zeros? Most pixels are transparent, so most tokens are 0;
predicting 0 everywhere would already score ~80%. Split the accuracy by token type."""
import sys, torch, numpy as np
sys.path.insert(0, "src/v6")
from train_mdm import MDM, to_ids, MASK, CH
from train_v7 import NativeSprites, BUCKETS, embed
from transformers import CLIPTextModel, CLIPTokenizer

dev = "cuda"
m = MDM(512, 12, max_pix=32*32).to(dev).eval()
m.load_state_dict(torch.load("workdir/probe_mdm/model_latest.pt", map_location=dev))
tokz = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
txt = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
ds = NativeSprites([("data/oga_clean", "data/oga_captions.csv", 1)], [])
idx = [i for i, b in enumerate(ds.bucket_of) if BUCKETS[b] == 16][:32]
xs, ts = [], []
for i in idx:
    x, t, b = ds[i]; xs.append(x); ts.append(t)
x = torch.stack(xs).to(dev)
cond = embed(ts, tokz, txt, dev)
lab = torch.full((len(idx),), BUCKETS.index(16), device=dev, dtype=torch.long)
ids = torch.stack([to_ids(xi) for xi in x]).to(dev)
L = ids.shape[1]
pix = torch.arange(L, device=dev) // CH
ch = torch.arange(L, device=dev) % CH

alpha = ids[:, ch == 3]                      # B, R*R
opaque_pix = (alpha >= 128)
print("token zero-rate: %.3f | opaque pixel rate: %.3f" % ((ids == 0).float().mean(), opaque_pix.float().mean()))
is_rgb = (ch != 3)[None].expand_as(ids)
pix_opaque = opaque_pix[:, pix]              # per token: does its pixel have alpha>=128
interesting = is_rgb & pix_opaque            # RGB of visible pixels -- the only tokens that carry the artwork

for frac in (0.5, 0.9, 1.0):
    t = torch.full((len(idx),), min(frac, 0.999), device=dev)
    mask = torch.rand(len(idx), L, device=dev) < frac if frac < 1.0 else torch.ones_like(ids, dtype=torch.bool)
    inp = torch.where(mask, torch.full_like(ids, MASK), ids)
    with torch.no_grad():
        pred = m(inp, pix, ch, t, cond, lab).argmax(-1)
    sel = mask
    all_acc = (pred[sel] == ids[sel]).float().mean().item()
    zero_share = (ids[sel] == 0).float().mean().item()
    i2 = sel & interesting
    int_acc = (pred[i2] == ids[i2]).float().mean().item() if i2.any() else float("nan")
    near = ((pred[i2] - ids[i2]).abs() <= 16).float().mean().item() if i2.any() else float("nan")
    predzero = (pred[i2] == 0).float().mean().item() if i2.any() else float("nan")
    print("mask=%.2f | all-token acc %.3f (zeros are %.0f%% of them) | VISIBLE-RGB acc %.3f  within16 %.3f  predicted-zero %.3f"
          % (frac, all_acc, 100*zero_share, int_acc, near, predzero))
