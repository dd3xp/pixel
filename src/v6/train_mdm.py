"""A1: masked discrete diffusion over per-channel 8-bit pixel tokens (arch_scout_2026-09-09, candidate A1).

Why this is not the v6f discrete probe that was killed: v6f defined the diffusion over the 32 indices of a
CORPUS-WIDE palette, while the reference's own median is 34 colours PER SPRITE, so its vocabulary could not
express the data at all.  Here the vocabulary is the actual 8-bit channel value, which is lossless (verified:
every sprite in the corpus is 8-bit RGBA), so the quantisation floor is exactly the real floor, 3.45.

Model:
  a 16x16 RGBA sprite -> 256 pixels x 4 channels = 1024 tokens, each a 256-way categorical (+1 MASK id).
  A bidirectional transformer sees all tokens at once; the four channels of one pixel are ADJACENT in the
  sequence and share a pixel-position embedding, so R,G,B,A of the same pixel can condition on each other.
  That is the PixelCNN++ lesson and the mitigation for the one failure mode the scout flagged.
  Text enters by cross-attention on frozen CLIP; the resolution bucket is a learned embedding.

Training is the simplified masked-diffusion ELBO (Shi et al., NeurIPS 2024, arXiv:2406.04329): draw t~U(0,1),
mask each token independently with probability t, predict the masked ones, weight the cross-entropy by 1/t.
Neighbouring channel values get partial credit through a Gaussian kernel over the 256-way target -- the graded
version of "which of two adjacent colours", which is exactly the sparse-signal problem that killed v6f.
"""
import argparse
import copy
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, "src/v6")
import train_v7
from train_v7 import BUCKETS, BucketSampler, NativeSprites, embed

V = 256      # per-channel values
MASK = V     # extra id meaning "masked"
CH = 4       # RGBA


class MDM(nn.Module):
    def __init__(self, dim=512, depth=12, heads=8, ctx=512, max_pix=32 * 32):
        super().__init__()
        self.tok = nn.Embedding(V + 1, dim)
        self.ch = nn.Embedding(CH, dim)
        self.pos = nn.Embedding(max_pix, dim)
        self.bucket = nn.Embedding(len(BUCKETS), dim)
        self.t_embed = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.blocks = nn.TransformerDecoder(layer, depth)
        self.ctx_proj = nn.Linear(ctx, dim)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, V))

    def forward(self, ids, pix_idx, ch_idx, t, cond, lab):
        h = self.tok(ids) + self.pos(pix_idx)[None] + self.ch(ch_idx)[None]
        h = h + self.t_embed(t[:, None, None].float()) + self.bucket(lab)[:, None]
        h = self.blocks(h, self.ctx_proj(cond))
        return self.out(h)


def smooth_target(x, sigma=2.0):
    """Gaussian label smoothing over the 256-way axis, so adjacent channel values get partial credit."""
    v = torch.arange(V, device=x.device).float()
    d = (v[None, :] - x[:, None].float()) ** 2
    w = torch.exp(-d / (2 * sigma ** 2))
    return w / w.sum(-1, keepdim=True)


def to_ids(x):
    """[-1,1] CHW float -> (H*W*C,) long ids, the four channels of a pixel adjacent."""
    a = ((x + 1) * 127.5).round().clamp(0, 255).long()
    return a.permute(1, 2, 0).reshape(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--out", default="workdir/probe_mdm")
    p.add_argument("--exclude", default="runs_out/holdout_exclude.txt")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--bs_scale", type=float, default=0.35)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--snap_every", type=int, default=20000)
    p.add_argument("--max_side", type=int, default=32,
                   help="largest bucket to train on; 48/64 give 9k-16k tokens, beyond full attention here")
    args = p.parse_args()
    dev = "cuda"
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    for k in train_v7.BATCH:
        train_v7.BATCH[k] = max(4, int(train_v7.BATCH[k] * args.bs_scale))
    sources = [("data/oga_clean", "data/oga_captions.csv", 1),
               ("data/extra_all", "data/extra_all.csv", 1),
               ("data/oga_clean", "data/tool_candidates.csv", 2)]
    exclude = [l.strip() for l in open(args.exclude) if l.strip()]
    ds = NativeSprites(sources, exclude)
    # Full attention over 4 tokens per pixel makes the 48 and 64 buckets 9k-16k tokens long, which this model
    # cannot carry; the paper's range is 12-32 px, so train on those buckets only and drop the rest.
    keep = [i for i, b in enumerate(ds.bucket_of) if BUCKETS[b] <= args.max_side]
    ds.rows = [ds.rows[i] for i in keep]
    ds.bucket_of = [ds.bucket_of[i] for i in keep]
    print(f"dataset: {len(ds)} (buckets <= {args.max_side} px)", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=BucketSampler(ds.bucket_of, args.steps), num_workers=8)

    tokz = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    txt = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval().requires_grad_(False)

    model = MDM(args.dim, args.depth, max_pix=args.max_side ** 2).to(dev)
    print(f"params: {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ema = copy.deepcopy(model).eval().requires_grad_(False)

    step = 0
    for x, texts, b in loader:
        texts = ["" if random.random() < 0.1 else t for t in texts]
        cond = embed(list(texts), tokz, txt, dev)
        x, b = x.to(dev), b.to(dev)
        B = x.shape[0]
        ids = torch.stack([to_ids(xi) for xi in x]).to(dev)
        L = ids.shape[1]
        pix_idx = torch.arange(L, device=dev) // CH
        ch_idx = torch.arange(L, device=dev) % CH

        t = torch.rand(B, device=dev).clamp(1e-3, 1.0)
        m = torch.rand(B, L, device=dev) < t[:, None]
        inp = torch.where(m, torch.full_like(ids, MASK), ids)
        logits = model(inp, pix_idx, ch_idx, t, cond, b)
        if m.any():
            tgt = smooth_target(ids[m], args.sigma)
            lp = F.log_softmax(logits[m], -1)
            per = -(tgt * lp).sum(-1)
            w = (1.0 / t)[:, None].expand_as(m)[m]
            loss = (per * w).mean()
        else:
            loss = logits.sum() * 0.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1.0 - args.ema)
        step += 1
        if step % 200 == 0:
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} bucket={BUCKETS[int(b[0])]}", flush=True)
        if step % 5000 == 0 or step == args.steps:
            torch.save(ema.state_dict(), out / "model_latest.pt")
        if args.snap_every and step % args.snap_every == 0:
            torch.save(ema.state_dict(), out / f"model_step{step:06d}.pt")
    torch.save(ema.state_dict(), out / "model_latest.pt")
    print("MDM_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
