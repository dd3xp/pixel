"""Sample a discrete palette-index diffusion model (ordinal OR absorbing) at a
fixed size, over a prompt list, for the 3-way probe comparison."""
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, "src")
from v6.train_ordinal import (load_hex_palette, oklab_order, DiscretePaletteUNet,
                              OrdinalSchedule, ordinal_sample, embed, BUCKETS)
from v6.train_f import AbsorbingSchedule, discrete_sample
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--schedule", choices=["ordinal", "absorbing"], required=True)
    p.add_argument("--palette", default="assets/palettes/dawnbringer32.hex")
    p.add_argument("--prompts", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    dev = "cuda"
    palette = load_hex_palette(args.palette).to(dev)
    if args.schedule == "ordinal":
        palette = palette[oklab_order(palette).to(dev)]
        sched = OrdinalSchedule(T=args.T, K=palette.shape[0], device=dev)
        sampler = ordinal_sample
    else:
        sched = AbsorbingSchedule(T=args.T, K=palette.shape[0])
        sampler = discrete_sample
    K = palette.shape[0]
    net = DiscretePaletteUNet(K, embed_dim=16).to(dev).eval()
    net.load_state_dict(torch.load(args.ckpt, map_location=dev))
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip() and not l.startswith("#")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    uncond = embed([""], tok, enc, dev)
    for pi, pr in enumerate(prompts):
        cond = embed([pr], tok, enc, dev)
        for j in range(args.n):
            rgba = sampler(net, sched, cond, uncond, args.size, device=dev,
                           steps=args.steps, cfg=args.cfg, seed=j, palette=palette)  # (1,4,H,W)
            arr = (rgba[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(arr, "RGBA").save(out / f"{pi:03d}_{j}.png")
        if pi % 50 == 0:
            print(f"[{pi+1}/{len(prompts)}] {pr}", flush=True)
    print(f"DONE -> {out} ({len(prompts)*args.n} imgs)")

if __name__ == "__main__":
    main()
