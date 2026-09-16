"""External baseline nobody has run yet: generate with a large pretrained text-to-image model and downscale.

This is the pipeline any practitioner would try first, and the comparison a reviewer will demand, so it has to
exist before we can say anything about being ahead of the field.  SDXL-base-1.0 generates at 1024, the sprite is
cut out of the plain background, and the result goes through EXACTLY the project's own to_tensor(R) path (the same
BOX downscale, the same hard alpha, the same centred canvas) so the FD is computed on identical footing.

Usage: python sdxl_downscale.py --n 1000 --size 16 --out runs_out/sdxl_dn_eval --steps 30
"""
import argparse, os, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "src/v6")
from train_v7 import to_tensor  # the project's own preprocessing, so the comparison is on identical footing


def to_rgba(x):  # [-1,1] CHW float -> PIL RGBA, same convention as fd_fair
    a = ((x.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(a, "RGBA")


def cutout(img, tol=18):
    """Plain-background cutout: flood the border-connected near-background colour to alpha 0, then crop to content.
    The background colour is taken as the median of the 1-px border, which is what a plain-background prompt gives."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    h, w, _ = a.shape
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    close = (np.abs(a - bg).max(axis=2) <= tol)
    # keep only background pixels connected to the border, so same-coloured pixels inside the sprite survive
    from collections import deque
    seen = np.zeros((h, w), bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if close[y, x] and not seen[y, x]:
                seen[y, x] = True; q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if close[y, x] and not seen[y, x]:
                seen[y, x] = True; q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and close[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True; q.append((ny, nx))
    alpha = (~seen).astype(np.uint8) * 255
    out = np.dstack([np.asarray(img.convert("RGB")), alpha])
    ys, xs = np.where(alpha > 0)
    if len(ys) == 0:
        return Image.fromarray(out, "RGBA")
    return Image.fromarray(out[ys.min():ys.max() + 1, xs.min():xs.max() + 1], "RGBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="runs_out/heldout3000_prompts.txt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--gen_res", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--suffix", default=", pixel art sprite of a single object, plain flat white background, centred")
    ap.add_argument("--out", default="runs_out/sdxl_dn_eval")
    ap.add_argument("--keep_big", action="store_true", help="also save the 1024 px renders (large)")
    args = ap.parse_args()

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip()][:args.n]
    outdir = Path(args.out) / f"s{args.size}"
    outdir.mkdir(parents=True, exist_ok=True)
    done = len(list(outdir.glob("*.png")))
    if done >= len(prompts):
        print(f"already have {done} samples in {outdir}"); return

    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    g = torch.Generator("cuda")
    for i in range(done, len(prompts), args.batch):
        chunk = prompts[i:i + args.batch]
        g.manual_seed(i)
        imgs = pipe([c + args.suffix for c in chunk], num_inference_steps=args.steps,
                    guidance_scale=args.cfg, height=args.gen_res, width=args.gen_res, generator=g).images
        for k, im in enumerate(imgs):
            idx = i + k
            sprite = cutout(im)
            x = to_tensor(sprite, args.size)          # the project's own BOX downscale + hard alpha
            to_rgba(x).save(outdir / f"{idx:02d}_0.png" if idx < 100 else outdir / f"{idx:05d}.png")
            if args.keep_big:
                im.save(Path(args.out) / f"big_{idx:05d}.jpg", quality=88)
        print(f"[{min(i + args.batch, len(prompts))}/{len(prompts)}]", flush=True)
    print("SDXL_DOWNSCALE_DONE", outdir)


if __name__ == "__main__":
    main()
