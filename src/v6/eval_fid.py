"""clean-FID / KID for the multi-res model at a given bucket size.
Reference set: held-out real sprites of that native bucket from data/oga_clean
(last N by sorted filename, never oversampled in training more than others —
note: not a strict held-out split; we report it as in-distribution FID).
Generated set: N samples from captions of the reference sprites (so text
conditions match the reference distribution), composited on white, upscaled
NEAREST to 64 for the Inception backbone (both sides identically).

Usage: CUDA_VISIBLE_DEVICES=0 python src/v6/eval_fid.py --ckpt workdir/v6e3_tools/model_latest.pt --size 16 --n 1000
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from sample_e import BUCKETS, build_model, embed, sample  # noqa: E402
from diffusers import DDPMScheduler  # noqa: E402
from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402


def composite_white(im: Image.Image, up: int) -> Image.Image:
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    out = Image.alpha_composite(bg, im.convert("RGBA")).convert("RGB")
    return out.resize((up, up), Image.NEAREST)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--up", type=int, default=64)
    p.add_argument("--bs", type=int, default=100)
    p.add_argument("--out", default="workdir/fid")
    args = p.parse_args()
    device = "cuda"
    tag = f"{Path(args.ckpt).parent.name}_s{args.size}"
    real_dir, fake_dir = Path(args.out) / f"real_s{args.size}", Path(args.out) / f"fake_{tag}"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open("data/oga_captions.csv", encoding="utf-8")))
    bidx = BUCKETS.index(args.size)
    picked = []
    for r in sorted(rows, key=lambda r: r["path"], reverse=True):  # tail of the sorted list = quasi held-out
        pth = Path("data/oga_clean") / r["path"]
        s = max(Image.open(pth).size)
        b = next((i for i, bb in enumerate(BUCKETS) if bb >= s), len(BUCKETS) - 1)
        if b == bidx:
            picked.append((pth, r["text"]))
        if len(picked) >= args.n:
            break
    print(f"reference sprites in bucket {args.size}: {len(picked)}", flush=True)
    for i, (pth, _) in enumerate(picked):
        im = Image.open(pth).convert("RGBA")
        side = max(im.size)
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        canvas = canvas.resize((args.size, args.size), Image.NEAREST)
        composite_white(canvas, args.up).save(real_dir / f"{i:05d}.png")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    model = build_model(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    texts = [t for _, t in picked]
    k = 0
    for i in range(0, len(texts), args.bs):
        chunk = texts[i:i + args.bs]
        cond = embed(chunk, tokenizer, enc, device)
        uncond = embed([""] * len(chunk), tokenizer, enc, device)
        imgs = sample(model, sched, cond, uncond, args.size, device, steps=100, cfg=4.0, seed=1000 + i)
        for im in imgs:
            a = (im[3] > 0.5).float()
            rgba = torch.cat([im[:3] * a, a[None]], 0)
            pil = Image.fromarray((rgba.permute(1, 2, 0) * 255).byte().numpy(), "RGBA")
            composite_white(pil, args.up).save(fake_dir / f"{k:05d}.png")
            k += 1
        print(f"generated {k}/{len(texts)}", flush=True)

    from cleanfid import fid
    score_fid = fid.compute_fid(str(real_dir), str(fake_dir), mode="clean", num_workers=4)
    score_kid = fid.compute_kid(str(real_dir), str(fake_dir), mode="clean", num_workers=4)
    line = f"{tag}: clean-FID={score_fid:.2f} KID={score_kid * 1000:.2f}e-3 (n={len(texts)}, up={args.up})"
    print(line, flush=True)
    with open(Path(args.out) / "results.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
