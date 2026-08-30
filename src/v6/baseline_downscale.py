"""The baseline every reviewer thinks of first: generate big with SDXL, then shrink.

Our claim is that low-resolution sprites need native low-resolution generation.
The obvious objection is "just render 1024px and downscale", and until now the
paper had no answer -- the only baseline was optimisation-based SD-piXL.

Deliberately STEELMANNED, revised twice after passes that crippled it:

  v1  asked for a "plain white background"; SDXL ignored it and produced wooden
      tables and grey scenery that swamped the object once downscaled.
  v2  added flood-fill matting from the borders.  That fixed the two prompts
      where SDXL happened to isolate the object (pickaxe, axe -- both then gave
      clean, recognisable 16px sprites) but six of eight still came back with
      foreground>=0.90, i.e. no background to remove: the GENERATION, not the
      matting, was the problem.
  v3  (this) generates N candidates per prompt and keeps the best one, judged
      first on whether it is actually an isolated object (foreground fraction in
      a plausible band) and then on CLIP agreement with the prompt.  That is the
      same best-of-N treatment our own model gets in the Pareto study, so
      neither side is being judged on a single unlucky sample.

  alpha     : matting at 1024 and a PREMULTIPLIED downscale, the same convention
              our own low-bucket augmentation uses -- otherwise background
              colour bleeds into the sprite's edge pixels.
  downscale : mean / mode / LANCZOS / NEAREST.  Block-mode is what pixel artists
              reach for: averaging invents in-between colours that a limited
              palette would never contain.
  palette   : none, or DawnBringer32 -- the palette SD-piXL was given.  Earlier
              passes showed quantisation clearly HURTS, turning noisy blocks
              into speckle; it is kept so that "the palette is not what holds
              the baseline back" is on the record rather than assumed.

We report the BEST variant, not a convenient strawman.

Usage: CUDA_VISIBLE_DEVICES=2 python src/v6/baseline_downscale.py \
           --prompts baseline/prompts8.txt --out runs_out/dsbaseline3 --n 8
"""
import argparse
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PALETTE = "/mnt/data/kw/RoundSquisheen/pixel/SD-piXL/assets/palettes/lospec/dawnbringer32.hex"
SIZES = [12, 16, 20, 24]
STYLE = ("isolated on a solid white background, die-cut sticker, centered, "
         "no shadow, no scenery, simple flat colours")
NEG = ("photograph, realistic, 3d render, text, watermark, busy background, "
       "scene, table, wood grain, gradient background, multiple objects, shadow")
# a real sprite covers a modest part of the canvas; 0.95 means matting found no
# background at all and 0.02 means the object vanished
FG_LO, FG_HI = 0.04, 0.70


def load_palette(path):
    cols = []
    for line in open(path, encoding="utf-8"):
        h = line.strip().lstrip("#")
        if len(h) >= 6:
            cols.append(tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)))
    return np.array(cols, dtype=np.int16)


def matte(img, tol=38):
    """Flood-fill the background inward from the borders.

    A white-threshold cut fails whenever SDXL renders a tinted or textured
    backdrop, and it also punches holes in light-coloured parts of the object.
    Growing only from the border removes background actually connected to the
    edge, so a white highlight inside the sprite survives.
    """
    a = np.array(img.convert("RGB")).astype(np.int16)
    h, w, _ = a.shape
    bg = np.zeros((h, w), bool)
    seeds = deque()
    ref = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).reshape(-1, 3), 0)

    def seed(y, x):
        if not bg[y, x] and np.abs(a[y, x] - ref).sum() <= tol:
            bg[y, x] = True
            seeds.append((y, x))

    for x in range(w):
        seed(0, x)
        seed(h - 1, x)
    for y in range(h):
        seed(y, 0)
        seed(y, w - 1)
    while seeds:
        y, x = seeds.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                seed(ny, nx)
    return np.dstack([a.astype(np.uint8), (~bg).astype(np.uint8) * 255])


def blocks(rgba, n):
    s = rgba.shape[0] // n * n
    k = s // n
    return rgba[:s, :s].reshape(n, k, n, k, 4).transpose(0, 2, 1, 3, 4).reshape(n, n, -1, 4)


def ds_mean(rgba, n):
    """Premultiplied average: a plain average drags background colour into the
    edge pixels of a block that is mostly transparent."""
    b = blocks(rgba, n).astype(np.float32)
    a = b[..., 3:] / 255.0
    rgb = (b[..., :3] * a).sum(2) / np.maximum(a.sum(2), 1e-6)
    return np.dstack([rgb.round().clip(0, 255).astype(np.uint8),
                      (a.mean(2) * 255).round().astype(np.uint8)[..., 0]])


def ds_mode(rgba, n):
    b = blocks(rgba, n)
    out = np.zeros((n, n, 4), np.uint8)
    for y in range(n):
        for x in range(n):
            px = b[y, x]
            vis = px[px[:, 3] > 128]
            out[y, x, 3] = 255 if len(vis) * 2 >= len(px) else 0
            src = vis if len(vis) else px
            out[y, x, :3] = Counter(tuple(p) for p in (src[:, :3] // 8 * 8)).most_common(1)[0][0]
    return out


def ds_filter(rgba, n, f):
    return np.array(Image.fromarray(rgba, "RGBA").resize((n, n), f))


def quantise(rgba, pal):
    d = ((rgba[:, :, :3].astype(np.int16)[:, :, None, :] - pal[None, None]) ** 2).sum(-1)
    return np.dstack([pal[d.argmin(-1)].astype(np.uint8), rgba[:, :, 3]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8, help="candidates per prompt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "big").mkdir(parents=True, exist_ok=True)

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8")
               if l.strip() and not l.startswith("#")]
    pal = load_palette(PALETTE)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    variants = {
        "mean": ds_mean,
        "mode": ds_mode,
        "lanczos": lambda r, n: ds_filter(r, n, Image.LANCZOS),
        "nearest": lambda r, n: ds_filter(r, n, Image.NEAREST),
    }

    for pi, p in enumerate(prompts):
        # Skip prompts already rendered.  This node clears GPU jobs every few
        # hours, and a 237-prompt distillation run that restarts from zero each
        # time never finishes; the matted 1024px file is written last for a
        # prompt, so its presence means that prompt is complete.
        if (out / "big" / f"{pi:02d}_matted.png").exists():
            continue
        cands = []
        for k in range(args.n):
            big = pipe(prompt=f"{p}, {STYLE}", negative_prompt=NEG,
                       num_inference_steps=30, guidance_scale=7.0,
                       height=1024, width=1024,
                       generator=torch.Generator("cuda").manual_seed(
                           args.seed + pi * 100 + k)).images[0]
            rgba = matte(big)
            fg = float((rgba[:, :, 3] > 128).mean())
            white = Image.alpha_composite(
                Image.new("RGBA", (1024, 1024), (255, 255, 255, 255)),
                Image.fromarray(rgba, "RGBA")).convert("RGB")
            with torch.no_grad():
                inp = proc(text=[p], images=[white], return_tensors="pt",
                           padding=True).to("cuda")
                f_i = F.normalize(clip.get_image_features(pixel_values=inp["pixel_values"]), dim=-1)
                f_t = F.normalize(clip.get_text_features(
                    input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]), dim=-1)
                score = float((f_i * f_t).sum())
            cands.append((FG_LO <= fg <= FG_HI, score, fg, big, rgba))
        # isolated candidates first, then CLIP agreement
        cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
        ok, score, fg, big, rgba = cands[0]
        big.save(out / "big" / f"{pi:02d}.png")
        Image.fromarray(rgba, "RGBA").save(out / "big" / f"{pi:02d}_matted.png")
        for n in SIZES:
            for vname, fn in variants.items():
                small = fn(rgba, n)
                for q, tag in ((False, "raw"), (True, "db32")):
                    arr = quantise(small, pal) if q else small
                    d = out / f"s{n}" / f"{vname}_{tag}"
                    d.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(arr, "RGBA").save(d / f"{pi:02d}.png")
        print(f"[{pi + 1}/{len(prompts)}] {p}  isolated={ok} fg={fg:.2f} clip={score:.3f} "
              f"(best of {args.n})", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
