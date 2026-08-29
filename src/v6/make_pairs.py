"""Build REAL paired conditioning data: (SDXL-style render of a sprite) -> (that sprite).

Why this exists
---------------
v8/v8b/v9 all fed the image branch `cond_view(target_sprite)` -- a blurred,
upscaled copy of the very sprite being generated.  So the model learned
"reference = a fuzzy big version of my own output".  At test time we hand it a
512px SDXL render of a phone, which sits nowhere near that distribution, and the
branch contributes little (GAP 0.017 against a 0.0017 floor).  v8b tried to close
the gap with heavier synthetic degradation and made transfer WORSE -- synthetic
blur is not a different renderer.  So build the real pairing instead.

Settings below are not defaults; each was chosen from a sweep whose output is in
runs/v10_pairs/:
  * img2img, not text2img -- captions like "a heart shaped like a heart" would
    render an object unrelated to the sprite.  Starting FROM the sprite keeps
    content, pose and palette aligned and changes only the style.
  * NEUTRAL prompt naming no style, and low guidance.  At guidance 6.0 with a
    "smooth digital illustration, product shot" prompt every sprite collapsed
    into the same glossy abstract blob -- prompt over-steering, not an img2img
    limit.  Dropping to ~2.0 with a contentless prompt preserved object identity
    (sweep: runs/v10_pairs/pair_sweep4_guidance.png).
  * strength 0.75 -- 0.5-0.6 leaves the pixel grid intact (no style gap crossed),
    0.85+ invents a different object (a stone axe became a magnifying glass).
  * NEAREST upscale.  The first version used LANCZOS reasoning that soft edges
    restyle better; on detailed sprites it actually lost more identity.
  * DETAIL FILTER (the important one).  Small flat sprites -- a 16px sword, a
    stone axe -- stay blocky at every usable strength and only break apart at
    high ones: there are too few pixels to reconstruct the object from.  Only
    sprites with enough size AND enough distinct colours yield a faithful
    restyle.  Resolution is a separate conditioning input in our model, so the
    reference->sprite mapping learned on detailed sprites is expected to carry
    over to the 12-24px buckets.

Usage:
  CUDA_VISIBLE_DEVICES=1 python src/v6/make_pairs.py --n 3000 --out data/pairs
"""
import argparse
import csv
import random
from pathlib import Path

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

SOURCES = [                       # (image dir, captions csv)
    ("data/oga_clean", "data/oga_captions_col.csv"),
    ("data/extra_all", "data/extra_all_col.csv"),
    ("data/bowtool_items", "data/bowtool_captions.csv"),
]
NEUTRAL = "a single object on a plain white background"
NEG = "text, watermark, multiple objects, busy background"
MIN_SIDE = 32
MIN_COLOURS = 24


def load_rows():
    rows, seen = [], set()
    for img_dir, csv_path in SOURCES:
        if not Path(csv_path).exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p = Path(img_dir) / r["path"]
                if p.as_posix() in seen or not p.exists():
                    continue
                seen.add(p.as_posix())
                rows.append((p, r["text"]))
    return rows


def detailed(path):
    """Enough pixels and enough distinct colours to survive a restyle."""
    try:
        im = Image.open(path).convert("RGBA")
    except Exception:
        return False
    if max(im.size) < MIN_SIDE:
        return False
    opaque = [px[:3] for px in im.getdata() if px[3] > 128]
    return len(set(opaque)) >= MIN_COLOURS


def sprite_to_canvas(p, side=512):
    im = Image.open(p).convert("RGBA")
    s = max(im.size)
    sq = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    sq.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
    bg = Image.new("RGBA", sq.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, sq).convert("RGB").resize((side, side), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="data/pairs")
    ap.add_argument("--strength", type=float, default=0.75)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    random.Random(args.seed).shuffle(rows)

    kept = []
    for p, t in rows:                      # filter lazily; stop once we have enough
        if detailed(p):
            kept.append((p, t))
            if len(kept) >= args.n:
                break
    print(f"{len(kept)} detailed sprites selected "
          f"(>={MIN_SIDE}px, >={MIN_COLOURS} colours) out of {len(rows)} scanned", flush=True)

    def build(**kw):
        return StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16, use_safetensors=True, **kw).to("cuda")

    try:
        pipe = build(variant="fp16")
    except Exception as e:
        print(f"fp16 variant unavailable ({type(e).__name__}), using fp32 weights", flush=True)
        pipe = build()
    pipe.set_progress_bar_config(disable=True)

    index = []
    for i, (sprite_path, text) in enumerate(kept):
        try:
            img = pipe(prompt=NEUTRAL, negative_prompt=NEG,
                       image=sprite_to_canvas(sprite_path),
                       strength=args.strength, num_inference_steps=args.steps,
                       guidance_scale=args.guidance,
                       generator=torch.Generator("cuda").manual_seed(i)).images[0]
        except Exception as e:                 # one bad sprite must not kill the run
            print(f"  skip {sprite_path.name}: {e}", flush=True)
            continue
        name = f"{i:05d}.jpg"
        img.save(out / name, quality=92)
        index.append((sprite_path.as_posix(), name, text))
        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{len(kept)}] {sprite_path.name}", flush=True)

    with open(f"{args.out}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sprite", "ref", "text"])
        w.writerows(index)
    print(f"{len(index)} pairs -> {args.out}.csv", flush=True)


if __name__ == "__main__":
    main()
