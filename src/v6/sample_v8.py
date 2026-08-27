"""Sample the image-conditioned model (train_v8.py).

Two modes:
  --refs <dir>   condition on reference images (any PNG/JPG), one row per file.
                 This is the "draw anything" path: the reference supplies the
                 shape, our model supplies the pixel grammar.
  (no --refs)    text-only, identical computation to v7 (77 tokens), so this
                 doubles as a regression check that the image branch did not
                 damage the original ability.

Text and image can be combined: pass --prompts with the same number of lines
as there are reference files, or a single line reused for all references.

Usage:
  python src/v6/sample_v8.py --ckpt workdir/v8_imgcond/model_latest.pt \
      --refs data/refs --prompts prompts/refs.txt --sizes 12 16 20 24 --n 4 --out runs_out/v8
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

BUCKETS = [12, 16, 20, 24, 32, 48, 64]
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def build_model(device):
    return UNet2DConditionModel(
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(128, 256, 512), cross_attention_dim=512,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        num_class_embeds=len(BUCKETS),
    ).to(device)


def load_ref(path):
    """Any image -> the 224px CLIP view, matching training's white-background
    convention (transparent references are composited, not left black)."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    side = max(im.size)
    sq = Image.new("RGB", (side, side), (255, 255, 255))
    sq.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    t = torch.from_numpy(np.array(sq.resize((224, 224), Image.BICUBIC))).permute(2, 0, 1).float() / 255.0
    return (t - CLIP_MEAN) / CLIP_STD


@torch.no_grad()
def embed_text(texts, tokenizer, encoder, device):
    tok = tokenizer(texts, padding="max_length", max_length=77, truncation=True,
                    return_tensors="pt").to(device)
    return encoder(**tok).last_hidden_state


@torch.no_grad()
def sample(model, scheduler, cond, uncond, size, device, steps=100, cfg=4.0, seed=0):
    scheduler.set_timesteps(steps)
    n = cond.shape[0]
    lab = torch.full((n,), BUCKETS.index(size), device=device, dtype=torch.long)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, 4, size, size, device=device, generator=g)
    for t in scheduler.timesteps:
        e_c = model(x, t, encoder_hidden_states=cond, class_labels=lab).sample
        e_u = model(x, t, encoder_hidden_states=uncond, class_labels=lab).sample
        x = scheduler.step(e_u + cfg * (e_c - e_u), t, x).prev_sample
    return ((x + 1) / 2).clamp(0, 1).cpu()


def to_rgba(img):
    a = (img[3] > 0.5).float()
    arr = torch.cat([img[:3] * a, a[None]], 0)
    return Image.fromarray((arr.permute(1, 2, 0) * 255).byte().numpy(), "RGBA")


def grid(images, rows, cols, scale, refs=None):
    """One row per prompt/reference. If refs given, column 0 shows the reference."""
    h, w = images[0].size[1], images[0].size[0]
    extra = 1 if refs else 0
    cell = max(1, h // 4)
    out = Image.new("RGBA", ((cols + extra) * w, rows * h))
    px = out.load()
    for y in range(rows * h):
        for x in range((cols + extra) * w):
            v = 209 if ((y // cell + x // cell) % 2) else 240
            px[x, y] = (v, v, v, 255)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        out.alpha_composite(im, ((c + extra) * w, r * h))
    out = out.resize((out.width * scale, out.height * scale), Image.NEAREST)
    if refs:  # paste references AFTER upscaling, so they stay full resolution
        cw, ch = w * scale, h * scale
        for r, rp in enumerate(refs):
            out.alpha_composite(Image.open(rp).convert("RGBA").resize((cw, ch), Image.LANCZOS),
                                (0, r * ch))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--refs", default=None, help="directory of reference images")
    p.add_argument("--prompts", default=None, help="text file; 1 line per ref, or a single shared line")
    p.add_argument("--sizes", type=int, nargs="+", default=[16])
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    model = build_model(device)
    img_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 512)).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    if isinstance(ck, dict) and "unet" in ck:
        model.load_state_dict(ck["unet"])
        img_proj.load_state_dict(ck["img_proj"])
    else:
        model.load_state_dict(ck)          # a plain v7 checkpoint: text-only
        print("warning: checkpoint has no image branch; --refs will be ignored", flush=True)
        args.refs = None
    model.eval()
    img_proj.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    ref_paths = []
    if args.refs:
        ref_paths = sorted([q for q in Path(args.refs).iterdir()
                            if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")])
        if not ref_paths:
            raise SystemExit(f"no images in {args.refs}")

    lines = []
    if args.prompts:
        lines = [l.strip() for l in open(args.prompts, encoding="utf-8")
                 if l.strip() and not l.startswith("#")]
    rows = len(ref_paths) if ref_paths else len(lines)
    if rows == 0:
        raise SystemExit("give --refs or --prompts")
    if ref_paths:
        texts = lines if len(lines) == rows else ([lines[0]] * rows if lines else ["a pixel art item"] * rows)
    else:
        texts = lines

    cond = embed_text(texts, tokenizer, enc, device).repeat_interleave(args.n, 0)
    uncond = embed_text([""] * rows * args.n, tokenizer, enc, device)
    if ref_paths:
        vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        px = torch.stack([load_ref(q) for q in ref_paths]).to(device)
        with torch.no_grad():
            vtok = img_proj(vision(pixel_values=px).last_hidden_state)
        vtok = vtok.repeat_interleave(args.n, 0)
        cond = torch.cat([cond, vtok], dim=1)
        # unconditional branch keeps the image tokens: CFG then steers on text
        # only, which is what we want -- the reference is not the thing we are
        # trying to amplify, the caption is.
        uncond = torch.cat([uncond, vtok], dim=1)

    for size in args.sizes:
        imgs = sample(model, scheduler, cond, uncond, size, device, args.steps, args.cfg, args.seed)
        rgba = [to_rgba(im) for im in imgs]
        (out / f"s{size}").mkdir(exist_ok=True)
        for i, im in enumerate(rgba):
            pi, k = divmod(i, args.n)
            im.save(out / f"s{size}" / f"{pi:02d}_{k}.png")
        grid(rgba, rows, args.n, max(1, 128 // size),
             refs=ref_paths or None).save(out / f"grid_s{size}.png")
        print(f"size {size}: {len(rgba)} samples -> {out}/grid_s{size}.png", flush=True)


if __name__ == "__main__":
    main()
