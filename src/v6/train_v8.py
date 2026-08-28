"""V8: image-conditioned pixel art -- "draw anything" via a borrowed shape prior.

Motivation
----------
v7 can only draw silhouettes it has seen (43.8k CC0 sprites, a fantasy-RPG
domain).  Asking for a smartphone fails because no smartphone sprite exists.
But a big text-to-image model HAS seen everything.  So: let the big model
supply the *shape*, and let our model supply the *pixel grammar*.

    "a coffee machine" -> SDXL/Flux 512px reference -> CLIP image encoder
                                                          |
                          text -> CLIP text encoder ----> [concat tokens]
                                                          |
                                                    our UNet -> 16x16 RGBA

Why this can work zero-shot: CLIP's image and text towers share a space, and
we already verified the text side transfers (the corpus contains "silver" 0
times, yet "grey iron ingot" works, because frozen CLIP maps grey near the
gray/white/silver we do have).  The image side is the same mechanism with a
stronger signal.

Training needs NO paired data
-----------------------------
We condition on the CLIP image embedding of the sprite *itself* (self-
supervised), but only after pushing the conditioning view through heavy
augmentation -- upscale to 224 with a random filter, blur, colour jitter,
white background.  That forces the model to reconstruct from a *coarse visual
impression* rather than to copy pixels, which is exactly the regime it will
face at inference when the reference is an SDXL render.

Keeping v7's text-only ability intact
-------------------------------------
Image tokens are concatenated to the text tokens (prefix conditioning), and a
per-BATCH coin flip decides whether this batch carries images at all.  On a
text-only batch the sequence is exactly v7's 77 tokens through exactly v7's
weights, so the old behaviour is preserved rather than perturbed.

Usage:
  CUDA_VISIBLE_DEVICES=1 python src/v6/train_v8.py --init_v7 workdir/v7c_bow/model_latest.pt
"""
import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from PIL import Image, ImageFilter
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

BUCKETS = [12, 16, 20, 24, 32, 48, 64]
BATCH = {12: 224, 16: 176, 20: 160, 24: 128, 32: 96, 48: 48, 64: 32}
LOW = [0, 1, 2, 3]
EVAL_SIZES = [12, 16, 20, 24]
EVAL_PROMPTS = [
    "a pixel art sword with a golden handle",
    "a pixel art red apple",
    "a pixel art flower with pink petals",
    "a pixel art kettle",
    "a pixel art potion bottle with blue liquid",
    "a pixel art golden coin",
    "a pixel art wooden shield",
    "a pixel art mushroom with a red cap",
]
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
RESAMPLE = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]


def downscale_rgba(im, side):
    """Premultiplied box downsample so transparent pixels don't bleed colour."""
    f = side / max(im.size)
    w, h = max(1, round(im.width * f)), max(1, round(im.height * f))
    a = np.array(im).astype(np.float32)
    a[:, :, :3] *= a[:, :, 3:4] / 255.0
    pm = Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGBA").resize((w, h), Image.BOX)
    b = np.array(pm).astype(np.float32)
    alpha = b[:, :, 3:4]
    b[:, :, :3] = np.where(alpha > 0, b[:, :, :3] / np.maximum(alpha, 1) * 255.0, 0)
    return Image.fromarray(b.clip(0, 255).astype(np.uint8), "RGBA")


def to_tensor(im, side):
    if max(im.size) > side:
        im = downscale_rgba(im, side)
    elif max(im.size) * 2 <= side:
        f = side // max(im.size)
        im = im.resize((im.width * f, im.height * f), Image.NEAREST)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    a = np.array(canvas).astype(np.float32)
    a[a[:, :, 3] < 128] = 0.0
    a[:, :, 3] = (a[:, :, 3] >= 128) * 255.0
    return torch.from_numpy(a).permute(2, 0, 1) / 127.5 - 1.0


def cond_view(im, strong=False):
    """Sprite -> a 224px 'reference render' the way an SDXL output would look:
    white background, upscaled with a random filter, blurred, colour-jittered.
    The degradation is what stops the model from learning a copy shortcut.

    strong=True widens every knob and adds crop/scale jitter plus a
    resample-roundtrip.  Rationale: with the mild schedule the model drifts
    towards the in-domain embedding distribution as training proceeds (v8:
    out-of-domain transfer peaked at 15k then fell back), so the conditioning
    view has to look less and less like our own sprites."""
    side = max(im.size)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    bg = Image.new("RGBA", sq.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, sq).convert("RGB")
    if strong and random.random() < 0.7:           # framing jitter: refs are not centred crops
        m = int(side * random.uniform(0.02, 0.18))
        box = (random.randint(0, m), random.randint(0, m),
               side - random.randint(0, m), side - random.randint(0, m))
        rgb = rgb.crop(box)
    rgb = rgb.resize((224, 224), random.choice(RESAMPLE))
    if strong and random.random() < 0.6:           # resample roundtrip = a different render pipeline
        k = random.choice([32, 48, 64, 96])
        rgb = rgb.resize((k, k), random.choice(RESAMPLE)).resize((224, 224), random.choice(RESAMPLE))
    blur_p, blur_hi = (0.9, 5.0) if strong else (0.8, 3.0)
    if random.random() < blur_p:
        rgb = rgb.filter(ImageFilter.GaussianBlur(random.uniform(0.5, blur_hi)))
    t = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0
    if strong:
        if random.random() < 0.9:                  # wider photometric jitter
            t = (t * random.uniform(0.7, 1.3) + random.uniform(-0.12, 0.12)).clamp(0, 1)
        if random.random() < 0.3:                  # desaturate: colour must not be the only cue
            g = t.mean(0, keepdim=True)
            t = (t * (1 - 0.6) + g * 0.6).clamp(0, 1)
        if random.random() < 0.3:                  # sensor-ish noise
            t = (t + torch.randn_like(t) * random.uniform(0.01, 0.05)).clamp(0, 1)
    elif random.random() < 0.7:
        t = (t * random.uniform(0.85, 1.15) + random.uniform(-0.06, 0.06)).clamp(0, 1)
    return (t - CLIP_MEAN) / CLIP_STD


class NativeSprites(torch.utils.data.Dataset):
    def __init__(self, sources, strong_aug=False):
        self.strong_aug = strong_aug
        self.rows, self.bucket_of = [], []
        for img_dir, captions_csv, repeat in sources:
            img_dir = Path(img_dir)
            with open(captions_csv, newline="", encoding="utf-8") as f:
                rows = [(img_dir / r["path"], r["text"]) for r in csv.DictReader(f)]
            n_aug = 0
            for path, text in rows:
                s = max(Image.open(path).size)
                b = next((i for i, bb in enumerate(BUCKETS) if bb >= s), len(BUCKETS) - 1)
                self.rows.extend([(path, text)] * repeat)
                self.bucket_of.extend([b] * repeat)
                for t in LOW:
                    if t != b and s >= BUCKETS[t] * 1.25:
                        self.rows.extend([(path, text)] * repeat)
                        self.bucket_of.extend([t] * repeat)
                        n_aug += repeat
            print(f"source {img_dir}: {len(rows)} x{repeat} (+{n_aug} low-bucket aug)", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, text = self.rows[i]
        b = self.bucket_of[i]
        im = Image.open(path).convert("RGBA")
        return to_tensor(im, BUCKETS[b]), text, b, cond_view(im, self.strong_aug)


class BucketSampler(torch.utils.data.Sampler):
    def __init__(self, bucket_of, steps):
        self.groups = {b: [i for i, x in enumerate(bucket_of) if x == b] for b in range(len(BUCKETS))}
        self.groups = {b: g for b, g in self.groups.items() if len(g) >= 8}
        self.keys = sorted(self.groups)
        self.weights = [len(self.groups[b]) for b in self.keys]
        self.steps = steps

    def __iter__(self):
        for _ in range(self.steps):
            b = random.choices(self.keys, weights=self.weights)[0]
            n = min(BATCH[BUCKETS[b]], len(self.groups[b]))
            yield random.sample(self.groups[b], n)

    def __len__(self):
        return self.steps


def make_grid(images, cols=8, scale=None):
    n, _, h, w = images.shape
    scale = scale or max(1, 256 // h)
    rows = math.ceil(n / cols)
    grid = torch.ones(4, rows * h, cols * w)
    for i in range(n):
        r, c = divmod(i, cols)
        grid[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = images[i]
    rgb, alpha = grid[:3], grid[3:4].clamp(0, 1)
    yy, xx = torch.meshgrid(torch.arange(rows * h), torch.arange(cols * w), indexing="ij")
    cell = max(1, h // 4)
    checker = (((yy // cell + xx // cell) % 2) * 0.12 + 0.82).unsqueeze(0)
    out = rgb * alpha + checker * (1 - alpha)
    arr = (out.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr).resize((grid.shape[2] * scale, grid.shape[1] * scale), Image.NEAREST)


@torch.no_grad()
def embed_text(texts, tokenizer, encoder, device):
    tok = tokenizer(texts, padding="max_length", max_length=77, truncation=True,
                    return_tensors="pt").to(device)
    return encoder(**tok).last_hidden_state


@torch.no_grad()
def sample(model, scheduler, cond, uncond, size, device="cuda", steps=100, cfg=4.0, seed=0):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--out", default="workdir/v8_imgcond")
    p.add_argument("--sample_every", type=int, default=2500)
    p.add_argument("--init_v7", default=None, help="v7 checkpoint (7-bucket) to start from")
    p.add_argument("--init", default=None, help="resume a v8 checkpoint (unet+proj)")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bs_scale", type=float, default=1.0)
    p.add_argument("--p_img", type=float, default=0.5, help="fraction of batches carrying image tokens")
    p.add_argument("--csv_suffix", default="")
    p.add_argument("--strong_aug", action="store_true",
                   help="wider conditioning-view degradation (counters self-conditioning drift)")
    args = p.parse_args()
    for k in BATCH:
        BATCH[k] = max(8, int(BATCH[k] * args.bs_scale))
    device = "cuda"
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    suf = args.csv_suffix
    sources = [
        ("data/oga_clean", f"data/oga_captions{suf}.csv", 1),
        ("data/extra_all", f"data/extra_all{suf}.csv", 1),
        ("data/oga_clean", f"data/tool_candidates{suf}.csv", 2),
        ("data/bowtool_items", "data/bowtool_focus.csv", 6),
        ("data/bowtool_items", "data/bowtool_captions.csv", 2),
    ]
    sources = [s for s in sources if Path(s[1]).exists()]
    ds = NativeSprites(sources, strong_aug=args.strong_aug)
    counts = [ds.bucket_of.count(b) for b in range(len(BUCKETS))]
    print(f"dataset: {len(ds)} buckets {dict(zip(BUCKETS, counts))}", flush=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=BucketSampler(ds.bucket_of, args.steps), num_workers=10, pin_memory=True)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    text_encoder.requires_grad_(False)
    vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    vision.requires_grad_(False)

    model = UNet2DConditionModel(
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(128, 256, 512), cross_attention_dim=512,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        num_class_embeds=len(BUCKETS),
    ).to(device)
    # projects CLIP vision tokens (768) into the text cross-attention space (512)
    img_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 512)).to(device)

    if args.init_v7:
        sd = torch.load(args.init_v7, map_location=device)
        model.load_state_dict(sd)
        print(f"unet init from {args.init_v7}", flush=True)
    elif args.init:
        ck = torch.load(args.init, map_location=device)
        model.load_state_dict(ck["unet"])
        img_proj.load_state_dict(ck["img_proj"])
        print(f"resumed from {args.init}", flush=True)
    print(f"unet {sum(q.numel() for q in model.parameters())/1e6:.1f}M + "
          f"proj {sum(q.numel() for q in img_proj.parameters())/1e6:.2f}M", flush=True)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    opt = torch.optim.AdamW(list(model.parameters()) + list(img_proj.parameters()), lr=args.lr)
    ema = None
    if args.ema > 0:
        import copy
        ema = copy.deepcopy(model).eval().requires_grad_(False)

    eval_cond = embed_text(EVAL_PROMPTS, tokenizer, text_encoder, device)
    eval_uncond = embed_text([""] * len(EVAL_PROMPTS), tokenizer, text_encoder, device)

    step = 0
    for x, texts, b, cimg in loader:
        texts = ["" if random.random() < 0.1 else t for t in texts]
        cond = embed_text(list(texts), tokenizer, text_encoder, device)
        # per-BATCH decision: a text-only batch runs the exact v7 computation
        if random.random() < args.p_img:
            with torch.no_grad():
                vtok = vision(pixel_values=cimg.to(device)).last_hidden_state
            cond = torch.cat([cond, img_proj(vtok)], dim=1)
        x, b = x.to(device), b.to(device)
        noise = torch.randn_like(x)
        t = torch.randint(0, 1000, (x.shape[0],), device=device)
        pred = model(scheduler.add_noise(x, noise, t), t,
                     encoder_hidden_states=cond, class_labels=b).sample
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if ema is not None:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1.0 - args.ema)
        if step % 200 == 0:
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} "
                  f"bucket={BUCKETS[int(b[0])]} img={'Y' if cond.shape[1] > 77 else 'N'}", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            net = ema if ema is not None else model
            net.eval()
            for s in EVAL_SIZES:                     # text-only grid, comparable to v7
                torch.manual_seed(args.seed)
                make_grid(sample(net, scheduler, eval_cond, eval_uncond, s, seed=args.seed)).save(
                    out / "samples" / f"step_{step:06d}_s{s}.png")
            torch.save({"unet": net.state_dict(), "img_proj": img_proj.state_dict()},
                       out / "model_latest.pt")
            model.train()
    print(f"Done -> {out}", flush=True)


if __name__ == "__main__":
    main()
