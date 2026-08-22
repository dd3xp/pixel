"""V6 stage-e: ONE model, many resolutions. Resolution-conditioned RGBA sprite
diffusion trained on sprites at their NATIVE size (never resampled): each sprite
is padded (centered, transparent) into the smallest bucket >= its size
(16/24/32/48/64). Batches are single-bucket; the bucket index is fed as a class
embedding so the net learns per-resolution pixel grammar. Text via CLIP
cross-attention + CFG. Eval grids at 16/32/64 for the same 8 prompts.

Usage: CUDA_VISIBLE_DEVICES=1 python src/v6/train_e.py [--steps 80000]
"""
import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

BUCKETS = [16, 24, 32, 48, 64]
BATCH = {16: 256, 24: 192, 32: 128, 48: 64, 64: 40}
EVAL_SIZES = [16, 32, 64]
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


def downscale_rgba(im, side):
    """Premultiplied box downsample so transparent pixels don't bleed color."""
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
    if max(im.size) > side:  # oversize (rare pad overflow) or multi-scale aug copy
        im = downscale_rgba(im, side)
    elif max(im.size) * 2 <= side:  # ms_up copy: lossless integer NEAREST upscale
        f = side // max(im.size)
        im = im.resize((im.width * f, im.height * f), Image.NEAREST)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    a = np.array(canvas).astype(np.float32)
    a[a[:, :, 3] < 128] = 0.0
    a[:, :, 3] = (a[:, :, 3] >= 128) * 255.0
    return torch.from_numpy(a).permute(2, 0, 1) / 127.5 - 1.0


class NativeSprites(torch.utils.data.Dataset):
    """sources: list of (img_dir, captions_csv, repeat)."""

    def __init__(self, sources, ms_aug=False, ms_up=False):
        self.rows, self.bucket_of = [], []
        for img_dir, captions_csv, repeat in sources:
            img_dir = Path(img_dir)
            with open(captions_csv, newline="", encoding="utf-8") as f:
                rows = [(img_dir / row["path"], row["text"]) for row in csv.DictReader(f)]
            n_aug = 0
            for path, text in rows:  # header-only open, fast
                s = max(Image.open(path).size)
                b = next((i for i, b in enumerate(BUCKETS) if b >= s), len(BUCKETS) - 1)
                self.rows.extend([(path, text)] * repeat)
                self.bucket_of.extend([b] * repeat)
                if ms_aug and b >= 2:  # 32+ sprite -> extra 16px copy (multi-scale augmentation)
                    self.rows.extend([(path, text)] * repeat)
                    self.bucket_of.extend([0] * repeat)
                    n_aug += repeat
                if ms_up and s <= 16:  # <=16px sprite -> extra x2 NEAREST copy in the 32 bucket
                    self.rows.extend([(path, text)] * repeat)
                    self.bucket_of.extend([2] * repeat)
                    n_aug += repeat
            print(f"source {img_dir}: {len(rows)} x{repeat} (+{n_aug} ms-aug@16)", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, text = self.rows[i]
        b = self.bucket_of[i]
        im = Image.open(path).convert("RGBA")
        return to_tensor(im, BUCKETS[b]), text, b


class BucketSampler(torch.utils.data.Sampler):
    """Single-bucket batches; bucket drawn proportional to its sample count."""

    def __init__(self, bucket_of, steps):
        self.groups = {b: [i for i, x in enumerate(bucket_of) if x == b] for b in range(len(BUCKETS))}
        self.weights = [len(self.groups[b]) for b in range(len(BUCKETS))]
        self.steps = steps

    def __iter__(self):
        for _ in range(self.steps):
            b = random.choices(range(len(BUCKETS)), weights=self.weights)[0]
            yield random.sample(self.groups[b], BATCH[BUCKETS[b]])

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
    img = Image.fromarray(arr)
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


@torch.no_grad()
def embed(texts, tokenizer, encoder, device):
    tok = tokenizer(texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
    return encoder(**tok).last_hidden_state


@torch.no_grad()
def sample(model, scheduler, cond, uncond, size, device="cuda", steps=100, cfg=4.0):
    scheduler.set_timesteps(steps)
    n = cond.shape[0]
    lab = torch.full((n,), BUCKETS.index(size), device=device, dtype=torch.long)
    x = torch.randn(n, 4, size, size, device=device)
    for t in scheduler.timesteps:
        e_c = model(x, t, encoder_hidden_states=cond, class_labels=lab).sample
        e_u = model(x, t, encoder_hidden_states=uncond, class_labels=lab).sample
        x = scheduler.step(e_u + cfg * (e_c - e_u), t, x).prev_sample
    return ((x + 1) / 2).clamp(0, 1).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default="workdir/v6e_multires")
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--init", default=None, help="state_dict to fine-tune from")
    p.add_argument("--extra", default=None, help="extra source: img_dir,captions_csv,repeat")
    p.add_argument("--seed", type=int, default=0, help="fixed eval sampling seed")
    p.add_argument("--bs_scale", type=float, default=1.0, help="multiply per-bucket batch sizes (memory knob)")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay for eval/save weights (0=off, e.g. 0.999)")
    p.add_argument("--ms_aug", action="store_true", help="also feed 32+ sprites downscaled into the 16 bucket")
    p.add_argument("--ms_up", action="store_true", help="also feed <=16px sprites x2 NEAREST-upscaled into the 32 bucket")
    p.add_argument("--extra2", default=None, help="second extra source: img_dir,captions_csv,repeat")
    args = p.parse_args()
    for k in BATCH:
        BATCH[k] = max(8, int(BATCH[k] * args.bs_scale))
    device = "cuda"
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    sources = [("data/oga_clean", "data/oga_captions.csv", 1)]
    if args.extra:
        d, c, r = args.extra.split(",")
        sources.append((d, c, int(r)))
    if args.extra2:
        d, c, r = args.extra2.split(",")
        sources.append((d, c, int(r)))
    ds = NativeSprites(sources, ms_aug=args.ms_aug, ms_up=args.ms_up)
    counts = [ds.bucket_of.count(b) for b in range(len(BUCKETS))]
    print(f"dataset: {len(ds)} buckets {dict(zip(BUCKETS, counts))}", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=BucketSampler(ds.bucket_of, args.steps), num_workers=8)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    text_encoder.requires_grad_(False)

    model = UNet2DConditionModel(
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(128, 256, 512), cross_attention_dim=512,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        num_class_embeds=len(BUCKETS),
    ).to(device)
    print(f"params: {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M", flush=True)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"init from {args.init}", flush=True)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = None
    if args.ema > 0:
        import copy
        ema = copy.deepcopy(model).eval().requires_grad_(False)
    eval_cond = embed(EVAL_PROMPTS, tokenizer, text_encoder, device)
    eval_uncond = embed([""] * len(EVAL_PROMPTS), tokenizer, text_encoder, device)

    step = 0
    for x, texts, b in loader:
        texts = ["" if random.random() < 0.1 else t for t in texts]
        cond = embed(list(texts), tokenizer, text_encoder, device)
        x, b = x.to(device), b.to(device)
        noise = torch.randn_like(x)
        t = torch.randint(0, 1000, (x.shape[0],), device=device)
        pred = model(scheduler.add_noise(x, noise, t), t, encoder_hidden_states=cond, class_labels=b).sample
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
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} bucket={BUCKETS[int(b[0])]}", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            net = ema if ema is not None else model
            net.eval()
            for s in EVAL_SIZES:
                torch.manual_seed(args.seed)  # fixed seed -> comparable grids across checkpoints
                make_grid(sample(net, scheduler, eval_cond, eval_uncond, s)).save(
                    out / "samples" / f"step_{step:06d}_s{s}.png")
            torch.save(net.state_dict(), out / "model_latest.pt")
            model.train()
    print(f"Done -> {out}", flush=True)


if __name__ == "__main__":
    main()
