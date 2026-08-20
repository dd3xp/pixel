"""V6 stage-d: caption-conditioned 16x16 RGBA sprite/item generation.
Data: cleaned OGA-CC0 sprites (data/oga_clean) + BLIP captions
(data/oga_captions.csv). 4-channel DDPM (RGB+A); RGB is zeroed where alpha=0
so the model never fits garbage under transparency; alpha is supervised as a
plain 4th channel.

Usage: CUDA_VISIBLE_DEVICES=0 python src/v6/train_d.py [--steps 60000]
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


class OGAItems(torch.utils.data.Dataset):
    def __init__(self, img_dir: str, captions_csv: str, size: int = 16):
        self.img_dir = Path(img_dir)
        self.size = size
        self.rows = []
        with open(captions_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append((row["path"], row["text"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name, text = self.rows[i]
        im = Image.open(self.img_dir / name).convert("RGBA")
        side = max(im.size)
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        im = canvas.resize((self.size, self.size), Image.NEAREST)
        a = np.array(im).astype(np.float32)
        a[a[:, :, 3] < 128] = 0.0  # zero RGB (and snap alpha) under transparency
        a[:, :, 3] = (a[:, :, 3] >= 128) * 255.0
        t = torch.from_numpy(a).permute(2, 0, 1) / 127.5 - 1.0
        return t, text


def make_grid(images, cols=8, scale=16):
    """images: (n,4,h,w) in [0,1]. Composite over a light checkerboard."""
    n, _, h, w = images.shape
    rows = math.ceil(n / cols)
    grid = torch.ones(4, rows * h, cols * w)
    for i in range(n):
        r, c = divmod(i, cols)
        grid[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = images[i]
    rgb, alpha = grid[:3], grid[3:4].clamp(0, 1)
    yy, xx = torch.meshgrid(torch.arange(rows * h), torch.arange(cols * w), indexing="ij")
    checker = (((yy // 4 + xx // 4) % 2) * 0.12 + 0.82).unsqueeze(0)
    out = rgb * alpha + checker * (1 - alpha)
    arr = (out.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    img = Image.fromarray(arr)
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


@torch.no_grad()
def embed(texts, tokenizer, encoder, device):
    tok = tokenizer(texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
    return encoder(**tok).last_hidden_state


@torch.no_grad()
def sample(model, scheduler, cond, uncond, size=16, device="cuda", steps=100, cfg=4.0):
    scheduler.set_timesteps(steps)
    n = cond.shape[0]
    x = torch.randn(n, 4, size, size, device=device)
    for t in scheduler.timesteps:
        e_c = model(x, t, encoder_hidden_states=cond).sample
        e_u = model(x, t, encoder_hidden_states=uncond).sample
        eps = e_u + cfg * (e_c - e_u)
        x = scheduler.step(eps, t, x).prev_sample
    return ((x + 1) / 2).clamp(0, 1).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--out", default="workdir/v6d_items16")
    p.add_argument("--sample_every", type=int, default=2000)
    args = p.parse_args()
    device = "cuda"
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    ds = OGAItems("data/oga_clean", "data/oga_captions.csv", args.size)
    print(f"dataset: {len(ds)}", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=8, drop_last=True)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    text_encoder.requires_grad_(False)

    model = UNet2DConditionModel(
        sample_size=args.size, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(128, 256, 512), cross_attention_dim=512,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
    ).to(device)
    print(f"params: {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M", flush=True)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    eval_cond = embed(EVAL_PROMPTS, tokenizer, text_encoder, device)
    eval_uncond = embed([""] * len(EVAL_PROMPTS), tokenizer, text_encoder, device)

    step = 0
    while step < args.steps:
        for x, texts in loader:
            texts = ["" if random.random() < 0.1 else t for t in texts]  # CFG dropout
            cond = embed(list(texts), tokenizer, text_encoder, device)
            x = x.to(device)
            noise = torch.randn_like(x)
            t = torch.randint(0, 1000, (x.shape[0],), device=device)
            loss = F.mse_loss(model(scheduler.add_noise(x, noise, t), t, encoder_hidden_states=cond).sample, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                print(f"[{step}/{args.steps}] loss={loss.item():.4f}", flush=True)
            if step % args.sample_every == 0 or step == args.steps:
                model.eval()
                make_grid(sample(model, scheduler, eval_cond, eval_uncond, size=args.size)).save(
                    out / "samples" / f"step_{step:06d}.png")
                torch.save(model.state_dict(), out / "model_latest.pt")
                model.train()
            if step >= args.steps:
                break
    print(f"Done -> {out}", flush=True)


if __name__ == "__main__":
    main()
