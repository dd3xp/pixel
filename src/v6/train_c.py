"""V6 stage-c: caption-conditioned sprite generation at 32x32.
Data: LPC 4-view sheets (128x128, 2x2) -> front view (top-left 64) -> 32.
Text: frozen CLIP text encoder -> cross-attention (IP-Adapter-style decoupling
comes later; v1 is plain text conditioning + classifier-free guidance).

Usage: python src/v6/train_c.py [--steps 30000]
"""
import argparse
import csv
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

EVAL_PROMPTS = [
    "a pixel art fantasy rpg character, male body template, dark elf, purple skin, leather chest armor",
    "a pixel art fantasy rpg character, female body template, orc, green skin, red dress, white long hair",
    "a pixel art fantasy rpg character, male body template, human, light skin, gold plate armor, metal helm, spear",
    "a pixel art fantasy rpg character, female body template, human, dark skin, blue robe, blonde princess hairstyle",
    "a pixel art fantasy rpg character, male body template, skeleton, bone white, black cloak",
    "a pixel art fantasy rpg character, female body template, elf, pale skin, green tunic, red long bangs hairstyle, bow",
    "a pixel art fantasy rpg character, male body template, human, tan skin, maroon long-sleeve shirt, leather pants",
    "a pixel art fantasy rpg character, female body template, dark elf, purple skin, white dress, silver hair",
]


class LPCFront(torch.utils.data.Dataset):
    def __init__(self, img_dir: str, captions_csv: str, size: int = 32):
        self.img_dir = Path(img_dir)
        self.size = size
        self.rows = []
        with open(captions_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append((row["image_path"], row["text"].split(", four views")[0]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name, text = self.rows[i]
        im = Image.open(self.img_dir / name).convert("RGB")
        im = im.crop((0, im.height // 2, im.width // 2, im.height))  # front view (bottom-left; TL is back)
        im = im.resize((self.size, self.size), Image.NEAREST)
        t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        t = t.view(self.size, self.size, 3).permute(2, 0, 1).float() / 127.5 - 1.0
        return t, text


def make_grid(images, cols=8, scale=8):
    n, _, h, w = images.shape
    rows = math.ceil(n / cols)
    grid = torch.ones(3, rows * h, cols * w)
    for i in range(n):
        r, c = divmod(i, cols)
        grid[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = images[i]
    arr = (grid.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    img = Image.fromarray(arr)
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


@torch.no_grad()
def embed(texts, tokenizer, encoder, device):
    tok = tokenizer(texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
    return encoder(**tok).last_hidden_state


@torch.no_grad()
def sample(model, scheduler, cond, uncond, size=32, device="cuda", steps=100, cfg=4.0):
    scheduler.set_timesteps(steps)
    n = cond.shape[0]
    x = torch.randn(n, 3, size, size, device=device)
    for t in scheduler.timesteps:
        e_c = model(x, t, encoder_hidden_states=cond).sample
        e_u = model(x, t, encoder_hidden_states=uncond).sample
        eps = e_u + cfg * (e_c - e_u)
        x = scheduler.step(eps, t, x).prev_sample
    return ((x + 1) / 2).clamp(0, 1).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--out", default="workdir/v6c_lpc32")
    p.add_argument("--sample_every", type=int, default=2000)
    args = p.parse_args()
    device = "cuda"
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    ds = LPCFront("data/lpc_images/train", "data/lpc-4view-pixel-art-diffusion/captions/captions.csv", args.size)
    print(f"dataset: {len(ds)}", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=8, drop_last=True)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    text_encoder.requires_grad_(False)

    model = UNet2DConditionModel(
        sample_size=args.size, in_channels=3, out_channels=3, layers_per_block=2,
        block_out_channels=(64, 128, 256), cross_attention_dim=512,
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
