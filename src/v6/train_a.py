"""V6 Model A: TinyUNet DDPM trained natively at 32x32 on real sprite data.
First target: unconditional quality on jiovine/pixel-art-nouns (49.9k, CC0).

Usage: python src/v6/train_a.py [--steps 30000] [--bs 256] [--out workdir/v6a_nouns]
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image


def make_grid(images: torch.Tensor, cols: int = 8, scale: int = 4) -> Image.Image:
    """images: (N,3,H,W) in [0,1] -> nearest-upscaled grid image."""
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
def sample(model, scheduler, n=64, size=32, device="cuda", steps=100) -> torch.Tensor:
    scheduler.set_timesteps(steps)
    x = torch.randn(n, 3, size, size, device=device)
    for t in scheduler.timesteps:
        eps = model(x, t).sample
        x = scheduler.step(eps, t, x).prev_sample
    return ((x + 1) / 2).clamp(0, 1).cpu()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="jiovine/pixel-art-nouns")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--out", default="workdir/v6a_nouns")
    p.add_argument("--sample_every", type=int, default=2000)
    args = p.parse_args()

    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    device = "cuda"

    ds = load_dataset(args.dataset, split="train")
    img_col = "image" if "image" in ds.column_names else ds.column_names[0]

    def to_tensor(batch):
        outs = []
        for im in batch[img_col]:
            im = im.convert("RGB").resize((args.size, args.size), Image.NEAREST)
            t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
            t = t.view(args.size, args.size, 3).permute(2, 0, 1).float() / 127.5 - 1.0
            outs.append(t)
        return {"pixel_values": outs}

    ds = ds.with_transform(to_tensor)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.bs, shuffle=True, num_workers=8, drop_last=True,
        collate_fn=lambda b: torch.stack([x["pixel_values"] for x in b]),
    )

    model = UNet2DModel(
        sample_size=args.size, in_channels=3, out_channels=3, layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    while step < args.steps:
        for batch in loader:
            x = batch.to(device)
            noise = torch.randn_like(x)
            t = torch.randint(0, 1000, (x.shape[0],), device=device)
            noisy = scheduler.add_noise(x, noise, t)
            loss = F.mse_loss(model(noisy, t).sample, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                print(f"[{step}/{args.steps}] loss={loss.item():.4f}", flush=True)
            if step % args.sample_every == 0 or step == args.steps:
                model.eval()
                grid = make_grid(sample(model, scheduler, device=device, size=args.size))
                grid.save(out / "samples" / f"step_{step:06d}.png")
                torch.save(model.state_dict(), out / "model_latest.pt")
                model.train()
            if step >= args.steps:
                break
    print(f"Done -> {out}", flush=True)


if __name__ == "__main__":
    main()
