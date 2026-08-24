"""Sample the multi-res model (train_e.py) for arbitrary prompts x sizes.
Outputs one grid per size (rows = prompts, cols = n samples) composited on a
checkerboard, plus raw RGBA PNGs per sample for direct use as textures.

Usage:
  python src/v6/sample_e.py --ckpt workdir/v6e2_kenney/model_latest.pt \
      --prompts prompts/mc_items.txt --sizes 16 32 --n 4 --out runs_out/mc16
"""
import argparse
import math
from pathlib import Path

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

BUCKETS = [16, 24, 32, 48, 64]  # overridden by --buckets


def build_model(device):
    return UNet2DConditionModel(
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(128, 256, 512), cross_attention_dim=512,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        num_class_embeds=len(BUCKETS),
    ).to(device)


@torch.no_grad()
def embed(texts, tokenizer, encoder, device):
    tok = tokenizer(texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
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


def to_rgba(img):  # (4,h,w) in [0,1] -> PIL RGBA with hard alpha
    a = (img[3] > 0.5).float()
    rgb = img[:3] * a  # clear RGB under transparency
    arr = torch.cat([rgb, a[None]], 0)
    return Image.fromarray((arr.permute(1, 2, 0) * 255).byte().numpy(), "RGBA")


def grid(images, rows, cols, scale):
    h, w = images[0].size[1], images[0].size[0]
    cell = max(1, h // 4)
    out = Image.new("RGBA", (cols * w, rows * h))
    px = out.load()
    for y in range(rows * h):
        for x in range(cols * w):
            v = 209 if ((y // cell + x // cell) % 2) else 240
            px[x, y] = (v, v, v, 255)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        out.alpha_composite(im, (c * w, r * h))
    return out.resize((out.width * scale, out.height * scale), Image.NEAREST)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompts", required=True, help="text file, one prompt per line")
    p.add_argument("--sizes", type=int, nargs="+", default=[16])
    p.add_argument("--n", type=int, default=4, help="samples per prompt")
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--buckets", default=None, help="comma list, e.g. 12,16,20,24,32,48,64 for v7 models")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.buckets:
        BUCKETS.clear()
        BUCKETS.extend(int(v) for v in args.buckets.split(","))
    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip() and not l.startswith("#")]
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    model = build_model(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    cond = embed(prompts, tokenizer, enc, device).repeat_interleave(args.n, 0)
    uncond = embed([""] * len(prompts) * args.n, tokenizer, enc, device)
    for size in args.sizes:
        imgs = sample(model, scheduler, cond, uncond, size, device, args.steps, args.cfg, args.seed)
        rgba = [to_rgba(im) for im in imgs]
        for i, im in enumerate(rgba):
            pi, k = divmod(i, args.n)
            (out / f"s{size}").mkdir(exist_ok=True)
            im.save(out / f"s{size}" / f"{pi:02d}_{k}.png")
        grid(rgba, len(prompts), args.n, max(1, 128 // size)).save(out / f"grid_s{size}.png")
        print(f"size {size}: {len(rgba)} samples -> {out}/grid_s{size}.png", flush=True)
    (out / "prompts.txt").write_text("\n".join(prompts), encoding="utf-8")


if __name__ == "__main__":
    main()
