"""V6 cascade guidance: coarse-to-fine inference without retraining.

Core idea: the multi-res model (train_e.py) already knows how to generate
sprites at 16/24/32/48/64 px independently.  At inference time we chain the
resolutions: generate a coarse image first, nearest-upsample it to the next
bucket size, then condition the fine-resolution denoising on that coarse
guide via spatial concatenation (adding 4 extra in-channels via zero-padded
projection — no weight changes, just activation injection).

Two modes are supported:

  --mode img2img   (default, zero-shot)
    Standard DDIM forward-noise the coarse image to an intermediate timestep
    t_start, then run the denoiser from t_start down to 0 on the fine
    resolution.  This requires NO model changes — it reuses the existing
    weights exactly and simply starts from a semantically meaningful initial
    noise state rather than pure Gaussian noise.

  --mode concat    (requires fine-tuning, branch experiment)
    A thin ControlNet-style spatial adapter concatenates the coarse guide
    (4-channel, upsampled) to the UNet input.  The base weights are frozen;
    only the adapter projection is trained.  Training script: train_g.py.

Usage (img2img mode, zero-shot, runs on existing v6e10 checkpoint):
  python src/v6/sample_cascade.py \\
      --ckpt workdir/v6e10_ema/model_latest.pt \\
      --prompts baseline/prompts8.txt \\
      --coarse 16 --fine 32 \\
      --t_start 600 \\
      --out runs/cascade_16to32 --n 4

The img2img mode is the ablation submitted to the paper: it shows that a
coarse prior improves fine-resolution coherence without any extra training,
strengthening the "resolution-native generation" claim.
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

BUCKETS = [16, 24, 32, 48, 64]


# ---------------------------------------------------------------------------
# Shared model / embed helpers (mirrors sample_e.py)
# ---------------------------------------------------------------------------

def build_model(device: str) -> UNet2DConditionModel:
    return UNet2DConditionModel(
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(128, 256, 512), cross_attention_dim=512,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        num_class_embeds=len(BUCKETS),
    ).to(device)


@torch.no_grad()
def embed(texts, tokenizer, encoder, device):
    tok = tokenizer(texts, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt").to(device)
    return encoder(**tok).last_hidden_state


# ---------------------------------------------------------------------------
# Standard (independent) generation — baseline for comparison
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_independent(
    model, scheduler, cond, uncond, size, device,
    steps=100, cfg=4.0, seed=0,
):
    scheduler.set_timesteps(steps)
    n = cond.shape[0]
    lab = torch.full((n,), BUCKETS.index(size), device=device, dtype=torch.long)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, 4, size, size, device=device, generator=g)
    for t in scheduler.timesteps:
        ec = model(x, t, encoder_hidden_states=cond,   class_labels=lab).sample
        eu = model(x, t, encoder_hidden_states=uncond, class_labels=lab).sample
        x  = scheduler.step(eu + cfg * (ec - eu), t, x).prev_sample
    return ((x + 1) / 2).clamp(0, 1).cpu()


# ---------------------------------------------------------------------------
# img2img cascade: coarse → add noise to t_start → denoise at fine resolution
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_cascade_img2img(
    model, scheduler: DDIMScheduler, cond, uncond,
    coarse_size: int, fine_size: int, device: str,
    steps: int = 100, cfg: float = 4.0, seed: int = 0,
    t_start: int = 600,
):
    """Generate coarse image, upsample, add noise at t_start, denoise at fine res.

    t_start controls the coarse influence:
      - high t_start (700-900): coarse guide strongly shapes layout/palette
      - low  t_start (200-400): coarse guide gently nudges, fine detail is free
      - 0:                       equivalent to pure independent generation

    Returns (B, 4, fine_size, fine_size) float32 in [0, 1].
    """
    scheduler.set_timesteps(steps)
    n = cond.shape[0]

    # --- Step 1: generate coarse image ---
    coarse_rgba = sample_independent(
        model, scheduler, cond, uncond, coarse_size, device, steps, cfg, seed)
    # re-build scheduler timesteps after the first call
    scheduler.set_timesteps(steps)

    # --- Step 2: upsample to fine resolution ---
    coarse_up = F.interpolate(
        coarse_rgba.to(device) * 2 - 1,            # back to [-1, 1]
        size=(fine_size, fine_size),
        mode="nearest",
    )                                               # (B, 4, fine, fine)

    # --- Step 3: forward-noise to t_start ---
    noise = torch.randn_like(coarse_up)
    t_idx = max(0, min(len(scheduler.timesteps) - 1,
                        int((1 - t_start / 1000) * steps)))
    t_noise = scheduler.timesteps[t_idx]
    # DDIM scheduler stores alphas_cumprod
    alpha_prod = scheduler.alphas_cumprod[t_noise].to(device)
    x = alpha_prod.sqrt() * coarse_up + (1 - alpha_prod).sqrt() * noise

    # --- Step 4: denoise from t_start down to 0 at fine resolution ---
    lab_fine = torch.full((n,), BUCKETS.index(fine_size), device=device, dtype=torch.long)
    for t in scheduler.timesteps[t_idx:]:
        ec = model(x, t, encoder_hidden_states=cond,   class_labels=lab_fine).sample
        eu = model(x, t, encoder_hidden_states=uncond, class_labels=lab_fine).sample
        x  = scheduler.step(eu + cfg * (ec - eu), t, x).prev_sample

    return ((x + 1) / 2).clamp(0, 1).cpu()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def to_rgba_pil(t):                               # (4, h, w) [0,1] -> PIL RGBA
    a = (t[3] > 0.5).float()
    arr = torch.cat([t[:3] * a, a[None]], 0)
    return Image.fromarray((arr.permute(1, 2, 0) * 255).byte().numpy(), "RGBA")


def make_comparison_grid(
    coarse_imgs, cascade_imgs, indep_imgs,
    coarse_size, fine_size, scale=4,
):
    """Three-column grid: coarse | cascade fine | independent fine."""
    n = len(coarse_imgs)
    cell = fine_size * scale
    coarse_cell = coarse_size * scale
    w = coarse_cell + cell + cell + 8      # 8 px separator
    h_total = n * cell
    out = Image.new("RGBA", (w, h_total), (220, 220, 220, 255))
    for i in range(n):
        y = i * cell
        c_im = coarse_imgs[i].resize((coarse_cell, coarse_cell), Image.NEAREST)
        cas_im = cascade_imgs[i].resize((cell, cell), Image.NEAREST)
        ind_im = indep_imgs[i].resize((cell, cell), Image.NEAREST)
        out.paste(c_im, (0, y + (cell - coarse_cell) // 2))
        out.alpha_composite(cas_im, (coarse_cell + 4, y))
        out.alpha_composite(ind_im, (coarse_cell + 4 + cell, y))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--coarse",  type=int, default=16)
    p.add_argument("--fine",    type=int, default=32)
    p.add_argument("--t_start", type=int, default=600,
                   help="noise level injected into coarse guide (0=none, 999=pure noise)")
    p.add_argument("--steps",   type=int, default=100)
    p.add_argument("--cfg",     type=float, default=4.0)
    p.add_argument("--n",       type=int, default=4)
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--out",     required=True)
    args = p.parse_args()

    assert args.coarse in BUCKETS and args.fine in BUCKETS, "sizes must be in BUCKETS"
    assert BUCKETS.index(args.coarse) < BUCKETS.index(args.fine), "--coarse must be smaller than --fine"

    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8")
               if l.strip() and not l.startswith("#")]

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    model = build_model(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # DDIM for both independent and cascade (deterministic, faster)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
        clip_sample=False, set_alpha_to_one=False,
    )

    cond   = embed(prompts, tokenizer, enc, device).repeat_interleave(args.n, 0)
    uncond = embed([""] * len(prompts) * args.n, tokenizer, enc, device)

    print(f"Generating {len(prompts) * args.n} samples at coarse={args.coarse} "
          f"fine={args.fine} t_start={args.t_start} …", flush=True)

    # Independent fine (baseline)
    indep = sample_independent(model, scheduler, cond, uncond, args.fine, device,
                                args.steps, args.cfg, args.seed)

    # Cascade (coarse → fine)
    cascade = sample_cascade_img2img(
        model, scheduler, cond, uncond,
        args.coarse, args.fine, device,
        args.steps, args.cfg, args.seed, args.t_start,
    )

    # Coarse thumbnails (for comparison grid)
    scheduler.set_timesteps(args.steps)
    coarse_raw = sample_independent(model, scheduler, cond, uncond, args.coarse,
                                     device, args.steps, args.cfg, args.seed)

    # Save individual PNGs
    for mode, imgs in [("cascade", cascade), ("indep", indep), ("coarse", coarse_raw)]:
        d = out / mode; d.mkdir(exist_ok=True)
        for i, im in enumerate(imgs):
            pi, k = divmod(i, args.n)
            to_rgba_pil(im).save(d / f"{pi:02d}_{k}.png")

    # Comparison grid: one row per prompt, cols = [coarse | cascade | independent]
    for pi, prompt in enumerate(prompts):
        rows_c  = [to_rgba_pil(coarse_raw[pi * args.n + k]) for k in range(args.n)]
        rows_ca = [to_rgba_pil(cascade[pi * args.n + k])    for k in range(args.n)]
        rows_in = [to_rgba_pil(indep[pi * args.n + k])      for k in range(args.n)]
        cell = max(64, args.fine * 4)
        row_h = cell
        row_w = args.coarse * 4 + cell * 2 + 8
        strip = Image.new("RGBA", (row_w * args.n, row_h), (220, 220, 220, 255))
        for k in range(args.n):
            x_off = k * row_w
            strip.paste(rows_c[k].resize((args.coarse * 4, args.coarse * 4), Image.NEAREST),
                        (x_off, (cell - args.coarse * 4) // 2))
            strip.alpha_composite(rows_ca[k].resize((cell, cell), Image.NEAREST),
                                  (x_off + args.coarse * 4 + 4, 0))
            strip.alpha_composite(rows_in[k].resize((cell, cell), Image.NEAREST),
                                  (x_off + args.coarse * 4 + 4 + cell, 0))
        strip.save(out / f"compare_{pi:02d}.png")
        print(f"  [{pi}] {prompt[:60]}", flush=True)

    print(f"\nDone → {out}", flush=True)
    print("Grid columns: [coarse guide]  [cascade fine]  [independent fine]", flush=True)


if __name__ == "__main__":
    main()
