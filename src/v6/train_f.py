"""V6 stage-f: Discrete palette-index diffusion for pixel art generation.

Core idea: pixel art is inherently discrete — each pixel belongs to one of K
palette colours. Standard continuous diffusion + hard-quantisation at the end
introduces a systematic continuous→discrete mismatch. We instead define the
forward/reverse process DIRECTLY in palette-index space using absorbing/masked
diffusion (D3PM-absorbing / MDLM style):

  Forward:  at step t, each pixel index is independently replaced with a
            special MASK token (index K) with probability t/T.  The un-masked
            positions keep their true index.
  Reverse:  the denoiser predicts p(x₀ | x_t, text, size), i.e. a K-way
            categorical distribution over palette colours for every pixel.
  Loss:     cross-entropy at masked positions only.
  Inference: begin from all-MASK, iteratively unmask following a schedule
             (cosine or linear) for `steps` iterations.

Architecture: same UNet2DConditionModel backbone as train_e.py, but:
  - Input channels = K_palette + 1 (one-hot palette index + MASK flag)
    projected by a learned embedding table; output is K palette logits.
  - Alpha channel treated separately: a 2-way head (transparent / opaque)
    also trained with masked CE.
  - Resolution class-embedding unchanged.

Reference: Austin et al. "Structured Denoising Diffusion Models in Discrete
State-Spaces" (NeurIPS 2021); Shi et al. "Simplified and Generalized Masked
Diffusion for Discrete Data" (2024).

Usage:
  CUDA_VISIBLE_DEVICES=1 python src/v6/train_f.py \
      --palette assets/palettes/dawnbringer32.hex \
      --steps 60000 --lr 3e-4 --out workdir/v6f_discrete
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
from diffusers import UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

BUCKETS = [16, 24, 32, 48, 64]
BATCH = {16: 128, 24: 96, 32: 64, 48: 32, 64: 20}
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

# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

def load_hex_palette(path: str) -> torch.Tensor:
    """Load a .hex palette file; returns (K, 3) float32 in [0, 1]."""
    colors = []
    for line in Path(path).read_text().splitlines():
        line = line.strip().lstrip("#")
        if len(line) == 6:
            r = int(line[0:2], 16) / 255.0
            g = int(line[2:4], 16) / 255.0
            b = int(line[4:6], 16) / 255.0
            colors.append([r, g, b])
    return torch.tensor(colors, dtype=torch.float32)


def quantise_to_palette(img_3hw: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Hard-assign each pixel to nearest palette color; returns (H, W) int64."""
    C, H, W = img_3hw.shape
    px = img_3hw.permute(1, 2, 0).reshape(-1, 3)          # (H*W, 3)
    dists = (px.unsqueeze(1) - palette.unsqueeze(0)).pow(2).sum(-1)  # (H*W, K)
    return dists.argmin(-1).reshape(H, W)                  # (H, W) int64


# ---------------------------------------------------------------------------
# Dataset  (reuses NativeSprites logic from train_e.py, but returns indices)
# ---------------------------------------------------------------------------

def _downscale_rgba(im: Image.Image, side: int) -> Image.Image:
    f = side / max(im.size)
    w, h = max(1, round(im.width * f)), max(1, round(im.height * f))
    a = np.array(im).astype(np.float32)
    a[:, :, :3] *= a[:, :, 3:4] / 255.0
    pm = Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGBA").resize((w, h), Image.BOX)
    b = np.array(pm).astype(np.float32)
    alpha = b[:, :, 3:4]
    b[:, :, :3] = np.where(alpha > 0, b[:, :, :3] / np.maximum(alpha, 1) * 255.0, 0)
    return Image.fromarray(b.clip(0, 255).astype(np.uint8), "RGBA")


class DiscreteSpriteDataset(torch.utils.data.Dataset):
    """Returns (palette_indices, alpha_mask, text, bucket_idx).

    palette_indices: (H, W) int64 — each pixel is its nearest palette index.
    alpha_mask:      (H, W) bool  — True = opaque, False = transparent.
    """

    def __init__(self, sources, palette: torch.Tensor, ms_aug: bool = False, ms_up: bool = False):
        self.palette = palette
        self.rows, self.bucket_of = [], []
        for img_dir, captions_csv, repeat in sources:
            img_dir = Path(img_dir)
            with open(captions_csv, newline="", encoding="utf-8") as f:
                rows = [(img_dir / row["path"], row["text"]) for row in csv.DictReader(f)]
            n_aug = 0
            for path, text in rows:
                s = max(Image.open(path).size)
                b = next((i for i, bb in enumerate(BUCKETS) if bb >= s), len(BUCKETS) - 1)
                self.rows.extend([(path, text)] * repeat)
                self.bucket_of.extend([b] * repeat)
                if ms_aug and b >= 2:
                    self.rows.extend([(path, text)] * repeat)
                    self.bucket_of.extend([0] * repeat)
                    n_aug += repeat
                if ms_up and s <= 16:
                    self.rows.extend([(path, text)] * repeat)
                    self.bucket_of.extend([2] * repeat)
                    n_aug += repeat
            print(f"source {img_dir}: {len(rows)} x{repeat} (+{n_aug} aug)", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, text = self.rows[i]
        b = self.bucket_of[i]
        side = BUCKETS[b]
        im = Image.open(path).convert("RGBA")
        # resize to bucket (same logic as train_e.py)
        if max(im.size) > side:
            im = _downscale_rgba(im, side)
        elif max(im.size) * 2 <= side:
            f = side // max(im.size)
            im = im.resize((im.width * f, im.height * f), Image.NEAREST)
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        arr = np.array(canvas).astype(np.float32)
        alpha = (arr[:, :, 3] >= 128)                      # (H, W) bool
        rgb = torch.from_numpy(arr[:, :, :3] / 255.0).permute(2, 0, 1)  # (3, H, W)
        idx = quantise_to_palette(rgb, self.palette)        # (H, W) int64
        idx[~alpha] = 0                                     # transparent -> colour 0 (irrelevant, masked by alpha)
        return idx, torch.from_numpy(alpha), text, b


class BucketSampler(torch.utils.data.Sampler):
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


# ---------------------------------------------------------------------------
# Discrete diffusion UNet wrapper
# ---------------------------------------------------------------------------

class DiscretePaletteUNet(nn.Module):
    """Wraps UNet2DConditionModel for discrete palette-index diffusion.

    The UNet expects continuous spatial features.  We embed discrete palette
    indices via a learnable table (K+1 entries; index K = MASK token), project
    to `embed_dim` channels, then feed the spatial feature map through the
    standard UNet.  The UNet output (4 channels) is projected back to K+1
    palette logits + 2 alpha logits.

    Input spatial representation:  (B, embed_dim, H, W)
    UNet input channels:            embed_dim  (replaces the usual 4 RGBA channels)
    UNet output channels:           embed_dim
    Palette head output:            K logits   (colour prediction)
    Alpha head output:              2 logits   (transparent / opaque)
    """

    def __init__(self, K: int, embed_dim: int = 16):
        super().__init__()
        self.K = K
        self.MASK = K                           # token index for the MASK state
        self.embed_dim = embed_dim

        # Learnable token table: K colours + 1 MASK
        self.tok_embed = nn.Embedding(K + 1, embed_dim)

        # UNet backbone (same capacity as train_e.py, adjusted in_channels)
        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=embed_dim,
            out_channels=embed_dim,
            layers_per_block=2,
            block_out_channels=(128, 256, 512),
            cross_attention_dim=512,
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            num_class_embeds=len(BUCKETS),
        )

        # Prediction heads
        self.palette_head = nn.Linear(embed_dim, K)    # colour logits
        self.alpha_head   = nn.Linear(embed_dim, 2)    # transparent / opaque

    def forward(self, idx_bhw: torch.Tensor, t: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                class_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        idx_bhw: (B, H, W) int64 — noisy palette indices (may contain MASK=K)
        t:       (B,) int64  — diffusion timestep (used only as UNet conditioning)
        Returns: (palette_logits (B, K, H, W), alpha_logits (B, 2, H, W))
        """
        B, H, W = idx_bhw.shape
        # embed tokens -> (B, H, W, embed_dim) -> (B, embed_dim, H, W)
        feat = self.tok_embed(idx_bhw).permute(0, 3, 1, 2)
        # UNet forward (t is a pseudo-noise-level, reusing the time conditioning)
        out = self.unet(feat, t, encoder_hidden_states=encoder_hidden_states,
                        class_labels=class_labels).sample  # (B, embed_dim, H, W)
        out = out.permute(0, 2, 3, 1)                      # (B, H, W, embed_dim)
        pal_logits   = self.palette_head(out).permute(0, 3, 1, 2)  # (B, K, H, W)
        alpha_logits = self.alpha_head(out).permute(0, 3, 1, 2)    # (B, 2, H, W)
        return pal_logits, alpha_logits


# ---------------------------------------------------------------------------
# Absorbing (masked) diffusion noise schedule
# ---------------------------------------------------------------------------

class AbsorbingSchedule:
    """Linear masking schedule: mask_prob(t) = t / T.

    At t=0: no masks. At t=T-1: almost all masked. The noise level t is a
    continuous integer in [0, T); we sample it uniformly each training step.
    """

    def __init__(self, T: int = 1000, K: int = 32):
        self.T = T
        self.K = K
        self.MASK = K

    def q_sample(self, x0_bhw: torch.Tensor, t_b: torch.Tensor) -> torch.Tensor:
        """Sample noisy indices x_t given clean indices x_0 and timestep t."""
        B, H, W = x0_bhw.shape
        mask_prob = t_b.float() / self.T                    # (B,)
        mask = torch.bernoulli(mask_prob.view(B, 1, 1).expand(B, H, W)).bool()
        return torch.where(mask, torch.full_like(x0_bhw, self.MASK), x0_bhw)

    def loss(self, pal_logits_bkhw: torch.Tensor, alpha_logits_b2hw: torch.Tensor,
             x0_bhw: torch.Tensor, alpha0_bhw: torch.Tensor,
             xt_bhw: torch.Tensor) -> torch.Tensor:
        """CE loss at masked palette positions + CE loss at masked alpha positions.

        We apply the loss at ALL positions (not just masked ones), following the
        MDLM recommendation for stable training — the unmasked positions are
        trivially correct and contribute little gradient.
        """
        B, K, H, W = pal_logits_bkhw.shape
        pal_loss   = F.cross_entropy(pal_logits_bkhw,  x0_bhw,   reduction="mean")
        alpha_loss = F.cross_entropy(alpha_logits_b2hw, alpha0_bhw.long(), reduction="mean")
        return pal_loss + alpha_loss


# ---------------------------------------------------------------------------
# Sampling (iterative unmasking, greedy argmax)
# ---------------------------------------------------------------------------

@torch.no_grad()
def discrete_sample(
    model: DiscretePaletteUNet,
    schedule: AbsorbingSchedule,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    size: int,
    device: str = "cuda",
    steps: int = 64,
    cfg: float = 3.0,
    seed: int = 0,
    palette: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate RGBA images via iterative unmasking.

    Returns (B, 4, H, W) float32 in [0, 1] — R/G/B from palette lookup, A hard.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    B = cond.shape[0]
    lab = torch.full((B,), BUCKETS.index(size), device=device, dtype=torch.long)
    K = schedule.K

    # Start from all-MASK
    x = torch.full((B, size, size), K, device=device, dtype=torch.long)

    # Cosine unmasking schedule: fraction unmasked = cos²(π/2 · (1 - s/S))
    ts = torch.linspace(steps - 1, 0, steps, device=device).long()

    for t_val in ts:
        t_b = t_val.expand(B)
        # CFG: run cond and uncond separately, combine logits
        pal_c, alp_c   = model(x, t_b, cond,   lab)
        pal_u, alp_u   = model(x, t_b, uncond, lab)
        pal_logits = pal_u + cfg * (pal_c - pal_u)   # (B, K, H, W)
        alp_logits = alp_u + cfg * (alp_c - alp_u)

        # At currently-masked positions, sample the predicted colour
        masked = (x == K)
        new_idx = pal_logits.argmax(1)                # greedy; could use categorical
        x = torch.where(masked, new_idx, x)

    # Build RGBA from palette
    alpha_hard = alp_logits.argmax(1).float()         # (B, H, W) 0=transparent, 1=opaque
    if palette is not None:
        pal = palette.to(device)                      # (K, 3)
        # clamp x to valid range (should be after full unmasking)
        x_safe = x.clamp(0, K - 1)
        rgb = pal[x_safe.reshape(-1)].reshape(B, size, size, 3).permute(0, 3, 1, 2)
    else:
        # fallback: uniform grey
        rgb = x.float().clamp(0, K - 1).unsqueeze(1).expand(B, 3, size, size) / (K - 1)

    rgba = torch.cat([rgb * alpha_hard.unsqueeze(1), alpha_hard.unsqueeze(1)], dim=1)
    return rgba.cpu()


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def make_grid(images: torch.Tensor, cols: int = 8, scale: int | None = None) -> Image.Image:
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--palette", default="assets/palettes/dawnbringer32.hex")
    p.add_argument("--steps",  type=int,   default=60000)
    p.add_argument("--lr",     type=float, default=3e-4)
    p.add_argument("--out",    default="workdir/v6f_discrete")
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--init",   default=None, help="path to DiscretePaletteUNet state_dict")
    p.add_argument("--seed",   type=int,   default=0)
    p.add_argument("--ema",    type=float, default=0.999, help="EMA decay (0 = off)")
    p.add_argument("--ms_aug", action="store_true")
    p.add_argument("--ms_up",  action="store_true")
    p.add_argument("--extra",  default=None, help="img_dir,captions_csv,repeat")
    p.add_argument("--extra2", default=None, help="img_dir,captions_csv,repeat")
    p.add_argument("--embed_dim", type=int, default=16, help="palette token embedding dimension")
    p.add_argument("--T",      type=int, default=1000, help="number of discrete noise steps")
    args = p.parse_args()

    device = "cuda"
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    palette = load_hex_palette(args.palette).to(device)
    K = palette.shape[0]
    print(f"palette: {K} colours from {args.palette}", flush=True)

    sources = [("data/oga_clean", "data/oga_captions.csv", 1)]
    if args.extra:
        d, c, r = args.extra.split(","); sources.append((d, c, int(r)))
    if args.extra2:
        d, c, r = args.extra2.split(","); sources.append((d, c, int(r)))

    ds = DiscreteSpriteDataset(sources, palette.cpu(), ms_aug=args.ms_aug, ms_up=args.ms_up)
    counts = [ds.bucket_of.count(b) for b in range(len(BUCKETS))]
    print(f"dataset: {len(ds)} samples, buckets {dict(zip(BUCKETS, counts))}", flush=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=BucketSampler(ds.bucket_of, args.steps), num_workers=8)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    text_encoder.requires_grad_(False)

    model = DiscretePaletteUNet(K, embed_dim=args.embed_dim).to(device)
    print(f"params: {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M", flush=True)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"init from {args.init}", flush=True)

    schedule = AbsorbingSchedule(T=args.T, K=K)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ema = None
    if args.ema > 0:
        import copy
        ema = copy.deepcopy(model).eval().requires_grad_(False)

    eval_cond  = embed(EVAL_PROMPTS, tokenizer, text_encoder, device)
    eval_uncond = embed([""] * len(EVAL_PROMPTS), tokenizer, text_encoder, device)

    step = 0
    for idx_bhw, alpha0_bhw, texts, b_list in loader:
        # 10 % CFG dropout
        texts = ["" if random.random() < 0.1 else t for t in texts]
        cond = embed(list(texts), tokenizer, text_encoder, device)

        idx_bhw   = idx_bhw.to(device)
        alpha0_bhw = alpha0_bhw.to(device)
        b_tensor  = b_list.to(device)

        # Sample noise level t ~ Uniform[1, T)
        t_b = torch.randint(1, args.T, (idx_bhw.shape[0],), device=device)
        xt  = schedule.q_sample(idx_bhw, t_b)

        pal_logits, alp_logits = model(xt, t_b, cond, b_tensor)
        loss = schedule.loss(pal_logits, alp_logits, idx_bhw, alpha0_bhw, xt)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if ema is not None:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1.0 - args.ema)

        if step % 200 == 0:
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} bucket={BUCKETS[int(b_tensor[0])]}",
                  flush=True)

        if step % args.sample_every == 0 or step == args.steps:
            net = ema if ema is not None else model
            net.eval()
            for s in EVAL_SIZES:
                imgs = discrete_sample(net, schedule, eval_cond, eval_uncond, s,
                                       device=device, steps=64, cfg=3.0,
                                       seed=args.seed, palette=palette)
                make_grid(imgs).save(out / "samples" / f"step_{step:06d}_s{s}.png")
            torch.save(net.state_dict(), out / "model_latest.pt")
            model.train()
    print(f"Done -> {out}", flush=True)


if __name__ == "__main__":
    main()
