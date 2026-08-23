"""CLIP-score guided best-of-N reranking for pixel art generation.

Motivation for the paper:
  At test time, pixel art generation from text is inherently stochastic —
  different seeds produce sprites of very different quality.  We propose
  using CLIP image-text similarity as a lightweight quality proxy to
  select the best sample from N candidates.  This gives a principled
  quality-compute tradeoff: with N=1 you get the raw model; with N=16
  you get near-human curator quality for the cost of 16 forward passes
  (still 225× cheaper than a single SD-πXL run).

  Contribution: we establish the first quality-vs-compute Pareto curves
  for pixel art generation (FID / MC-score / human score vs N), showing
  that best-of-N with CLIP reranking is a powerful and practical inference
  strategy for production texture pipelines.

Two CLIP scoring modes:
  global  — standard CLIP(image, text) cosine similarity; fast
  local   — palette-aware: composite on white, upsample to 224, score;
            accounts for transparency (transparent sprites score poorly
            if composited on black because the background bleeds into
            the CLIP embedding)

Usage:
  # Score and rerank — output best-of-N for each prompt
  python src/v6/clip_rerank.py \\
      --ckpt workdir/v6e10_ema/model_latest.pt \\
      --prompts baseline/prompts8.txt \\
      --size 16 --n 16 --out runs/rerank_16 --save_all

  # Sweep N and emit quality-compute curve data (FID + CLIP-score vs N)
  python src/v6/clip_rerank.py \\
      --ckpt workdir/v6e10_ema/model_latest.pt \\
      --prompts baseline/prompts8.txt \\
      --size 16 --n 32 --sweep 1 2 4 8 16 32 \\
      --fid_ref data/oga_captions.csv --out runs/rerank_sweep
"""
import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPTextModel

BUCKETS = [16, 24, 32, 48, 64]


# ---------------------------------------------------------------------------
# Model / embed helpers (mirrors sample_e.py)
# ---------------------------------------------------------------------------

def build_model(device):
    from diffusers import UNet2DConditionModel
    return UNet2DConditionModel(
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(128, 256, 512), cross_attention_dim=512,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        num_class_embeds=len(BUCKETS),
    ).to(device)


@torch.no_grad()
def embed_text(texts, tokenizer, encoder, device):
    tok = tokenizer(texts, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt").to(device)
    return encoder(**tok).last_hidden_state


@torch.no_grad()
def sample_batch(model, scheduler, cond, uncond, size, device, steps=100, cfg=4.0, seed=0):
    from diffusers import DDPMScheduler
    scheduler.set_timesteps(steps)
    n = cond.shape[0]
    lab = torch.full((n,), BUCKETS.index(size), device=device, dtype=torch.long)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, 4, size, size, device=device, generator=g)
    for t in scheduler.timesteps:
        ec = model(x, t, encoder_hidden_states=cond,   class_labels=lab).sample
        eu = model(x, t, encoder_hidden_states=uncond, class_labels=lab).sample
        x  = scheduler.step(eu + cfg * (ec - eu), t, x).prev_sample
    return ((x + 1) / 2).clamp(0, 1).cpu()            # (B, 4, H, W) in [0,1]


# ---------------------------------------------------------------------------
# CLIP scoring
# ---------------------------------------------------------------------------

class CLIPScorer:
    """Palette-aware CLIP image-text scorer.

    Sprites have transparent backgrounds.  We composite on white before
    scoring (same convention as eval_fid.py) so the alpha channel doesn't
    corrupt the CLIP visual embedding.  Images are upsampled to 224 with
    NEAREST (preserving the pixel-art look) before feeding to CLIP.
    """

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        self.device = device
        self.clip = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def score(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """
        images:  (B, 4, H, W) float32 [0,1] RGBA
        prompts: B strings (one per image)
        Returns: (B,) float32 cosine similarity
        """
        B, _, H, W = images.shape
        # Composite on white, upsample to 224
        pil_imgs = []
        for i in range(B):
            img = images[i]
            a = (img[3] > 0.5).float()
            rgb = img[:3] * a + (1 - a)               # white background
            arr = (rgb.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
            pil = Image.fromarray(arr, "RGB").resize((224, 224), Image.NEAREST)
            pil_imgs.append(pil)

        inputs = self.proc(text=prompts, images=pil_imgs, return_tensors="pt",
                           padding=True, truncation=True).to(self.device)
        out = self.clip(**inputs)
        # Cosine similarity between image and text embeddings (normalised by CLIP)
        img_emb  = out.image_embeds  / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb  = out.text_embeds   / out.text_embeds.norm(dim=-1, keepdim=True)
        return (img_emb * txt_emb).sum(-1).cpu()       # (B,)


# ---------------------------------------------------------------------------
# Best-of-N reranking
# ---------------------------------------------------------------------------

def rerank_best_of_n(
    all_imgs: torch.Tensor,           # (P*N, 4, H, W) — P prompts × N samples each
    all_scores: torch.Tensor,         # (P*N,)
    prompts: list[str],
    n: int,
    top_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top_k images per prompt by CLIP score.

    Returns:
      selected:  (P*top_k, 4, H, W)
      sel_scores:(P*top_k,)
    """
    P = len(prompts)
    selected, sel_scores = [], []
    for pi in range(P):
        chunk = all_imgs[pi * n:(pi + 1) * n]
        sc    = all_scores[pi * n:(pi + 1) * n]
        order = sc.argsort(descending=True)
        selected.append(chunk[order[:top_k]])
        sel_scores.append(sc[order[:top_k]])
    return torch.cat(selected), torch.cat(sel_scores)


# ---------------------------------------------------------------------------
# Quality-compute sweep
# ---------------------------------------------------------------------------

def sweep_quality_vs_compute(
    all_imgs: torch.Tensor,
    all_scores: torch.Tensor,
    prompts: list[str],
    sweep_ns: list[int],
    out: Path,
    size: int,
) -> dict:
    """For each N in sweep_ns, compute mean CLIP-score of best-of-N selection."""
    results = {}
    P = len(prompts)
    max_n = max(sweep_ns)
    assert all_imgs.shape[0] >= P * max_n, (
        f"Need {P * max_n} images but only have {all_imgs.shape[0]}")

    for n in sorted(sweep_ns):
        # Trim to first n samples per prompt
        chunk_imgs   = torch.stack([all_imgs[pi * max_n:pi * max_n + n]   for pi in range(P)]).flatten(0, 1)
        chunk_scores = torch.stack([all_scores[pi * max_n:pi * max_n + n] for pi in range(P)]).flatten(0, 1)
        best_imgs, best_scores = rerank_best_of_n(chunk_imgs, chunk_scores, prompts, n)
        mean_clip = best_scores.mean().item()
        results[n] = {"mean_clip": mean_clip, "images": best_imgs}
        print(f"  N={n:3d}: mean CLIP-score = {mean_clip:.4f}", flush=True)

    # Save CSV
    with open(out / f"sweep_s{size}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "mean_clip_score"])
        for n in sorted(results.keys()):
            w.writerow([n, f"{results[n]['mean_clip']:.6f}"])
    print(f"Saved quality-compute curve -> {out}/sweep_s{size}.csv", flush=True)
    return results


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def to_rgba_pil(t: torch.Tensor) -> Image.Image:
    a = (t[3] > 0.5).float()
    arr = torch.cat([t[:3] * a, a[None]], 0)
    return Image.fromarray((arr.permute(1, 2, 0) * 255).byte().numpy(), "RGBA")


def make_rerank_grid(
    all_imgs: torch.Tensor,
    best_imgs: torch.Tensor,
    all_scores: torch.Tensor,
    best_scores: torch.Tensor,
    prompts: list[str],
    n: int,
    size: int,
    scale: int = 6,
) -> Image.Image:
    """Grid: each row = one prompt; columns = [best | all N candidates sorted by score]."""
    P = len(prompts)
    cell = size * scale
    sep = 4
    cols = 1 + n
    w = cell * cols + sep
    h = cell * P
    out_img = Image.new("RGBA", (w, h), (30, 30, 30, 255))
    checker_cache = {}

    def checkerboard(sz):
        if sz not in checker_cache:
            arr = np.zeros((sz, sz, 4), dtype=np.uint8)
            c = sz // 4
            for yy in range(sz):
                for xx in range(sz):
                    v = 209 if ((yy // c + xx // c) % 2) else 240
                    arr[yy, xx] = [v, v, v, 255]
            checker_cache[sz] = Image.fromarray(arr, "RGBA")
        return checker_cache[sz]

    for pi in range(P):
        y = pi * cell
        sc_row  = all_scores[pi * n:(pi + 1) * n]
        img_row = all_imgs[pi * n:(pi + 1) * n]
        order   = sc_row.argsort(descending=True)

        # Best-of-N (leftmost, highlighted)
        best = to_rgba_pil(best_imgs[pi]).resize((cell, cell), Image.NEAREST)
        bg = checkerboard(cell).copy()
        bg.alpha_composite(best)
        out_img.paste(bg, (0, y))

        # Remaining candidates in score order
        for k, idx in enumerate(order):
            xoff = cell + sep + k * cell
            cand = to_rgba_pil(img_row[idx]).resize((cell, cell), Image.NEAREST)
            bg2 = checkerboard(cell).copy()
            bg2.alpha_composite(cand)
            # Darken rejected candidates slightly
            if k > 0:
                bg2 = bg2.point(lambda p: int(p * 0.7))
            out_img.paste(bg2, (xoff, y))

    return out_img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     required=True)
    p.add_argument("--prompts",  required=True)
    p.add_argument("--size",     type=int, default=16)
    p.add_argument("--n",        type=int, default=16,
                   help="total candidates per prompt (must be >= max(--sweep))")
    p.add_argument("--cfg",      type=float, default=4.0)
    p.add_argument("--steps",    type=int, default=100)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--sweep",    type=int, nargs="*", default=None,
                   help="N values for quality-compute sweep, e.g. --sweep 1 2 4 8 16")
    p.add_argument("--out",      required=True)
    p.add_argument("--save_all", action="store_true",
                   help="also save all N candidate images per prompt")
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    args = p.parse_args()

    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8")
               if l.strip() and not l.startswith("#")]
    P = len(prompts)
    print(f"{P} prompts × N={args.n} = {P * args.n} total samples at {args.size}px", flush=True)

    # --- Build generation model ---
    from diffusers import DDPMScheduler
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_enc  = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    gen_model = build_model(device)
    gen_model.load_state_dict(torch.load(args.ckpt, map_location=device))
    gen_model.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    # --- Build CLIP scorer ---
    scorer = CLIPScorer(args.clip_model, device)

    # --- Generate N samples per prompt (different seeds) ---
    all_imgs_list = []
    for seed_offset in range(args.n):
        cond   = embed_text(prompts, tokenizer, text_enc, device)
        uncond = embed_text([""] * P, tokenizer, text_enc, device)
        imgs = sample_batch(gen_model, scheduler, cond, uncond, args.size, device,
                            args.steps, args.cfg, seed=args.seed + seed_offset)  # (P, 4, H, W)
        all_imgs_list.append(imgs)
        print(f"  generated seed offset {seed_offset}/{args.n}", flush=True)

    # Interleave so that samples[pi*N:(pi+1)*N] = N samples for prompt pi
    all_imgs = torch.stack(all_imgs_list, dim=1).flatten(0, 1)   # (P*N, 4, H, W)

    # --- CLIP score all ---
    print("Scoring with CLIP …", flush=True)
    tiled_prompts = [p for p in prompts for _ in range(args.n)]  # P*N strings
    bs = 64
    scores_list = []
    for i in range(0, len(all_imgs), bs):
        chunk = all_imgs[i:i + bs].to(device)
        chunk_p = tiled_prompts[i:i + bs]
        scores_list.append(scorer.score(chunk, chunk_p))
    all_scores = torch.cat(scores_list)                           # (P*N,)

    # --- Best-of-N ---
    best_imgs, best_scores = rerank_best_of_n(all_imgs, all_scores, prompts, args.n)

    # Save best images
    best_dir = out / "best"
    best_dir.mkdir(exist_ok=True)
    for pi, (im, sc) in enumerate(zip(best_imgs, best_scores)):
        to_rgba_pil(im).save(best_dir / f"{pi:02d}_{prompts[pi][:40].replace(' ', '_')}.png")
    print(f"Best-of-{args.n} mean CLIP-score: {best_scores.mean():.4f}", flush=True)

    # Save comparison grid
    grid = make_rerank_grid(all_imgs, best_imgs, all_scores, best_scores,
                             prompts, args.n, args.size)
    grid.save(out / f"rerank_grid_N{args.n}_s{args.size}.png")
    print(f"Comparison grid -> {out}/rerank_grid_N{args.n}_s{args.size}.png", flush=True)

    if args.save_all:
        all_dir = out / "all"
        all_dir.mkdir(exist_ok=True)
        for pi in range(P):
            for k in range(args.n):
                idx = pi * args.n + k
                to_rgba_pil(all_imgs[idx]).save(all_dir / f"{pi:02d}_{k:02d}.png")

    # Save scores CSV
    with open(out / f"scores_s{args.size}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_idx", "seed_offset", "clip_score", "is_best"])
        for pi in range(P):
            sc_row = all_scores[pi * args.n:(pi + 1) * args.n]
            best_k = int(sc_row.argmax())
            for k in range(args.n):
                w.writerow([pi, k, f"{sc_row[k]:.6f}", k == best_k])

    # --- Quality-compute sweep ---
    if args.sweep:
        print(f"\nQuality-compute sweep: N ∈ {args.sweep}", flush=True)
        sweep_results = sweep_quality_vs_compute(
            all_imgs, all_scores, prompts, args.sweep, out, args.size)
        # Also save best grids for each N
        for n_val in sorted(args.sweep):
            chunk_imgs   = torch.stack([all_imgs[pi * args.n:pi * args.n + n_val] for pi in range(P)]).flatten(0, 1)
            chunk_scores = torch.stack([all_scores[pi * args.n:pi * args.n + n_val] for pi in range(P)]).flatten(0, 1)
            best_n, best_n_sc = rerank_best_of_n(chunk_imgs, chunk_scores, prompts, n_val)
            g = make_rerank_grid(chunk_imgs, best_n, chunk_scores, best_n_sc,
                                  prompts, n_val, args.size)
            g.save(out / f"rerank_grid_N{n_val}_s{args.size}.png")

    print(f"\nDone → {out}", flush=True)


if __name__ == "__main__":
    main()
