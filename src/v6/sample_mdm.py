"""Sample from the A1 masked discrete diffusion model (train_mdm.py).

Decoding algorithm: confidence-ranked parallel unmasking (MaskGIT / Ghazvininejad et al.):
  1. Start with all tokens = MASK.
  2. For each step i in [0, T):
     - Forward pass -> logits for all positions.
     - For MASKED positions: sample or argmax, compute confidence = max softmax prob.
     - Schedule says how many tokens should still be masked after this step: n_mask(i).
     - Among all currently-masked positions, pick the (L - n_mask(i)) MOST confident to unmask.
     - Lock them in; the rest stay MASK for the next round.
  3. After T steps all tokens are unmasked -> reshape to RGBA image.

The masking schedule is cosine: n_mask(i) = ceil(L * cos(pi/2 * (i+1)/T)), following MaskGIT.

Text conditioning: CLIP cross-attention (same as training).  CFG with w>1 is supported by
running two forwards (cond + uncond), combining logits as log p_cfg = (1+w) log p_c - w log p_u
(the discrete analogue of continuous CFG).

Usage:
  python src/v6/sample_mdm.py --ckpt workdir/probe_mdm/model_latest.pt \
      --prompts runs_out/heldout3000_prompts.txt --sizes 16 --n 1 --out runs_out/probe_mdm_eval
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

import sys
sys.path.insert(0, "src/v6")
from train_mdm import MDM, V, MASK, CH, BUCKETS


@torch.no_grad()
def embed(texts, tokenizer, encoder, device):
    tok = tokenizer(texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
    return encoder(**tok).last_hidden_state


def cosine_schedule(i, T, L):
    """How many tokens should still be masked after step i (0-indexed)."""
    frac = math.cos(math.pi / 2 * (i + 1) / T)
    return max(0, math.ceil(L * frac))


def ids_to_rgba(ids, size):
    """(H*W*4,) long ids -> (4, H, W) float tensor in [0,1]."""
    return ids.reshape(size, size, CH).permute(2, 0, 1).float() / 255.0


def to_rgba_pil(img):
    """(4,h,w) float [0,1] -> PIL RGBA with hard alpha."""
    a = (img[3] > 0.5).float()
    rgb = img[:3] * a
    arr = torch.cat([rgb, a[None]], 0)
    return Image.fromarray((arr.permute(1, 2, 0) * 255).byte().numpy(), "RGBA")


@torch.no_grad()
def sample_mdm(model, cond, uncond, size, device, steps=64, cfg=2.0, seed=0,
               temperature=1.0, use_gumbel=False):
    """Generate one batch of images via confidence-ranked unmasking."""
    B = cond.shape[0]
    L = size * size * CH  # total tokens per image
    lab_idx = BUCKETS.index(size)
    lab = torch.full((B,), lab_idx, device=device, dtype=torch.long)

    pix_idx = torch.arange(L, device=device) // CH
    ch_idx = torch.arange(L, device=device) % CH

    g = torch.Generator(device=device).manual_seed(seed)

    # start all masked
    ids = torch.full((B, L), MASK, device=device, dtype=torch.long)

    for i in range(steps):
        # how many should remain masked after this step
        n_remain = cosine_schedule(i, steps, L)

        # current mask
        is_masked = (ids == MASK)  # (B, L)

        # noise level for the model: fraction of tokens still masked
        t_val = is_masked.float().mean(dim=1)  # (B,)

        # forward pass (conditional)
        logits_c = model(ids, pix_idx, ch_idx, t_val, cond, lab)  # (B, L, V)

        if cfg > 1.0 and uncond is not None:
            logits_u = model(ids, pix_idx, ch_idx, t_val, uncond, lab)
            # discrete CFG in log-probability space
            log_p = (1 + cfg) * F.log_softmax(logits_c, -1) - cfg * F.log_softmax(logits_u, -1)
        else:
            log_p = F.log_softmax(logits_c, -1)

        # sample or argmax from the distribution
        probs = F.softmax(log_p / temperature, -1)  # (B, L, V)
        if use_gumbel:
            # Gumbel sampling for diversity
            noise = torch.empty_like(probs).exponential_(generator=g).clamp_min(1e-10)
            predicted = (probs / noise).argmax(-1)  # Gumbel-max trick
        else:
            predicted = probs.argmax(-1)  # (B, L)

        # confidence = probability of the chosen token
        confidence = probs.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)  # (B, L)

        # only consider currently masked positions
        confidence = confidence.masked_fill(~is_masked, float('inf'))

        if n_remain == 0:
            # unmask everything
            ids = torch.where(is_masked, predicted, ids)
        else:
            # find the n_remain LEAST confident masked positions -> keep them masked
            # equivalently: unmask everything except the n_remain lowest-confidence ones
            n_to_unmask = is_masked.long().sum(dim=1) - n_remain  # per sample

            for b in range(B):
                masked_pos = is_masked[b].nonzero(as_tuple=True)[0]
                if masked_pos.numel() <= n_remain:
                    # fewer masked than target: unmask all
                    ids[b] = torch.where(is_masked[b], predicted[b], ids[b])
                    continue
                conf_at_masked = confidence[b, masked_pos]
                # top-k most confident among masked -> unmask those
                k = min(max(int(n_to_unmask[b].item()), 1), masked_pos.numel())
                _, topk_idx = conf_at_masked.topk(k, largest=True)
                unmask_pos = masked_pos[topk_idx]
                ids[b, unmask_pos] = predicted[b, unmask_pos]

    # convert to images
    imgs = torch.stack([ids_to_rgba(ids[b], size) for b in range(B)])
    return imgs.cpu()


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
    p.add_argument("--n", type=int, default=1, help="samples per prompt")
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=64, help="number of unmasking steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gumbel", action="store_true", help="use Gumbel sampling instead of argmax")
    p.add_argument("--chunk", type=int, default=200, help="max samples per forward batch")
    p.add_argument("--out", required=True)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--max_side", type=int, default=32)
    args = p.parse_args()

    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip() and not l.startswith("#")]

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    model = MDM(args.dim, args.depth, max_pix=args.max_side ** 2).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f"loaded MDM from {args.ckpt}, params {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M", flush=True)

    all_prompts = [pr for pr in prompts for _ in range(args.n)]
    cond_all = embed(all_prompts, tokenizer, enc, device)
    uncond_all = embed([""] * len(all_prompts), tokenizer, enc, device)

    for size in args.sizes:
        if size > args.max_side:
            print(f"skipping size {size} (> max_side {args.max_side})", flush=True)
            continue
        (out / f"s{size}").mkdir(exist_ok=True)
        imgs_all = []
        for start in range(0, len(all_prompts), args.chunk):
            end = min(start + args.chunk, len(all_prompts))
            imgs = sample_mdm(model, cond_all[start:end], uncond_all[start:end],
                              size, device, steps=args.steps, cfg=args.cfg,
                              seed=args.seed + start // args.chunk,
                              temperature=args.temperature, use_gumbel=args.gumbel)
            imgs_all.append(imgs)
        imgs_all = torch.cat(imgs_all, 0)
        rgba = [to_rgba_pil(im) for im in imgs_all]
        for i, im in enumerate(rgba):
            pi, k = divmod(i, args.n)
            im.save(out / f"s{size}" / f"{pi:02d}_{k}.png")
        if len(prompts) <= 50:
            grid(rgba, len(prompts), args.n, max(1, 128 // size)).save(out / f"grid_s{size}.png")
        print(f"size {size}: {len(rgba)} samples -> {out}/s{size}/", flush=True)
    (out / "prompts.txt").write_text("\n".join(prompts), encoding="utf-8")
    print("SAMPLE_MDM_DONE", flush=True)


if __name__ == "__main__":
    main()
