"""Confidence-ordered decoder for the masked-discrete-diffusion probe (A1).

Training masks tokens independently with probability t and predicts them; generation runs that backwards.
Start from an all-masked canvas and, over T rounds, keep the predictions the model is most confident about and
re-mask the rest -- MaskGIT's parallel decoding (Chang et al., CVPR 2022) with a cosine schedule.  Unlike a DDPM
reverse step, every kept token is a COMMITTED value that later rounds condition on, which is the property this
architecture exists to test.

`--edit_rounds` adds Nemotron-style token editing: after the canvas is full, re-mask the least confident tokens
and re-predict, so an early bad commitment is not permanent.

Output layout matches the rest of the pipeline: runs_out/<name>/s<R>/<idx:02d>_0.png, scored by fd_fair.py.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, "src/v6")
from train_mdm import CH, MASK, V, MDM
from train_v7 import BUCKETS, embed


def to_rgba(ids, R):
    """(H*W*C,) ids -> PIL RGBA at R x R, hard alpha like the rest of the protocol."""
    a = ids.reshape(R, R, CH).clamp(0, 255).to(torch.uint8).cpu().numpy()
    a = a.copy()
    op = a[..., 3] >= 128
    a[~op] = 0
    a[op, 3] = 255
    return Image.fromarray(a, "RGBA")


@torch.no_grad()
def decode(model, cond, R, lab, steps, device, temp=1.0, edit_rounds=0, edit_frac=0.15, seed=0, greedy=False,
           ctx_ids=None, ctx_mask=None):
    B = cond.shape[0]
    L = R * R * CH
    pix_idx = torch.arange(L, device=device) // CH
    ch_idx = torch.arange(L, device=device) % CH
    g = torch.Generator(device=device).manual_seed(seed)

    ids = torch.full((B, L), MASK, device=device, dtype=torch.long)
    unknown = torch.ones(B, L, dtype=torch.bool, device=device)
    conf = torch.zeros(B, L, device=device)
    if ctx_ids is not None:  # oracle context: these positions start known and are never re-opened
        ids = torch.where(ctx_mask, ctx_ids, ids)
        unknown = unknown & ~ctx_mask
        conf = torch.where(ctx_mask, torch.ones_like(conf), conf)

    for s in range(steps):
        frac = (s + 1) / steps
        t = torch.full((B,), max(1e-3, 1.0 - s / steps), device=device)
        logits = model(ids, pix_idx, ch_idx, t, cond, lab) / max(temp, 1e-4)
        p = F.softmax(logits, -1)
        if greedy:
            pk, samp = p.max(-1)
        else:
            samp = torch.multinomial(p.reshape(-1, V), 1, generator=g).reshape(B, L)
            pk = p.gather(-1, samp[..., None]).squeeze(-1)

        # Cosine schedule on the number of tokens that should still be MASKED after this round, following
        # MaskGIT.  The ranking must run over the unknown tokens only, so score known ones -inf and take the
        # top `reveal` of what is left -- ranking over all L (with known ones at +inf) reveals almost nothing
        # per round and forces the whole canvas to be guessed at once in the final round.
        n_unknown = int(unknown[0].sum())
        target_masked = int(L * math.cos(math.pi * frac / 2))          # -> 0 at frac = 1
        reveal = max(1, n_unknown - target_masked) if s < steps - 1 else n_unknown
        score = torch.where(unknown, pk, torch.full_like(pk, float("-inf")))
        order = score.argsort(dim=-1, descending=True)
        rank = order.argsort(dim=-1)
        newly = unknown & (rank < reveal)

        ids = torch.where(newly, samp, ids)
        conf = torch.where(newly, pk, conf)
        unknown = unknown & ~newly
        if not unknown.any():
            break

    for _ in range(edit_rounds):  # re-open the least confident commitments and try again
        k = max(1, int(L * edit_frac))
        worst = conf.argsort(dim=-1)[:, :k]
        m = torch.zeros_like(unknown)
        m.scatter_(1, worst, True)
        if ctx_mask is not None:
            m = m & ~ctx_mask
        ids = torch.where(m, torch.full_like(ids, MASK), ids)
        t = torch.full((B,), edit_frac, device=device)
        logits = model(ids, pix_idx, ch_idx, t, cond, lab) / max(temp, 1e-4)
        p = F.softmax(logits, -1)
        if greedy:
            pk, samp = p.max(-1)
        else:
            samp = torch.multinomial(p.reshape(-1, V), 1, generator=g).reshape(B, L)
            pk = p.gather(-1, samp[..., None]).squeeze(-1)
        ids = torch.where(m, samp, ids)
        conf = torch.where(m, pk, conf)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", default="runs_out/heldout3000_prompts.txt")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=32, help="decoding rounds")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--greedy", action="store_true",
                    help="take the argmax instead of sampling; the diagnostic showed 80% exact token accuracy at "
                         "90% masking, so sampling the 256-way tail is what darkens the canvas")
    ap.add_argument("--edit_rounds", type=int, default=2)
    ap.add_argument("--edit_frac", type=float, default=0.15)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--max_side", type=int, default=32)
    ap.add_argument("--context_alpha", default=None,
                    help="ORACLE DIAGNOSTIC: directory of real sprites whose ALPHA channel is given to the model as "
                         "context, so it only has to decode RGB. Tests whether probe_mdm's collapse is a missing "
                         "anchor (it is a good inpainter: 80%% exact tokens at 90%% masking) rather than the "
                         "representation. Not a generator -- it consumes ground truth.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    R = a.size

    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    out = Path(a.out) / f"s{R}"
    out.mkdir(parents=True, exist_ok=True)
    if len(list(out.glob("*.png"))) >= len(prompts):
        print(f"already have {len(prompts)} samples in {out}"); return

    tokz = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    txt = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval().requires_grad_(False)
    model = MDM(a.dim, a.depth, max_pix=a.max_side ** 2).to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    lab_i = BUCKETS.index(R)

    ctx_files = sorted(Path(a.context_alpha).glob("*.png")) if a.context_alpha else None

    for i in range(0, len(prompts), a.chunk):
        chunk = prompts[i:i + a.chunk]
        cond = embed(list(chunk), tokz, txt, dev)
        lab = torch.full((len(chunk),), lab_i, device=dev, dtype=torch.long)
        ctx_ids = ctx_mask = None
        if ctx_files:
            arrs = []
            for k in range(len(chunk)):
                im = Image.open(ctx_files[(i + k) % len(ctx_files)]).convert("RGBA").resize((R, R), Image.NEAREST)
                arrs.append(np.asarray(im).reshape(-1))
            ctx_ids = torch.from_numpy(np.stack(arrs)).long().to(dev)
            ch = torch.arange(R * R * CH, device=dev) % CH
            ctx_mask = (ch == 3)[None].expand(len(chunk), -1).contiguous()   # alpha channel only
        ids = decode(model, cond, R, lab, a.steps, dev, a.temp, a.edit_rounds, a.edit_frac, a.seed + i, a.greedy,
                     ctx_ids, ctx_mask)
        for k in range(len(chunk)):
            to_rgba(ids[k], R).save(out / f"{i + k:02d}_0.png" if i + k < 100 else out / f"{i + k:05d}.png")
        print(f"[{min(i + a.chunk, len(prompts))}/{len(prompts)}]", flush=True)
    print("MDM_SAMPLE_DONE", out, flush=True)


if __name__ == "__main__":
    main()
