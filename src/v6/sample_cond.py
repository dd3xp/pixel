"""Oracle sampling for train_cond.py checkpoints.

Side-conditions (structure map or palette) are extracted from HELD-OUT real
sprites that are disjoint from the FD-DINOv2 reference set (the reference is
REAL[:3000] of the seed-0 shuffle used by eval_probe.sh / run_derisk_metric.sh;
sources are drawn from the remainder). Each source's own caption is the text.
Output layout matches sample_e.py so the eval_probe FD block works unchanged:
  <out>/s<size>/*.png  (raw RGBA)  +  <out>/grid_s<size>.png (ref row / gen row)

Usage:
  python src/v6/sample_cond.py --ckpt workdir/probe_struct/model_latest.pt \
      --n_src 413 --n 8 --size 16 --out runs_out/probe_struct_eval
"""
import argparse
import csv
import glob
import random
import sys
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from train_cond import PalTok, build_unet, embed, make_grid, make_palette, make_struct, sample, to_tensor  # noqa: E402


def to_rgba(img):
    a = (img[3] > 0.5).float()
    arr = torch.cat([img[:3] * a, a[None]], 0)
    return Image.fromarray((arr.permute(1, 2, 0) * 255).byte().numpy(), "RGBA")


def heldout_sources(n_src, seed=1):
    """Real sprites with captions, disjoint from the FD reference set."""
    real = glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")
    random.seed(0)
    random.shuffle(real)
    ref = {p.replace("\\", "/") for p in real[:3000]}
    cap = {}
    for f in ["data/oga_captions.csv", "data/tool_candidates.csv"]:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            cap["data/oga_clean/" + r["path"]] = r["text"]
    pool = sorted({p.replace("\\", "/") for p in real} - ref)
    pool = [p for p in pool if p in cap and cap[p].strip()]
    random.seed(seed)
    random.shuffle(pool)
    return [(p, cap[p]) for p in pool[:n_src]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_src", type=int, default=413)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--bs", type=int, default=64, help="sources per batch (x n samples each)")
    args = ap.parse_args()
    device = "cuda"
    out = Path(args.out)
    (out / f"s{args.size}").mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location=device)
    cond_type = ck["cond"]
    model = build_unet(6 if cond_type == "struct" else 4).to(device).eval()
    model.load_state_dict(ck["unet"])
    paltok = None
    if cond_type == "paltok":
        paltok = PalTok().to(device).eval()
        paltok.load_state_dict(ck["paltok"])
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    srcs = heldout_sources(args.n_src)
    print(f"cond={cond_type} sources={len(srcs)} x {args.n} samples @ {args.size}px", flush=True)
    grid_ref, grid_gen, k = [], [], 0
    for i in range(0, len(srcs), args.bs):
        chunk = srcs[i:i + args.bs]
        xr = torch.stack([to_tensor(Image.open(p).convert("RGBA"), args.size) for p, _ in chunk]).to(device)
        texts = [t for _, t in chunk]
        # repeat each source n times
        xr_n = xr.repeat_interleave(args.n, 0)
        cond = embed([t for t in texts for _ in range(args.n)], tokenizer, enc, device)
        uncond = embed([""] * xr_n.shape[0], tokenizer, enc, device)
        S = None
        with torch.no_grad():
            if cond_type == "struct":
                S = make_struct(xr_n)
            else:
                tok = paltok(make_palette(xr_n))
                cond, uncond = torch.cat([cond, tok], 1), torch.cat([uncond, tok], 1)
        torch.manual_seed(i)
        gen = sample(model, scheduler, cond, uncond, args.size, S=S, steps=args.steps, cfg=args.cfg)
        for j in range(gen.shape[0]):
            to_rgba(gen[j]).save(out / f"s{args.size}" / f"{k:05d}.png")
            k += 1
        if len(grid_ref) < 16:
            grid_ref.append(((xr + 1) / 2).clamp(0, 1).cpu()[:8])
            grid_gen.append(gen[:: args.n][:8])
        print(f"  {k} samples", flush=True)
    both = torch.cat([torch.cat(grid_ref, 0)[:16], torch.cat(grid_gen, 0)[:16]], 0)
    make_grid(both, cols=16).save(out / f"grid_s{args.size}.png")
    print(f"size {args.size}: {k} samples -> {out}", flush=True)


if __name__ == "__main__":
    main()
