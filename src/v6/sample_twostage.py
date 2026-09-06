"""Two-stage sampling: structure generator (train_sgen.py) -> colouriser (train_cond.py --cond struct).

Text prompt -> stage 1 samples S-as-RGBA -> quantise to S=[alpha, 4-level Lq] ->
stage 2 samples the RGBA sprite conditioned on (text, S).  Output layout matches
sample_e.py / eval_probe.sh:  <out>/s<size>/NNNNN.png  + grids.

Usage:
  python src/v6/sample_twostage.py --sgen workdir/probe_sgen/model_latest.pt \
      --color workdir/probe_struct/model_latest.pt --prompts runs_out/derisk_prompts.txt \
      --n 8 --size 16 --out runs_out/probe_sgen_eval
"""
import argparse
import sys
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from sample_cond import to_rgba  # noqa: E402
from train_cond import build_unet, embed, make_grid, sample  # noqa: E402
from train_sgen import rgba_to_struct, struct_to_rgba  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgen", required=True)
    ap.add_argument("--color", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--cfg2", type=float, default=4.0, help="text CFG of the colouriser")
    ap.add_argument("--bs", type=int, default=64, help="prompts per batch (x n samples each)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"
    out = Path(args.out)
    (out / f"s{args.size}").mkdir(parents=True, exist_ok=True)
    (out / f"struct{args.size}").mkdir(parents=True, exist_ok=True)

    sg = build_unet(4).to(device).eval()
    sg.load_state_dict(torch.load(args.sgen, map_location=device)["unet"])
    ck = torch.load(args.color, map_location=device)
    assert ck["cond"] == "struct", ck["cond"]
    col = build_unet(6).to(device).eval()
    col.load_state_dict(ck["unet"])
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip() and not l.startswith("#")]
    print(f"two-stage: {len(prompts)} prompts x {args.n} @ {args.size}px", flush=True)
    grid_s, grid_x, k = [], [], 0
    for i in range(0, len(prompts), args.bs):
        chunk = [t for t in prompts[i:i + args.bs] for _ in range(args.n)]
        cond = embed(chunk, tokenizer, enc, device)
        uncond = embed([""] * len(chunk), tokenizer, enc, device)
        torch.manual_seed(args.seed + i)
        s_rgba = sample(sg, scheduler, cond, uncond, args.size, steps=args.steps, cfg=args.cfg)
        S = rgba_to_struct(s_rgba.to(device))
        torch.manual_seed(args.seed + i + 1)
        gen = sample(col, scheduler, cond, uncond, args.size, S=S, steps=args.steps, cfg=args.cfg2)
        s_q = ((struct_to_rgba(S) + 1) / 2).clamp(0, 1).cpu()
        for j in range(gen.shape[0]):
            to_rgba(gen[j]).save(out / f"s{args.size}" / f"{k:05d}.png")
            to_rgba(s_q[j]).save(out / f"struct{args.size}" / f"{k:05d}.png")
            k += 1
        if len(grid_s) < 4:
            grid_s.append(s_q[:: args.n][:8])
            grid_x.append(gen[:: args.n][:8])
        print(f"  {k} samples", flush=True)
    # grid: row pairs (structure, final) for the first 32 prompts
    rows = []
    for a, b in zip(grid_s, grid_x):
        rows += [a, b]
    make_grid(torch.cat(rows, 0), cols=8).save(out / f"grid_s{args.size}.png")
    print(f"size {args.size}: {k} samples -> {out}", flush=True)


if __name__ == "__main__":
    main()
