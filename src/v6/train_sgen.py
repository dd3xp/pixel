"""Stage-1 STRUCTURE generator: fine-tune v7 to generate S-as-RGBA instead of the sprite.

S-as-RGBA = [Lq, Lq, Lq, alpha]  where Lq is the 4-level OKLab-lightness value sketch
(train_cond.make_struct), i.e. a greyscale 4-tone sprite; transparent pixels are
(-1,-1,-1,-1) exactly like v7's data convention.  Same UNet, same text/bucket
conditioning, same epsilon objective -- only the target changes, so any FD gain of the
two-stage pipeline (sgen -> probe_struct colouriser) over v7 is attributable to the
structure-first decomposition and not to capacity.

Usage:
  python src/v6/train_sgen.py --init workdir/v7_lowres/model_latest.pt --steps 20000 --out workdir/probe_sgen
"""
import argparse
import copy
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from train_cond import (BUCKETS, EVAL_SIZES, N_LEVELS, BucketSampler, NativeSprites, build_unet,  # noqa: E402
                        embed, load_v7_into, make_grid, make_struct, sample, to_tensor)


@torch.no_grad()
def struct_to_rgba(S):
    """S (B,2,H,W) [alpha, Lq] -> S-as-RGBA (B,4,H,W) in [-1,1]."""
    alpha, Lq = S[:, :1], S[:, 1:2]
    op = alpha > 0
    grey = torch.where(op, Lq, torch.full_like(Lq, -1.0))
    return torch.cat([grey, grey, grey, torch.where(op, torch.ones_like(alpha), -torch.ones_like(alpha))], 1)


@torch.no_grad()
def rgba_to_struct(x):
    """Generated S-as-RGBA (B,4,H,W) in [0,1] -> S (B,2,H,W) [alpha(+-1), Lq in {-1,-1/3,1/3,1}, transparent=0]."""
    alpha = (x[:, 3:4] > 0.5).float() * 2 - 1
    grey = x[:, :3].mean(1, keepdim=True)  # [0,1]
    q = (grey * N_LEVELS).floor().clamp(0, N_LEVELS - 1) / (N_LEVELS - 1) * 2 - 1
    return torch.cat([alpha, torch.where(alpha > 0, q, torch.zeros_like(q))], 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="v7 checkpoint")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", required=True)
    p.add_argument("--sample_every", type=int, default=4000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda"
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    sources = [
        ("data/oga_clean", "data/oga_captions.csv", 1),
        ("data/extra_all", "data/extra_all.csv", 1),
        ("data/oga_clean", "data/tool_candidates.csv", 2),
    ]
    ds = NativeSprites(sources)
    print(f"dataset: {len(ds)}", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=BucketSampler(ds.bucket_of, args.steps), num_workers=8)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    text_encoder.requires_grad_(False)

    model = build_unet(4).to(device)
    load_v7_into(model, args.init, 4, device)
    print(f"unet params: {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M, init from {args.init}", flush=True)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if args.ema > 0 else None

    ref_rows = [ds.rows[i] for i in random.sample(range(len(ds)), 8)]
    ref_cond = embed([t for _, t in ref_rows], tokenizer, text_encoder, device)
    ref_uncond = embed([""] * 8, tokenizer, text_encoder, device)

    step = 0
    for x, texts, b in loader:
        texts = ["" if random.random() < 0.1 else t for t in texts]
        cond = embed(list(texts), tokenizer, text_encoder, device)
        x, b = x.to(device), b.to(device)
        x0 = struct_to_rgba(make_struct(x))  # the only change vs v7: target is the structure sprite
        noise = torch.randn_like(x0)
        t = torch.randint(0, 1000, (x0.shape[0],), device=device)
        xt = scheduler.add_noise(x0, noise, t)
        pred = model(xt, t, encoder_hidden_states=cond, class_labels=b).sample
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if ema is not None:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1.0 - args.ema)
        if step % 200 == 0:
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} bucket={BUCKETS[int(b[0])]}", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            net = ema if ema is not None else model
            net.eval()
            for s in EVAL_SIZES:
                xr = torch.stack([to_tensor(Image.open(pth).convert("RGBA"), s) for pth, _ in ref_rows]).to(device)
                real_s = ((struct_to_rgba(make_struct(xr)) + 1) / 2).clamp(0, 1).cpu()
                torch.manual_seed(args.seed)
                gen = sample(net, scheduler, ref_cond, ref_uncond, s)
                gen_q = ((struct_to_rgba(rgba_to_struct(gen.to(device))) + 1) / 2).clamp(0, 1).cpu()
                # row 1 = real structure of the ref prompt, row 2 = generated (raw), row 3 = generated quantised
                make_grid(torch.cat([real_s, gen, gen_q], 0)).save(out / "samples" / f"step_{step:06d}_s{s}.png")
            torch.save({"cond": "sgen", "unet": net.state_dict()}, out / "model_latest.pt")
            model.train()
    print(f"Done -> {out}", flush=True)


if __name__ == "__main__":
    main()
