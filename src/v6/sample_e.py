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
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel
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


def cads_anneal(cond, t, T, g, tau1=0.6, tau2=0.9, s=0.1, psi=1.0):
    """CADS (Sadat et al. 2024): corrupt the condition embedding with noise that anneals from
    s (high noise levels) to 0 (tau <= tau1), with mean/std rescaling mixed in by psi."""
    tau = float(t) / T
    gamma = 1.0 if tau <= tau1 else 0.0 if tau >= tau2 else (tau2 - tau) / (tau2 - tau1)
    if gamma >= 1.0:
        return cond
    noise = torch.randn(cond.shape, device=cond.device, generator=g)
    c_hat = (gamma ** 0.5) * cond + s * ((1 - gamma) ** 0.5) * noise
    if psi > 0:
        mu, sd = cond.mean(), cond.std()
        c_res = (c_hat - c_hat.mean()) / c_hat.std().clamp_min(1e-6) * sd + mu
        c_hat = psi * c_res + (1 - psi) * c_hat
    return c_hat


@torch.no_grad()
def sample(model, scheduler, cond, uncond, size, device, steps=100, cfg=4.0, seed=0, guide=None,
           cads=None, gi=(0.0, 1.0), guide_mode=None):
    scheduler.set_timesteps(steps)
    n = cond.shape[0]
    lab = torch.full((n,), BUCKETS.index(size), device=device, dtype=torch.long)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, 4, size, size, device=device, generator=g)
    last = scheduler.timesteps[-1]
    sc = None
    T = scheduler.config.num_train_timesteps
    cond0 = cond
    if getattr(model, "es", False):  # energy-score posterior-sample denoiser (train_es.py): few-step re-noising
        from train_es import sample_es
        return sample_es(model, cond, uncond, size, device, steps=steps, cfg=cfg, seed=seed)
    for t in scheduler.timesteps:
        if cads is not None:
            cond = cads_anneal(cond0, t, T, g, **cads)
        # guidance interval (Kynkaanniemi et al. 2024): CFG only for t/T in [gi_lo, gi_hi], else plain conditional
        w = cfg if gi[0] <= float(t) / T <= gi[1] else 1.0
        if getattr(model, "selfcond", False):
            # projection-in-the-loop self-conditioning (train_selfq.py): feed P(x0_hat) of previous step
            e_c = model(x, t, encoder_hidden_states=cond, class_labels=lab, sc=sc).sample
            e_u = model(x, t, encoder_hidden_states=uncond, class_labels=lab, sc=sc).sample
            e = e_u + w * (e_c - e_u)
            sc = model.project_from_eps(x, t, e)
            x = scheduler.step(e, t, x).prev_sample
            continue
        if t == last and getattr(model, "palhead", False) and model.hard_final:
            # palette-head model (train_palhead.py): final step = hard palette snap of the conditional x0
            x = model(x, t, encoder_hidden_states=cond, class_labels=lab, hard=True).x0.clamp(-1, 1)
            break
        e_c = model(x, t, encoder_hidden_states=cond, class_labels=lab).sample
        if w == 1.0:
            x = scheduler.step(e_c, t, x).prev_sample
            continue
        if guide is not None:  # autoguidance (Karras et al. 2024): weak model's conditional prediction as the reference
            e_u = guide(x, t, encoder_hidden_states=cond, class_labels=lab).sample
        elif guide_mode is not None:  # zero-training "bad versions" of the SAME model (pixel-art specific probes)
            kind, _, arg = guide_mode.partition(":")
            if kind == "bucket":  # wrong resolution embedding: model believes it is denoising a <arg>px sprite
                lab_bad = torch.full_like(lab, BUCKETS.index(int(arg)))
                e_u = model(x, t, encoder_hidden_states=cond, class_labels=lab_bad).sample
            elif kind == "blur":  # degraded input: prediction from a k x k box-blurred x_t (loses sub-block detail)
                k = int(arg or 2)
                xb = F.interpolate(F.avg_pool2d(x, k), size=(size, size), mode="nearest")
                e_u = model(xb, t, encoder_hidden_states=cond, class_labels=lab).sample
            elif kind == "shift":  # degraded input: 1-px cyclic shift (breaks pixel-grid alignment)
                xb = torch.roll(x, shifts=(1, 1), dims=(2, 3))
                e_u = torch.roll(model(xb, t, encoder_hidden_states=cond, class_labels=lab).sample, shifts=(-1, -1), dims=(2, 3))
            else:
                raise ValueError(guide_mode)
        else:
            e_u = model(x, t, encoder_hidden_states=uncond, class_labels=lab).sample
        x = scheduler.step(e_u + w * (e_c - e_u), t, x).prev_sample
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
    p.add_argument("--steps", type=int, default=None, help="DDPM steps (default 100; es models default to their few-step count)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--buckets", default=None, help="comma list, e.g. 12,16,20,24,32,48,64 for v7 models")
    p.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim"])
    p.add_argument("--guide_ckpt", default=None, help="plain 4ch ckpt of a WEAKER model -> autoguidance instead of CFG")
    p.add_argument("--guide_mode", default=None, help="zero-training bad model from the same net: bucket:<px> | blur:<k> | shift")
    p.add_argument("--cads", action="store_true", help="CADS condition annealing (tau1/tau2/s/psi below)")
    p.add_argument("--cads_tau1", type=float, default=0.6)
    p.add_argument("--cads_tau2", type=float, default=0.9)
    p.add_argument("--cads_s", type=float, default=0.1)
    p.add_argument("--cads_psi", type=float, default=1.0)
    p.add_argument("--gi", type=float, nargs=2, default=[0.0, 1.0],
                   help="guidance interval as t/T range [lo hi]; CFG off (w=1) outside it")
    p.add_argument("--chunk", type=int, default=1000, help="max samples per forward batch")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cads = dict(tau1=args.cads_tau1, tau2=args.cads_tau2, s=args.cads_s, psi=args.cads_psi) if args.cads else None
    if args.buckets:
        BUCKETS.clear()
        BUCKETS.extend(int(v) for v in args.buckets.split(","))
    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip() and not l.startswith("#")]
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    sd = torch.load(args.ckpt, map_location=device)
    if isinstance(sd, dict) and "palhead" in sd:  # palette-factorised head probe (train_palhead.py)
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from train_palhead import build_palhead
        model = build_palhead(sd["palhead"], device)
        model.load_state_dict(sd["state"])
        print(f"palhead model K={sd['palhead']['K']} {sd['palhead']['pal_mode']}", flush=True)
    elif isinstance(sd, dict) and "selfq" in sd:  # projection-in-the-loop self-cond probe (train_selfq.py)
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from train_selfq import build_selfq
        model = build_selfq(sd["selfq"], device)
        model.load_state_dict(sd["state"])
        import os
        model.use_sc = not os.environ.get("SELFQ_NOSC")  # ablation switch
        print(f"selfq model K={sd['selfq']['K']} use_sc={model.use_sc}", flush=True)
    elif isinstance(sd, dict) and "es" in sd:  # energy-score stochastic denoiser probe (train_es.py)
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from train_es import build_es
        model = build_es(sd["es"], device)
        model.load_state_dict(sd["state"])
        if args.steps is None:
            args.steps = sd["es"].get("sample_steps", 16)
        print(f"es model use_xi={model.use_xi} steps={args.steps}", flush=True)
    else:
        model = build_model(device)
        model.load_state_dict(sd)
    model.eval()
    if args.steps is None:
        args.steps = 100
    guide = None
    if args.guide_ckpt:
        guide = build_model(device)
        guide.load_state_dict(torch.load(args.guide_ckpt, map_location=device))
        guide.eval()
        print(f"autoguidance with weak model {args.guide_ckpt}", flush=True)
    if args.sampler == "ddim":
        scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=True)
    else:
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    cond = embed(prompts, tokenizer, enc, device).repeat_interleave(args.n, 0)
    uncond = embed([""] * len(prompts) * args.n, tokenizer, enc, device)
    for size in args.sizes:
        # chunked so 3000-prompt matched evals fit next to other jobs (one 3000 batch needs ~70GB);
        # chunk i uses seed+i, so results are seed-reproducible per chunk size
        imgs = torch.cat([sample(model, scheduler, cond[i:i + args.chunk], uncond[i:i + args.chunk], size, device,
                                 args.steps, args.cfg, args.seed + i // args.chunk, guide, cads=cads, gi=tuple(args.gi),
                                 guide_mode=args.guide_mode)
                          for i in range(0, cond.shape[0], args.chunk)])
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
