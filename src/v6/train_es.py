"""Probe: energy-score stochastic denoiser ("es").

The v7 UNet is turned from a POSTERIOR-MEAN predictor into a POSTERIOR-SAMPLE generator:
conv_in is widened 4 -> 8 channels and the extra channels carry a fresh latent xi ~ N(0, I)
(zero-initialised weights, so training starts exactly at the MSE solution).  Loss = energy
score (a strictly proper scoring rule, Gneiting & Raftery 2007; used for diffusion denoisers by
DDM-SR 2502.02483):

    L = E_xi ||eps - eps_hat(xi)||^beta  -  lam / (2 (m-1)) * sum_{j != j'} ||eps_hat(xi_j) - eps_hat(xi_j')||^beta

with m draws of xi per training example.  Minimised iff eps_hat(xi) ~ p(eps | x_t, c), i.e. the
denoiser returns a SAMPLE of the clean image instead of the blurry mean that eps-MSE forces at
16px (the mean of a multimodal posterior is the source of the coverage gap, fd_decomp.py).
beta = 1 and the norm is over the whole 4xHxW image; eps-space and x0-space scores differ only by
a per-sample scale, so the standard eps weighting is kept.  lam warms 0 -> lam over --warm steps
(lam = 1 is the proper energy score); the pairwise term is logged so the known failure mode
(network ignores xi, pairwise term -> 0, loss degenerates to MAE) is visible.

Sampling (sample_e.py, "es" ckpt): few-step re-noising -- at each of K steps draw xi, predict
x0_hat(xi), re-noise to the next timestep; final step returns x0_hat.  CFG applies to eps_hat
with a shared xi.

Usage:
  python src/v6/train_es.py --init workdir/v7h/model_latest.pt --exclude runs_out/holdout_exclude.txt \
      --steps 20000 --out workdir/probe_es
  paired control (same data/steps, plain MSE):  --mse --m 1 --no_xi
"""
import argparse
import copy
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel

sys.path.insert(0, str(Path(__file__).parent))
from train_v7 import (BATCH, BUCKETS, EVAL_PROMPTS, EVAL_SIZES, BucketSampler,  # noqa: E402
                      NativeSprites, embed, make_grid)
from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402


def build_unet(device):
    return UNet2DConditionModel(
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(128, 256, 512), cross_attention_dim=512,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        num_class_embeds=len(BUCKETS),
    ).to(device)


class ESUNet(nn.Module):
    """UNet with conv_in widened to 8 channels: [x_t, xi]."""
    es = True

    def __init__(self, unet, use_xi=True):
        super().__init__()
        old = unet.conv_in
        new = nn.Conv2d(8, old.out_channels, old.kernel_size, padding=old.padding)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :4] = old.weight
            new.bias.copy_(old.bias)
        unet.conv_in = new
        self.unet = unet
        self.use_xi = use_xi
        self.register_buffer("abar", DDPMScheduler(1000, beta_schedule="squaredcos_cap_v2").alphas_cumprod.float())

    def forward(self, x_t, t, encoder_hidden_states=None, class_labels=None, xi=None):
        if xi is None or not self.use_xi:
            xi = torch.zeros_like(x_t)
        return self.unet(torch.cat([x_t, xi], 1), t, encoder_hidden_states=encoder_hidden_states,
                         class_labels=class_labels)

    def x0_from_eps(self, x_t, t, eps):
        tt = t if torch.is_tensor(t) else torch.tensor(t, device=x_t.device)
        ab = self.abar[tt.long()].view(-1, 1, 1, 1) if tt.dim() else self.abar[tt.long()]
        return ((x_t - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-4)).clamp(-1, 1)


def build_es(cfg, device):
    return ESUNet(build_unet(device), use_xi=cfg.get("use_xi", True)).to(device)


def energy_score(pred, target, m, lam, beta):
    """pred: (m*B, C, H, W) in xi-major order [xi_0 batch, xi_1 batch, ...]; target: (B, C, H, W).
    Returns (loss, fidelity term, pairwise term) -- pairwise term is the collapse monitor."""
    B = target.shape[0]
    p = pred.view(m, B, -1)
    fid = (p - target.view(1, B, -1)).norm(dim=-1).pow(beta).mean()
    if m < 2 or lam == 0:
        return fid, fid.detach(), torch.zeros((), device=pred.device)
    d = torch.cdist(p.transpose(0, 1), p.transpose(0, 1))  # (B, m, m)
    pair = d.pow(beta).sum(dim=(1, 2)) / (m * (m - 1))      # mean over ordered pairs j != j'
    pair = pair.mean()
    return fid - lam * 0.5 * pair, fid.detach(), pair.detach()


@torch.no_grad()
def sample_es(model, cond, uncond, size, device="cuda", steps=16, cfg=4.0, seed=None):
    """Few-step re-noising sampler for a posterior-sample denoiser."""
    n = cond.shape[0]
    lab = torch.full((n,), BUCKETS.index(size), device=device, dtype=torch.long)
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    x = torch.randn(n, 4, size, size, device=device, generator=g)
    ts = torch.linspace(999, 0, steps + 1).round().long().tolist()  # 999 -> ... -> 0 (last is unused)
    for i, t in enumerate(ts[:-1]):
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        xi = torch.randn(n, 4, size, size, device=device, generator=g)
        e_c = model(x, tt, encoder_hidden_states=cond, class_labels=lab, xi=xi).sample
        e_u = model(x, tt, encoder_hidden_states=uncond, class_labels=lab, xi=xi).sample
        x0 = model.x0_from_eps(x, tt, e_u + cfg * (e_c - e_u))
        t_next = ts[i + 1]
        if t_next <= 0:
            x = x0
            break
        ab = model.abar[t_next]
        x = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn(x.shape, device=device, generator=g)
    return ((x + 1) / 2).clamp(0, 1).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="4-channel v7-family checkpoint (v7h)")
    p.add_argument("--exclude", default="runs_out/holdout_exclude.txt")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", default="workdir/probe_es")
    p.add_argument("--sample_every", type=int, default=4000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m", type=int, default=4, help="xi draws per example")
    p.add_argument("--lam", type=float, default=1.0, help="pairwise (repulsion) weight; 1 = proper energy score")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--warm", type=int, default=2000, help="linear warm-up of lam")
    p.add_argument("--no_xi", action="store_true", help="control: xi channels forced to zero (deterministic)")
    p.add_argument("--mse", action="store_true", help="control: plain eps-MSE loss (with --m 1 --no_xi = paired v7 recipe)")
    p.add_argument("--sample_steps", type=int, default=16)
    p.add_argument("--bs_scale", type=float, default=0.5, help="batch scaled down because m copies are run")
    args = p.parse_args()
    for k in BATCH:
        BATCH[k] = max(8, int(BATCH[k] * args.bs_scale))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    sources = [("data/oga_clean", "data/oga_captions.csv", 1), ("data/extra_all", "data/extra_all.csv", 1),
               ("data/oga_clean", "data/tool_candidates.csv", 2)]
    exclude = [l.strip() for l in open(args.exclude) if l.strip()] if args.exclude else []
    ds = NativeSprites(sources, exclude)
    print(f"dataset: {len(ds)}", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=BucketSampler(ds.bucket_of, args.steps), num_workers=8)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    text_encoder.requires_grad_(False)

    unet = build_unet(device)
    unet.load_state_dict(torch.load(args.init, map_location=device))
    model = ESUNet(unet, use_xi=not args.no_xi).to(device)
    print(f"init from {args.init}; params {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M; "
          f"m={args.m} lam={args.lam} beta={args.beta} use_xi={model.use_xi}", flush=True)
    cfg = {"use_xi": model.use_xi, "buckets": BUCKETS, "m": args.m, "lam": args.lam, "beta": args.beta,
           "sample_steps": args.sample_steps}

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if args.ema > 0 else None

    eval_cond = embed(EVAL_PROMPTS, tokenizer, text_encoder, device)
    eval_uncond = embed([""] * len(EVAL_PROMPTS), tokenizer, text_encoder, device)

    step = 0
    m = args.m
    for x, texts, b in loader:
        texts = ["" if random.random() < 0.1 else t for t in texts]
        cond = embed(list(texts), tokenizer, text_encoder, device)
        x, b = x.to(device), b.to(device)
        B = x.shape[0]
        noise = torch.randn_like(x)
        t = torch.randint(0, 1000, (B,), device=device)
        xt = scheduler.add_noise(x, noise, t)
        # m xi-draws share the same (x_t, t, c): stack xi-major so energy_score can regroup
        xi = torch.randn(m * B, 4, x.shape[2], x.shape[3], device=device)
        pred = model(xt.repeat(m, 1, 1, 1), t.repeat(m), encoder_hidden_states=cond.repeat(m, 1, 1),
                     class_labels=b.repeat(m), xi=xi).sample
        lam = args.lam * min(1.0, (step + 1) / max(1, args.warm))
        if args.mse:
            loss = F.mse_loss(pred, noise.repeat(m, 1, 1, 1))
            fid, pair = loss.detach(), torch.zeros((), device=device)
        else:
            loss, fid, pair = energy_score(pred, noise, m, lam, args.beta)
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
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} fid={fid.item():.4f} pair={pair.item():.4f} "
                  f"lam={lam:.2f} bucket={BUCKETS[int(b[0])]}", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            net = ema if ema is not None else model
            net.eval()
            for s in EVAL_SIZES:
                make_grid(sample_es(net, eval_cond, eval_uncond, s, steps=args.sample_steps, seed=args.seed)).save(
                    out / "samples" / f"step_{step:06d}_s{s}.png")
            torch.save({"es": cfg, "state": net.state_dict()}, out / "model_latest.pt")
            model.train()
    print(f"Done -> {out}", flush=True)


if __name__ == "__main__":
    main()
