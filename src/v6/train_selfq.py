"""Probe: projection-in-the-loop self-conditioning ("selfq").

Continuous RGBA DDPM (v7 / probe_tv weights) whose input is widened 4 -> 8 channels: the
extra 4 channels carry a STOP-GRAD, NON-LEARNED discrete projection P(x0_hat) of the
model's own previous x0 estimate -- per-image k-means colour quantisation (K colours over
opaque pixels, hard alpha).  Bit-Diffusion-style: during training the self-cond input is
zeros with p=0.5, otherwise P(x0_hat) from a no-grad forward; at sampling every step feeds
the projected estimate of the previous step.  Loss unchanged (eps MSE, optional TV).

Why: the network sees a hard-edged, piecewise-constant hypothesis of its own output and
learns to CORRECT it, instead of averaging modes (v7) or being forced through a learned
bottleneck (palhead, collapsed).  P is not learned and not in the loss, so it cannot collapse;
the train/test mismatch of external post-hoc snapping is closed because the projection is
inside the loop at both times.

Usage:
  python src/v6/train_selfq.py --init workdir/probe_tv/model_latest.pt --tv 0.1 --steps 20000 --out workdir/probe_selfq
"""
import argparse
import copy
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel

sys.path.insert(0, str(Path(__file__).parent))
from train_probe import (BATCH, BUCKETS, EVAL_PROMPTS, EVAL_SIZES, BucketSampler,  # noqa: E402
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


@torch.no_grad()
def kmeans_quantise(x0, K=16, iters=6, seed=None):
    """Per-image k-means over opaque pixels' RGB (in [-1,1]); returns projected RGBA in [-1,1]
    with transparent RGB=-1 and hard alpha.  x0: (B,4,H,W).  Stop-grad by construction."""
    B, _, H, W = x0.shape
    x0 = x0.float()
    rgb = x0[:, :3].permute(0, 2, 3, 1).reshape(B, H * W, 3)
    m = (x0[:, 3] > 0).reshape(B, H * W)  # opaque mask
    mf = m.float()
    g = torch.Generator(device=x0.device)
    if seed is not None:
        g.manual_seed(seed)
    # init: K random opaque pixels per image (fallback to any pixel if too few opaque)
    w = mf + 1e-6
    idx = torch.multinomial(w, K, replacement=True, generator=g)  # (B,K)
    C = torch.gather(rgb, 1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (B,K,3)
    for _ in range(iters):
        d = torch.cdist(rgb, C)  # (B,N,K)
        a = d.argmin(-1)  # (B,N)
        oh = F.one_hot(a, K).float() * mf.unsqueeze(-1)  # only opaque pixels vote
        cnt = oh.sum(1)  # (B,K)
        newC = torch.einsum("bnk,bnc->bkc", oh, rgb) / cnt.clamp_min(1).unsqueeze(-1)
        C = torch.where(cnt.unsqueeze(-1) > 0, newC, C)
    d = torch.cdist(rgb, C)
    a = d.argmin(-1)
    q = torch.gather(C, 1, a.unsqueeze(-1).expand(-1, -1, 3))  # (B,N,3)
    q = torch.where(m.unsqueeze(-1), q, torch.full_like(q, -1.0))
    q = q.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    alpha = (2 * mf - 1).reshape(B, 1, H, W)
    return torch.cat([q, alpha], 1).clamp(-1, 1)


class SelfQUNet(nn.Module):
    """UNet with conv_in widened to 8 channels: [x_t, self-cond P(x0_hat) or zeros]."""
    selfcond = True

    def __init__(self, unet, K=16, iters=6):
        super().__init__()
        old = unet.conv_in
        new = nn.Conv2d(8, old.out_channels, old.kernel_size, padding=old.padding)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :4] = old.weight
            new.bias.copy_(old.bias)
        unet.conv_in = new
        self.unet = unet
        self.K, self.iters = K, iters
        self.register_buffer("abar", DDPMScheduler(1000, beta_schedule="squaredcos_cap_v2").alphas_cumprod.float())

    def forward(self, x_t, t, encoder_hidden_states=None, class_labels=None, sc=None):
        if sc is None:
            sc = torch.zeros_like(x_t)
        return self.unet(torch.cat([x_t, sc], 1), t, encoder_hidden_states=encoder_hidden_states,
                         class_labels=class_labels)

    def x0_from_eps(self, x_t, t, eps):
        tt = t if torch.is_tensor(t) else torch.tensor(t, device=x_t.device)
        ab = self.abar[tt.long()].view(-1, 1, 1, 1) if tt.dim() else self.abar[tt.long()]
        return ((x_t - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-4)).clamp(-1, 1)

    def project(self, x0):
        return kmeans_quantise(x0, self.K, self.iters)

    def project_from_eps(self, x_t, t, eps):
        return self.project(self.x0_from_eps(x_t, t, eps))


def build_selfq(cfg, device):
    return SelfQUNet(build_unet(device), K=cfg["K"], iters=cfg.get("iters", 6)).to(device)


@torch.no_grad()
def sample(model, scheduler, cond, uncond, size, device="cuda", steps=100, cfg=4.0, seed=None):
    scheduler.set_timesteps(steps)
    n = cond.shape[0]
    lab = torch.full((n,), BUCKETS.index(size), device=device, dtype=torch.long)
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(n, 4, size, size, device=device)
    sc = None
    for t in scheduler.timesteps:
        e_c = model(x, t, encoder_hidden_states=cond, class_labels=lab, sc=sc).sample
        e_u = model(x, t, encoder_hidden_states=uncond, class_labels=lab, sc=sc).sample
        e = e_u + cfg * (e_c - e_u)
        sc = model.project_from_eps(x, t, e)
        x = scheduler.step(e, t, x).prev_sample
    return ((x + 1) / 2).clamp(0, 1).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="4-channel v7-family checkpoint (v7 or probe_tv)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", default="workdir/probe_selfq")
    p.add_argument("--sample_every", type=int, default=4000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--p_sc", type=float, default=0.5, help="prob. of feeding self-cond during training")
    p.add_argument("--tv", type=float, default=0.1, help="TV weight on x0_hat (0 = off); probe_tv used 0.1")
    p.add_argument("--bs_scale", type=float, default=1.0)
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
    ds = NativeSprites(sources)
    print(f"dataset: {len(ds)}", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=BucketSampler(ds.bucket_of, args.steps), num_workers=8)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    text_encoder.requires_grad_(False)

    unet = build_unet(device)
    unet.load_state_dict(torch.load(args.init, map_location=device))
    model = SelfQUNet(unet, K=args.K, iters=args.iters).to(device)
    print(f"init from {args.init}; params {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M", flush=True)
    cfg = {"K": args.K, "iters": args.iters, "buckets": BUCKETS}

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    abar_all = scheduler.alphas_cumprod.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if args.ema > 0 else None

    eval_cond = embed(EVAL_PROMPTS, tokenizer, text_encoder, device)
    eval_uncond = embed([""] * len(EVAL_PROMPTS), tokenizer, text_encoder, device)

    step = 0
    for x, texts, b in loader:
        texts = ["" if random.random() < 0.1 else t for t in texts]
        cond = embed(list(texts), tokenizer, text_encoder, device)
        x, b = x.to(device), b.to(device)
        noise = torch.randn_like(x)
        t = torch.randint(0, 1000, (x.shape[0],), device=device)
        xt = scheduler.add_noise(x, noise, t)
        abar = abar_all[t].view(-1, 1, 1, 1)
        use_sc = random.random() < args.p_sc
        sc = None
        if use_sc:
            with torch.no_grad():
                model.eval()
                e0 = model(xt, t, encoder_hidden_states=cond, class_labels=b).sample
                model.train()
                sc = model.project(((xt - (1 - abar).sqrt() * e0) / abar.sqrt().clamp_min(1e-4)).clamp(-1, 1))
        pred = model(xt, t, encoder_hidden_states=cond, class_labels=b, sc=sc).sample
        loss = F.mse_loss(pred, noise)
        main_l = loss.item()
        if args.tv > 0:
            x0p = ((xt - (1 - abar).sqrt() * pred) / abar.sqrt().clamp_min(1e-4)).clamp(-1, 1)
            tv = (x0p[:, :, 1:] - x0p[:, :, :-1]).abs().mean() + (x0p[:, :, :, 1:] - x0p[:, :, :, :-1]).abs().mean()
            loss = loss + args.tv * tv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if ema is not None:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1.0 - args.ema)
        if step % 200 == 0:
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} main={main_l:.4f} sc={int(use_sc)} "
                  f"bucket={BUCKETS[int(b[0])]}", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            net = ema if ema is not None else model
            net.eval()
            for s in EVAL_SIZES:
                make_grid(sample(net, scheduler, eval_cond, eval_uncond, s, seed=args.seed)).save(
                    out / "samples" / f"step_{step:06d}_s{s}.png")
            torch.save({"selfq": cfg, "state": net.state_dict()}, out / "model_latest.pt")
            model.train()
    print(f"Done -> {out}", flush=True)


if __name__ == "__main__":
    main()
