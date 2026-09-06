"""Probe: palette-factorised x0 head on the v7 UNet (cycle 3, 2026-09-06).

Continuous Gaussian DDPM unchanged.  Only the denoiser's OUTPUT parameterisation changes:
    f        = UNet features after conv_norm_out/conv_act (conv_out -> Identity)
    eps_d    = eps_direct(f)                       # copy of v7's conv_out  (direct branch)
    x0_d     = (x_t - sqrt(1-abar) eps_d) / sqrt(abar)
    logits   = logits(f)  -> K palette logits + 1 alpha logit per pixel
    P        = tanh(MLP(mean_hw f))  (B,K,3)       # per-image palette   ("mlp")
             | soft-k-means centroids of x0_d      #                     ("centroid")
    w        = softmax(logits/tau);  a = sigmoid(alpha logit)
    x0_p     = [a * (w @ P) + (1-a) * (-1),  2a-1]  # transparent RGB = -1 (data convention)
    x0       = lam * x0_p + (1-lam) * x0_d          # lam ramps 0 -> 1, so step 0 == v7
    eps_hat  = (x_t - sqrt(abar) x0) / sqrt(1-abar) # .sample -> sample_e.py / eval_probe.sh unchanged
Loss: x0-space MSE with min-SNR-gamma weighting (gamma=5) + palette-branch consistency +
slot-usage entropy (anti-collapse) + abar_t-weighted per-pixel confidence (anti-grey).
At the final sampling step the hard-assigned palette image (argmax, hard alpha) is returned
via .x0 (see `sample` below); sample_e.py has a matching 'palhead' branch.

Usage:
  python src/v6/train_palhead.py --init workdir/v7_lowres/model_latest.pt --steps 20000 --out workdir/probe_palhead
"""
import argparse
import copy
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from train_probe import (BATCH, BUCKETS, EVAL_PROMPTS, EVAL_SIZES, BucketSampler, NativeSprites,  # noqa: E402
                         embed, make_grid)


def build_unet(device):
    return UNet2DConditionModel(
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(128, 256, 512), cross_attention_dim=512,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        num_class_embeds=len(BUCKETS),
    ).to(device)


class PalHeadUNet(nn.Module):
    palhead = True

    def __init__(self, unet, K=16, pal_mode="mlp", hard_final=True):
        super().__init__()
        self.K, self.pal_mode, self.hard_final = K, pal_mode, hard_final
        ch = unet.conv_out.in_channels
        self.eps_direct = nn.Conv2d(ch, 4, 3, padding=1)
        self.eps_direct.load_state_dict(unet.conv_out.state_dict())  # step 0 == v7
        unet.conv_out = nn.Identity()
        self.unet = unet
        self.logits = nn.Conv2d(ch, K + 1, 3, padding=1)
        self.pal_mlp = nn.Sequential(nn.Linear(ch, 256), nn.SiLU(), nn.Linear(256, 3 * K))
        sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
        self.register_buffer("abar", sched.alphas_cumprod.float())
        self.register_buffer("sched", torch.tensor([0.0, 1.0]))  # [lam, tau]; saved in state_dict

    def set_sched(self, lam, tau):
        self.sched[0], self.sched[1] = lam, tau

    def forward(self, x_t, t, encoder_hidden_states=None, class_labels=None, hard=False):
        f = self.unet(x_t, t, encoder_hidden_states=encoder_hidden_states, class_labels=class_labels).sample
        tt = t if torch.is_tensor(t) else torch.tensor(t, device=x_t.device)
        ab = self.abar[tt.long()].view(-1, 1, 1, 1)
        sa, sb = ab.sqrt(), (1 - ab).sqrt()
        eps_d = self.eps_direct(f)
        x0_d = (x_t - sb * eps_d) / sa.clamp_min(1e-4)
        lg = self.logits(f)
        lam, tau = self.sched[0].item(), self.sched[1].item()
        w = F.softmax(lg[:, :self.K] / tau, 1)
        a01 = torch.sigmoid(lg[:, self.K:self.K + 1])
        if hard:
            w = F.one_hot(w.argmax(1), self.K).permute(0, 3, 1, 2).float()
            a01 = (a01 > 0.5).float()
        if self.pal_mode == "mlp":
            P = torch.tanh(self.pal_mlp(f.mean((2, 3)))).view(-1, self.K, 3)
        else:  # soft k-means centroids of the direct prediction, weighted by assignment x opacity
            r = x0_d[:, :3].detach().clamp(-1, 1)
            wm = w * a01
            P = torch.einsum("bkhw,bchw->bkc", wm, r) / wm.sum((2, 3)).clamp_min(1e-4).unsqueeze(-1)
        rgb_p = torch.einsum("bkhw,bkc->bchw", w, P)
        x0_p = torch.cat([a01 * rgb_p + (1 - a01) * (-1.0), 2 * a01 - 1], 1)
        x0 = lam * x0_p + (1 - lam) * x0_d
        eps = (x_t - sa * x0) / sb.clamp_min(1e-4)
        return SimpleNamespace(sample=eps, x0=x0, x0_p=x0_p, x0_d=x0_d, w=w, a01=a01, P=P, ab=ab)


def build_palhead(cfg, device):
    unet = build_unet(device)
    return PalHeadUNet(unet, K=cfg["K"], pal_mode=cfg["pal_mode"], hard_final=cfg.get("hard_final", True)).to(device)


def load_palhead(ckpt, device):
    sd = torch.load(ckpt, map_location=device)
    m = build_palhead(sd["palhead"], device)
    m.load_state_dict(sd["state"])
    return m.eval()


@torch.no_grad()
def sample(model, scheduler, cond, uncond, size, device="cuda", steps=100, cfg=4.0, generator=None):
    """DDPM sampling with CFG on eps; the final step returns the hard palette image from the
    conditional branch (at t=0 the scheduler ignores eps almost entirely, so the snap must be explicit)."""
    scheduler.set_timesteps(steps)
    n = cond.shape[0]
    lab = torch.full((n,), BUCKETS.index(size), device=device, dtype=torch.long)
    x = torch.randn(n, 4, size, size, device=device, generator=generator)
    last = scheduler.timesteps[-1]
    for t in scheduler.timesteps:
        if t == last and getattr(model, "palhead", False) and model.hard_final:
            x = model(x, t, encoder_hidden_states=cond, class_labels=lab, hard=True).x0.clamp(-1, 1)
            break
        e_c = model(x, t, encoder_hidden_states=cond, class_labels=lab).sample
        e_u = model(x, t, encoder_hidden_states=uncond, class_labels=lab).sample
        x = scheduler.step(e_u + cfg * (e_c - e_u), t, x).prev_sample
    return ((x + 1) / 2).clamp(0, 1).cpu()


def schedules(step, lam_steps, tau_start, tau_end, tau_s0, tau_s1):
    lam = min(1.0, step / max(1, lam_steps))
    if step <= tau_s0:
        tau = tau_start
    elif step >= tau_s1:
        tau = tau_end
    else:
        u = (step - tau_s0) / (tau_s1 - tau_s0)
        tau = tau_end + (tau_start - tau_end) * 0.5 * (1 + math.cos(math.pi * u))
    return lam, tau


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="v7 checkpoint")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", required=True)
    p.add_argument("--sample_every", type=int, default=4000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--pal_mode", default="mlp", choices=["mlp", "centroid"])
    p.add_argument("--snr_gamma", type=float, default=5.0)
    p.add_argument("--lam_steps", type=int, default=4000)
    p.add_argument("--tau", type=float, nargs=2, default=[1.0, 0.3], help="tau start/end")
    p.add_argument("--tau_steps", type=int, nargs=2, default=[2000, 10000], help="tau anneal window")
    p.add_argument("--w_pal", type=float, default=0.1, help="palette-branch consistency weight")
    p.add_argument("--w_use", type=float, default=0.01, help="slot-usage entropy (anti-collapse)")
    p.add_argument("--w_conf", type=float, default=0.01, help="abar-weighted per-pixel confidence (anti-grey)")
    p.add_argument("--bs_scale", type=float, default=1.0)
    args = p.parse_args()
    for k in BATCH:
        BATCH[k] = max(8, int(BATCH[k] * args.bs_scale))
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

    unet = build_unet(device)
    sd = torch.load(args.init, map_location=device)
    unet.load_state_dict(sd["unet"] if isinstance(sd, dict) and "unet" in sd else sd)
    model = PalHeadUNet(unet, K=args.K, pal_mode=args.pal_mode).to(device)
    n_head = sum(q.numel() for n, q in model.named_parameters() if not n.startswith("unet."))
    print(f"unet {sum(q.numel() for q in model.unet.parameters()) / 1e6:.1f}M + head {n_head / 1e3:.1f}k "
          f"(K={args.K}, {args.pal_mode}); init from {args.init}", flush=True)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if args.ema > 0 else None

    eval_cond = embed(EVAL_PROMPTS, tokenizer, text_encoder, device)
    eval_uncond = embed([""] * len(EVAL_PROMPTS), tokenizer, text_encoder, device)
    cfg = {"K": args.K, "pal_mode": args.pal_mode, "hard_final": True, "buckets": BUCKETS}

    step = 0
    for x, texts, b in loader:
        lam, tau = schedules(step, args.lam_steps, args.tau[0], args.tau[1], *args.tau_steps)
        model.set_sched(lam, tau)
        texts = ["" if random.random() < 0.1 else t for t in texts]
        cond = embed(list(texts), tokenizer, text_encoder, device)
        x, b = x.to(device), b.to(device)
        noise = torch.randn_like(x)
        t = torch.randint(0, 1000, (x.shape[0],), device=device)
        xt = scheduler.add_noise(x, noise, t)
        o = model(xt, t, encoder_hidden_states=cond, class_labels=b)
        ab = o.ab.view(-1)
        wt = (ab / (1 - ab)).clamp(max=args.snr_gamma)  # min-SNR-gamma weight == eps-MSE where SNR < gamma
        per = lambda y: ((y - x) ** 2).mean((1, 2, 3))  # noqa: E731
        loss_main = (wt * per(o.x0)).mean()
        loss_pal = (wt * per(o.x0_p)).mean()
        opaque = (x[:, 3:4] > 0).float()
        u = (o.w * opaque).sum((0, 2, 3)) / opaque.sum().clamp_min(1)  # batch-level slot usage
        ent_u = -(u * (u + 1e-8).log()).sum()
        loss_use = math.log(args.K) - ent_u
        h_pix = -(o.w * (o.w + 1e-8).log()).sum(1, keepdim=True)  # per-pixel assignment entropy
        loss_conf = ((h_pix * opaque).sum((1, 2, 3)) / opaque.sum((1, 2, 3)).clamp_min(1) * ab).mean()
        loss = loss_main + args.w_pal * loss_pal + args.w_use * loss_use + args.w_conf * loss_conf
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if ema is not None:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1.0 - args.ema)
                ema.sched.copy_(model.sched)
        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                peak = (o.w.max(1)[0] * opaque[:, 0]).sum() / opaque.sum().clamp_min(1)
                Pd = torch.cdist(o.P, o.P) + torch.eye(args.K, device=device) * 9
                pmin = Pd.min(-1)[0].mean()
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} main={loss_main.item():.4f} pal={loss_pal.item():.4f} "
                  f"useH={ent_u.item():.2f}/{math.log(args.K):.2f} peak={peak.item():.2f} palsep={pmin.item():.3f} "
                  f"lam={lam:.2f} tau={tau:.2f} bucket={BUCKETS[int(b[0])]}", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            net = ema if ema is not None else model
            net.eval()
            for s in EVAL_SIZES:
                g = torch.Generator(device=device).manual_seed(args.seed)
                make_grid(sample(net, scheduler, eval_cond, eval_uncond, s, device, generator=g)).save(
                    out / "samples" / f"step_{step:06d}_s{s}.png")
            torch.save({"palhead": cfg, "state": net.state_dict()}, out / "model_latest.pt")
            model.train()
    print(f"Done -> {out}", flush=True)


if __name__ == "__main__":
    main()
