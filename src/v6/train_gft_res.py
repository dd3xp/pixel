"""Candidate 1b (arch_crossres_module.md, 2.1): GFT-form internaliser of cross-resolution guidance, 1 NFE.

Why now: CRSC's gates T0 and T1 both said STOP (w* 0.83-1.07 on forward-noised data, <= 1.07 on self-rolled
states), so no module trained with the plain denoising loss can learn the guidance. The remaining route that
still trains only on data targets is to rewrite the loss so its fixed point IS the guided prediction -- GFT
(Chen et al., ICML 2025, arXiv 2501.15420), with CFG's unconditional term replaced by our lower-resolution
reference:

    eps_pred = beta * s_theta(x_t, c, b, beta) + (1 - beta) * sg[ r(x_t, c, b_low) ]
    loss     = || eps_pred - eps ||^2,          beta ~ U(0.4, 1), beta = 1 for 25 % of rows
At the optimum s = r + (1/beta)(mu - r): sampling s_theta(., beta = 1/w) in ONE forward pass is the guided rule.
  r = the frozen step-10k EMA snapshot under the lower label (composed reference; the snapshot is needed only in
      training), or --ref online: this network itself at beta = 1 under the lower label (label reference).
Novelty is incremental (GFT with a new reference); its value is cost: the composed rule (7.53 at 2 NFE + a stored
snapshot) at 1 NFE and nothing stored. Bars: keep as a row if 1-NFE FD <= 8.5; claim INTERNALISER if <= 7.8.

beta enters through diffusers' `time_cond_proj_dim` (the LCM guidance-scale input): a sinusoidal embedding of beta,
projected by a ZERO-initialised linear layer and added to the timestep embedding, so step 0 is exactly v7h.
Buckets without a lower rung (12 px) always train at beta = 1.
"""
import argparse
import copy
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, "src/v6")
import train_v7
from train_v7 import BUCKETS, BucketSampler, NativeSprites, embed

LOWER = {12: None, 16: 12, 20: 16, 24: 16, 32: 24, 48: 32, 64: 48}   # the reference rung used by the sampler
BETA_DIM = 256


def beta_embedding(beta, dim=BETA_DIM):
    """LCM's guidance-scale embedding (sinusoidal of 1000*beta)."""
    w = beta.float() * 1000.0
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=beta.device, dtype=torch.float32) * -(math.log(10000.0) / (half - 1)))
    e = w[:, None] * freqs[None]
    return torch.cat([torch.sin(e), torch.cos(e)], dim=1)


def build(width=128, cond_dim=BETA_DIM):
    return UNet2DConditionModel(
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(width, 2 * width, 4 * width), cross_attention_dim=512,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        num_class_embeds=len(BUCKETS), time_cond_proj_dim=cond_dim)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", default="workdir/probe_gft")
    p.add_argument("--init", default="workdir/v7h/model_latest.pt")
    p.add_argument("--snap", default="workdir/v7h/model_step010000.pt")
    p.add_argument("--ref", default="snapshot", choices=["snapshot", "online"])
    p.add_argument("--beta_lo", type=float, default=0.4)
    p.add_argument("--p_plain", type=float, default=0.25)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--bs_scale", type=float, default=1.0)
    p.add_argument("--csv_suffix", default="")
    p.add_argument("--exclude", default="runs_out/holdout_exclude.txt")
    p.add_argument("--ckpt_every", type=int, default=2500)
    args = p.parse_args()
    dev = "cuda"
    for k in train_v7.BATCH:
        train_v7.BATCH[k] = max(8, int(train_v7.BATCH[k] * args.bs_scale))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    suf = args.csv_suffix
    sources = [("data/oga_clean", f"data/oga_captions{suf}.csv", 1),
               ("data/extra_all", f"data/extra_all{suf}.csv", 1),
               ("data/oga_clean", f"data/tool_candidates{suf}.csv", 2)]
    exclude = [l.strip() for l in open(args.exclude) if l.strip()]
    ds = NativeSprites(sources, exclude)
    print(f"dataset: {len(ds)}", flush=True)

    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    txt = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval().requires_grad_(False)

    model = build().to(dev)
    missing, unexpected = model.load_state_dict(torch.load(args.init, map_location=dev), strict=False)
    assert missing == ["time_embedding.cond_proj.weight"] and not unexpected, (missing, unexpected)
    with torch.no_grad():
        model.time_embedding.cond_proj.weight.zero_()          # step 0 == the base model
    ref_net = None
    if args.ref == "snapshot":
        ref_net = build().to(dev).eval().requires_grad_(False)
        ref_net.load_state_dict(torch.load(args.snap, map_location=dev), strict=False)
        ref_net.time_embedding.cond_proj.weight.data.zero_()
    print(f"init {args.init}; reference = {args.ref}"
          + (f" ({args.snap})" if ref_net is not None else ""), flush=True)

    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    lower_of = torch.tensor([BUCKETS.index(LOWER[s]) if LOWER[s] else -1 for s in BUCKETS], device=dev)

    step = 0
    resume = out / "ckpt.pt"
    if resume.exists():
        ck = torch.load(resume, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); ema.load_state_dict(ck["ema"])
        step = ck["step"]
        print(f"resumed at step {step}", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=BucketSampler(ds.bucket_of, args.steps - step),
                                         num_workers=8)
    for x, texts, b in loader:
        texts = ["" if random.random() < 0.1 else t for t in texts]
        cond = embed(list(texts), tok, txt, dev)
        x, b = x.to(dev), b.to(dev)
        B = x.shape[0]
        noise = torch.randn_like(x)
        t = torch.randint(0, 1000, (B,), device=dev)
        xt = sched.add_noise(x, noise, t)
        low = lower_of[b]
        beta = torch.empty(B, device=dev).uniform_(args.beta_lo, 1.0)
        beta = torch.where((torch.rand(B, device=dev) < args.p_plain) | (low < 0), torch.ones_like(beta), beta)
        s = model(xt, t, encoder_hidden_states=cond, class_labels=b, timestep_cond=beta_embedding(beta)).sample
        need = beta < 1.0
        r = torch.zeros_like(s)
        if need.any():
            with torch.no_grad():
                net = ref_net if ref_net is not None else model
                one = torch.ones(int(need.sum()), device=dev)
                r[need] = net(xt[need], t[need], encoder_hidden_states=cond[need], class_labels=low[need],
                              timestep_cond=beta_embedding(one)).sample
        bb = beta.view(-1, 1, 1, 1)
        loss = F.mse_loss(bb * s + (1 - bb) * r, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1.0 - args.ema)
        if step % 200 == 0:
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} bucket={BUCKETS[int(b[0])]} "
                  f"mean_beta={beta.mean().item():.2f}", flush=True)
        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save(ema.state_dict(), out / "model_latest.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "ema": ema.state_dict(),
                        "step": step}, resume)
        if step >= args.steps:
            break
    torch.save(ema.state_dict(), out / "model_latest.pt")
    print("GFT_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
