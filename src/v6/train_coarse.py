"""Coarse-belief guidance probe (probe_cg): fine-tune v7h so that the SAME network also carries a
"coarse view" of every sprite, addressed by a second set of resolution labels (bucket idx + 7).

Motivation (experiment_log 09-07): using the model's own LOWER-resolution bucket embedding as the
autoguidance reference (bucket:12 while sampling 16 px) cuts matched FD 21.98 -> 8.59 with zero
training, while higher-resolution embeddings do nothing.  The lower-res belief is the pixel-bleeding /
mode-averaged version of the same sprite, i.e. Karras' "compatible bad model" along the resolution
axis.  Here we make that reference explicit and trainable: with probability --coarse_p a training
sample is replaced by its k x k block-averaged (then nearest-upsampled) version and labelled
"coarse-<bucket>".  Sampling: e = e_coarse + w (e_fine - e_coarse)  (sample_e.py --guide_mode coarse),
optionally stacked with an early EMA snapshot as the coarse model (--guide_ckpt).
Works at every resolution (the zero-training trick needs a lower bucket to exist, so not at 12 px).

Usage: python src/v6/train_coarse.py --init workdir/v7h/model_latest.pt --steps 20000 --out workdir/probe_cg
"""
import argparse
import copy
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from train_v7 import BATCH, BUCKETS, EVAL_PROMPTS, EVAL_SIZES, BucketSampler, NativeSprites, embed, make_grid  # noqa: E402

NB = len(BUCKETS)


def degrade(x, k):
    """k x k block average of a [-1,1] RGBA batch, nearest-upsampled back (alpha averaged too)."""
    h = x.shape[-1]
    return F.interpolate(F.avg_pool2d(x, k, ceil_mode=True), size=(h, h), mode="nearest")


def build_coarse(device):
    return UNet2DConditionModel(
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(128, 256, 512), cross_attention_dim=512,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        num_class_embeds=2 * NB,
    ).to(device)


def load_expanded(model, ckpt, device):
    """7-label v7h state dict -> 14-label model; coarse rows start as copies of the fine rows."""
    sd = torch.load(ckpt, map_location=device)
    key = "class_embedding.weight"
    if sd[key].shape[0] == NB:
        sd[key] = torch.cat([sd[key], sd[key].clone()], 0)
    model.load_state_dict(sd)


@torch.no_grad()
def sample_cg(model, scheduler, cond, size, device, w=2.0, steps=100):
    scheduler.set_timesteps(steps)
    n = cond.shape[0]
    lab = torch.full((n,), BUCKETS.index(size), device=device, dtype=torch.long)
    x = torch.randn(n, 4, size, size, device=device)
    for t in scheduler.timesteps:
        e_f = model(x, t, encoder_hidden_states=cond, class_labels=lab).sample
        e_c = model(x, t, encoder_hidden_states=cond, class_labels=lab + NB).sample
        x = scheduler.step(e_c + w * (e_f - e_c), t, x).prev_sample
    return ((x + 1) / 2).clamp(0, 1).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="v7h EMA checkpoint (7 labels) or a 14-label checkpoint")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", default="workdir/probe_cg")
    p.add_argument("--exclude", default="runs_out/holdout_exclude.txt")
    p.add_argument("--k", type=int, default=2, help="block size of the coarse degradation")
    p.add_argument("--coarse_p", type=float, default=0.5, help="fraction of samples trained as coarse view")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bs_scale", type=float, default=1.0)
    p.add_argument("--sample_every", type=int, default=2500)
    p.add_argument("--snap_every", type=int, default=5000)
    args = p.parse_args()
    for kk in BATCH:
        BATCH[kk] = max(8, int(BATCH[kk] * args.bs_scale))
    device = "cuda"
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    sources = [("data/oga_clean", "data/oga_captions.csv", 1),
               ("data/extra_all", "data/extra_all.csv", 1),
               ("data/oga_clean", "data/tool_candidates.csv", 2)]
    exclude = [l.strip() for l in open(args.exclude) if l.strip()] if args.exclude else []
    ds = NativeSprites(sources, exclude)
    print(f"dataset: {len(ds)}", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=BucketSampler(ds.bucket_of, args.steps), num_workers=8)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    text_encoder.requires_grad_(False)

    model = build_coarse(device)
    load_expanded(model, args.init, device)
    print(f"init from {args.init}; params {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M", flush=True)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if args.ema > 0 else None
    eval_cond = embed(EVAL_PROMPTS, tokenizer, text_encoder, device)

    step = 0
    resume = out / "ckpt.pt"
    if resume.exists():
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        step = ck["step"]
        loader = torch.utils.data.DataLoader(ds, batch_sampler=BucketSampler(ds.bucket_of, args.steps - step), num_workers=8)
        print(f"resumed at step {step}", flush=True)
    lf = lc = 0.0
    for x, texts, b in loader:
        texts = ["" if random.random() < 0.1 else t for t in texts]
        cond = embed(list(texts), tokenizer, text_encoder, device)
        x, b = x.to(device), b.to(device)
        coarse = torch.rand(x.shape[0], device=device) < args.coarse_p
        if coarse.any():
            x = torch.where(coarse[:, None, None, None], degrade(x, args.k), x)
            b = b + coarse.long() * NB
        noise = torch.randn_like(x)
        t = torch.randint(0, 1000, (x.shape[0],), device=device)
        pred = model(scheduler.add_noise(x, noise, t), t, encoder_hidden_states=cond, class_labels=b).sample
        per = F.mse_loss(pred, noise, reduction="none").mean((1, 2, 3))
        loss = per.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if ema is not None:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1.0 - args.ema)
        if coarse.any():
            lc = 0.9 * lc + 0.1 * per[coarse].mean().item()
        if (~coarse).any():
            lf = 0.9 * lf + 0.1 * per[~coarse].mean().item()
        if step % 200 == 0:
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} fine={lf:.4f} coarse={lc:.4f} bucket={BUCKETS[int(b[0]) % NB]}", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            net = ema if ema is not None else model
            net.eval()
            for s in EVAL_SIZES:
                torch.manual_seed(args.seed)
                make_grid(sample_cg(net, scheduler, eval_cond, s, device)).save(out / "samples" / f"step_{step:06d}_s{s}.png")
            torch.save(net.state_dict(), out / "model_latest.pt")
            model.train()
        if args.snap_every and step % args.snap_every == 0:
            net = ema if ema is not None else model
            torch.save(net.state_dict(), out / f"model_step{step:06d}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "ema": ema.state_dict() if ema is not None else None}, resume)
    print(f"Done -> {out}", flush=True)


if __name__ == "__main__":
    main()
