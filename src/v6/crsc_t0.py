"""CRSC gate T0: can a trained module learn the cross-resolution guidance at all?

The key fact (arch_crossres_module.md, Section 1): a module trained with the ordinary denoising loss converges to
the posterior mean mu. The lower-bucket belief r is computed from the same noisy input, so at that optimum it adds
no information. A module can only learn to extrapolate away from r as far as the FINITE network's residual
(mu - e_b) is predictable from d = e_b - r. The linear version of that is directly measurable, with no training:

    w*(t) = E[ d . (eps - r) ] / E[ d . d ]          (expectation over held-out (x0, eps) at fixed t)

It is unbiased for E[d . (mu - r)] / E||d||^2 because eps - mu has zero mean given x_t.
  w* ~ 1   -> the error is not linearly visible in the lower-bucket belief; no MSE-trained module will learn the
              guidance, whatever its architecture.
  w* >> 1  -> the network's own error is partly readable from its counterfactual belief; CRSC has a signal.

Decision rule (from the report): if w*(t) >= 1.3 over at least 30% of the timesteps for any reference, train CRSC
on forward-noised states. If w* < 1.1 everywhere, the guidance is not a posterior-error correction on the data
distribution, and T1 (self-rolled states) decides between the drift-corrector route and dropping CRSC.

Three references, all queried on the SAME x_t and caption:
  label     : the final model at the LOWER bucket label          (our label self-guidance)
  snapshot  : the early EMA snapshot at the TARGET label         (autoguidance)
  composed  : the early EMA snapshot at the LOWER bucket label   (our composed guidance)

Also reports the relative loss L(w)/L(1) at w = 1.5 and 2, where the guided prediction is r + w (e_b - r).
Run on held-out sprites and, separately, on training sprites, to separate generalisation from approximation.

Usage: python src/v6/crsc_t0.py --size 16 --lower 12 --n 3000
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, "src/v6")
from train_v7 import BUCKETS, embed, to_tensor

CFG = dict(sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
           block_out_channels=(128, 256, 512), cross_attention_dim=512,
           down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
           up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
           num_class_embeds=len(BUCKETS))


def load_unet(path, dev):
    m = UNet2DConditionModel(**CFG).to(dev).eval().requires_grad_(False)
    m.load_state_dict(torch.load(path, map_location=dev))
    return m


def load_set(kind, size, n):
    """held-out: the 3000 reference sprites and their captions, exactly as the FD protocol uses them.
    train: sprites the model was trained on, with their (recaptioned-or-not) training captions."""
    if kind == "heldout":
        fs = sorted(glob.glob(f"runs_out/heldout3000_totensor_s{size}/*.png"))[:n]
        caps = [l.strip() for l in open("runs_out/heldout3000_prompts.txt", encoding="utf-8") if l.strip()][:n]
        xs = [to_tensor(Image.open(f).convert("RGBA"), size) for f in fs]
        return torch.stack(xs), caps[:len(xs)]
    import csv
    rows = list(csv.DictReader(open("data/oga_captions.csv", encoding="utf-8")))
    ex = {l.strip() for l in open("runs_out/holdout_exclude.txt") if l.strip()}
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    xs, caps = [], []
    for r in rows:
        p = "data/oga_clean/" + r["path"]
        if p in ex or not Path(p).exists():
            continue
        im = Image.open(p).convert("RGBA")
        if max(im.size) < size:
            continue
        xs.append(to_tensor(im, size)); caps.append(r["text"])
        if len(xs) >= n:
            break
    return torch.stack(xs), caps


@torch.no_grad()
def run(models, x0_all, caps, size, lower, tsteps, bs, dev, tok, txt):
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    lab_t = BUCKETS.index(size)
    lab_l = BUCKETS.index(lower)
    out = {}
    for t_int in tsteps:
        acc = {k: {"num": [], "den": [], "l1": [], "l15": [], "l2": []} for k in ("label", "snapshot", "composed")}
        g = torch.Generator(device=dev).manual_seed(1000 + t_int)
        for i in range(0, len(x0_all), bs):
            x0 = x0_all[i:i + bs].to(dev)
            B = x0.shape[0]
            cond = embed(caps[i:i + bs], tok, txt, dev)
            eps = torch.randn(x0.shape, generator=g, device=dev)
            t = torch.full((B,), t_int, device=dev, dtype=torch.long)
            xt = sched.add_noise(x0, eps, t)
            lt = torch.full((B,), lab_t, device=dev, dtype=torch.long)
            ll = torch.full((B,), lab_l, device=dev, dtype=torch.long)
            e_b = models["final"](xt, t, encoder_hidden_states=cond, class_labels=lt).sample
            refs = {
                "label": models["final"](xt, t, encoder_hidden_states=cond, class_labels=ll).sample,
                "snapshot": models["snap"](xt, t, encoder_hidden_states=cond, class_labels=lt).sample,
                "composed": models["snap"](xt, t, encoder_hidden_states=cond, class_labels=ll).sample,
            }
            for k, r in refs.items():
                d = (e_b - r).flatten(1)
                er = (eps - r).flatten(1)
                acc[k]["num"].append((d * er).sum(1).cpu())
                acc[k]["den"].append((d * d).sum(1).cpu())
                for w, key in ((1.0, "l1"), (1.5, "l15"), (2.0, "l2")):
                    gd = (r + w * (e_b - r)) - eps
                    acc[k][key].append(gd.flatten(1).pow(2).mean(1).cpu())
        res = {}
        rng = np.random.default_rng(0)
        for k, a in acc.items():
            num = torch.cat(a["num"]).numpy(); den = torch.cat(a["den"]).numpy()
            w = num.sum() / max(den.sum(), 1e-12)
            boots = []
            for _ in range(200):
                idx = rng.integers(0, len(num), len(num))
                boots.append(num[idx].sum() / max(den[idx].sum(), 1e-12))
            l1 = torch.cat(a["l1"]).numpy().mean()
            res[k] = dict(w_star=float(w), ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                          rel_loss_w1p5=float(torch.cat(a["l15"]).numpy().mean() / l1),
                          rel_loss_w2=float(torch.cat(a["l2"]).numpy().mean() / l1))
        out[int(t_int)] = res
        print(f"t={t_int:4d} " + "  ".join(
            f"{k}: w*={v['w_star']:.3f} [{v['ci'][0]:.2f},{v['ci'][1]:.2f}] L1.5/L1={v['rel_loss_w1p5']:.3f}"
            for k, v in res.items()), flush=True)
    return out


def decide(res):
    """w* >= 1.3 over at least 30% of timesteps, for any reference."""
    verdict = {}
    for k in ("label", "snapshot", "composed"):
        ws = [res[t][k]["w_star"] for t in res]
        frac13 = float(np.mean([w >= 1.3 for w in ws]))
        verdict[k] = dict(frac_ge_1p3=frac13, max_w=float(max(ws)), all_below_1p1=bool(all(w < 1.1 for w in ws)))
    go = any(v["frac_ge_1p3"] >= 0.30 for v in verdict.values())
    stop = all(v["all_below_1p1"] for v in verdict.values())
    return verdict, ("GO: train CRSC on forward-noised states" if go else
                     "STOP on data distribution: run T1 (self-rolled states) to decide" if stop else
                     "AMBIGUOUS: w* between 1.1 and 1.3; run T1 before deciding")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--lower", type=int, default=12)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--ckpt", default="workdir/v7h/model_latest.pt")
    ap.add_argument("--snap", default="workdir/v7h/model_step010000.pt")
    ap.add_argument("--out", default="runs_out/crsc_t0.json")
    ap.add_argument("--device", default="cuda", help="'cpu' for a smoke test that does not touch the GPUs")
    ap.add_argument("--n_t", type=int, default=10, help="number of timesteps (10 in the real gate)")
    a = ap.parse_args()
    dev = a.device
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    txt = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    models = {"final": load_unet(a.ckpt, dev), "snap": load_unet(a.snap, dev)}
    tsteps = [int(round(v)) for v in np.linspace(10, 990, a.n_t)]
    report = {"size": a.size, "lower": a.lower, "timesteps": tsteps}
    for kind in ("heldout", "train"):
        x0, caps = load_set(kind, a.size, a.n)
        print(f"=== {kind}: {len(x0)} sprites at {a.size}px, reference bucket {a.lower} ===", flush=True)
        res = run(models, x0, caps, a.size, a.lower, tsteps, a.bs, dev, tok, txt)
        verdict, call = decide(res)
        report[kind] = dict(per_t=res, verdict=verdict, decision=call)
        print(f"--- {kind} DECISION: {call}", flush=True)
        for k, v in verdict.items():
            print(f"    {k}: frac(w*>=1.3)={v['frac_ge_1p3']:.2f}  max w*={v['max_w']:.3f}  all<1.1={v['all_below_1p1']}")
    Path(a.out).write_text(json.dumps(report, indent=2))
    print("CRSC_T0_DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
