"""CRSC gate T1: the T0 measurement on the model's OWN sampler states instead of forward-noised data.

T0 (crsc_t0.py) found w* ~ 0.85-1.07 at 16/20/24 px for every reference, and L(1.5)/L(1) > 1 everywhere: on
forward-noised data, extrapolating away from the lower-resolution belief only increases the denoising error.
So the guidance is not a posterior-error correction there. The remaining explanation (arch_crossres_module.md,
Section 1b) is drift: the deficit may build up along the sampler, on states the training loss never sees.

Procedure (arch_crossres_module.md, T1):
  noise x0 to the timestep k sampler steps BEFORE the target step, run the deployed 100-step DDPM sampler for k
  steps (either bare conditional, w=1, or label-guided, bucket:<lower> w=2), reach x~_t, and measure w* in
  x0-space against the KNOWN x0:
      d  = x0hat_strong - x0hat_ref,   w* = E[d . (x0 - x0hat_ref)] / E[d . d]
  for k in {1, 5, 20}. (The same ratio holds in eps-space with eps~ = (x~_t - sqrt(ab) x0)/sqrt(1-ab).)

Decision: w* >= 1.3 for k >= 5 while T0 < 1.1 -> the guidance is a drift corrector; CRSC must train on rolled
states (SOAR recipe, with SOAR-alone as the control). Both T0 and T1 < 1.1 -> no data-trained module will
internalise the guidance; drop CRSC as the headline.

Usage: python src/v6/crsc_t1.py --size 16 --lower 12 --n 2000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, "src/v6")
from crsc_t0 import load_set, load_unet
from train_v7 import BUCKETS, embed


@torch.no_grad()
def run(models, x0_all, caps, size, lower, idxs, ks, rollers, bs, dev, tok, txt):
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    sched.set_timesteps(100)
    ts = sched.timesteps.to(dev)                       # descending, ts[0] ~ 990
    ab_all = sched.alphas_cumprod.to(dev)
    lab_t, lab_l = BUCKETS.index(size), BUCKETS.index(lower)
    final, snap = models["final"], models["snap"]
    out = {}
    for roll in rollers:
        for k in ks:
            for i in idxs:
                if i - k < 0:
                    continue
                acc = {r: {"num": [], "den": [], "l1": [], "l15": [], "l2": []} for r in ("label", "snapshot", "composed")}
                g = torch.Generator(device=dev).manual_seed(7000 + 31 * i + k)
                for b0 in range(0, len(x0_all), bs):
                    x0 = x0_all[b0:b0 + bs].to(dev)
                    B = x0.shape[0]
                    cond = embed(caps[b0:b0 + bs], tok, txt, dev)
                    lt = torch.full((B,), lab_t, device=dev, dtype=torch.long)
                    ll = torch.full((B,), lab_l, device=dev, dtype=torch.long)
                    eps = torch.randn(x0.shape, generator=g, device=dev)
                    x = sched.add_noise(x0, eps, ts[i - k].repeat(B))
                    for j in range(i - k, i):              # k deployed sampler steps
                        t = ts[j]
                        e_c = final(x, t, encoder_hidden_states=cond, class_labels=lt).sample
                        if roll == "guided":
                            e_r = final(x, t, encoder_hidden_states=cond, class_labels=ll).sample
                            e_c = e_r + 2.0 * (e_c - e_r)
                        # DDPMScheduler.step draws its own noise; pass our generator for reproducibility
                        x = sched.step(e_c, t, x, generator=g).prev_sample
                    t = ts[i]
                    ab = ab_all[t]
                    to_x0 = lambda e: (x - (1 - ab).sqrt() * e) / ab.sqrt()
                    xb = to_x0(final(x, t, encoder_hidden_states=cond, class_labels=lt).sample)
                    refs = {
                        "label": to_x0(final(x, t, encoder_hidden_states=cond, class_labels=ll).sample),
                        "snapshot": to_x0(snap(x, t, encoder_hidden_states=cond, class_labels=lt).sample),
                        "composed": to_x0(snap(x, t, encoder_hidden_states=cond, class_labels=ll).sample),
                    }
                    for r, xr in refs.items():
                        d = (xb - xr).flatten(1)
                        er = (x0 - xr).flatten(1)
                        acc[r]["num"].append((d * er).sum(1).cpu())
                        acc[r]["den"].append((d * d).sum(1).cpu())
                        for w, key in ((1.0, "l1"), (1.5, "l15"), (2.0, "l2")):
                            acc[r][key].append(((xr + w * (xb - xr)) - x0).flatten(1).pow(2).mean(1).cpu())
                res = {}
                rng = np.random.default_rng(0)
                for r, a in acc.items():
                    num = torch.cat(a["num"]).numpy(); den = torch.cat(a["den"]).numpy()
                    boots = []
                    for _ in range(200):
                        ix = rng.integers(0, len(num), len(num))
                        boots.append(num[ix].sum() / max(den[ix].sum(), 1e-12))
                    l1 = torch.cat(a["l1"]).numpy().mean()
                    res[r] = dict(w_star=float(num.sum() / max(den.sum(), 1e-12)),
                                  ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                  rel_loss_w1p5=float(torch.cat(a["l15"]).numpy().mean() / l1),
                                  rel_loss_w2=float(torch.cat(a["l2"]).numpy().mean() / l1))
                key = f"{roll}_k{k}_i{i}"
                out[key] = dict(roll=roll, k=k, i=i, t=int(ts[i]), **res)
                print(f"{roll:7s} k={k:2d} t={int(ts[i]):4d} " + "  ".join(
                    f"{r}: w*={v['w_star']:.3f} [{v['ci'][0]:.2f},{v['ci'][1]:.2f}] L1.5/L1={v['rel_loss_w1p5']:.3f}"
                    for r, v in res.items()), flush=True)
    return out


def decide(res):
    """Drift corrector if w* >= 1.3 for k >= 5 (any reference, any roller) on at least 30% of its points."""
    verdict = {}
    for r in ("label", "snapshot", "composed"):
        pts = [v[r]["w_star"] for v in res.values() if v["k"] >= 5]
        verdict[r] = dict(frac_ge_1p3=float(np.mean([w >= 1.3 for w in pts])) if pts else 0.0,
                          max_w=float(max(pts)) if pts else float("nan"),
                          all_below_1p1=bool(all(w < 1.1 for w in pts)))
    go = any(v["frac_ge_1p3"] >= 0.30 for v in verdict.values())
    stop = all(v["all_below_1p1"] for v in verdict.values())
    return verdict, ("GO: drift corrector -> CRSC on rolled states (SOAR recipe, SOAR-alone control)" if go else
                     "STOP: T0 and T1 both < 1.1 -> drop CRSC as the headline" if stop else
                     "AMBIGUOUS: w* between 1.1 and 1.3 on rolled states")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--lower", type=int, default=12)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--idxs", type=int, nargs="+", default=[25, 45, 65, 85, 95],
                    help="target positions in the 100-step schedule (0 = noisiest)")
    ap.add_argument("--rollers", nargs="+", default=["bare", "guided"])
    ap.add_argument("--ckpt", default="workdir/v7h/model_latest.pt")
    ap.add_argument("--snap", default="workdir/v7h/model_step010000.pt")
    ap.add_argument("--out", default="runs_out/crsc_t1.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    txt = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    models = {"final": load_unet(a.ckpt, dev), "snap": load_unet(a.snap, dev)}
    x0, caps = load_set("heldout", a.size, a.n)
    print(f"=== T1 held-out: {len(x0)} sprites at {a.size}px, reference bucket {a.lower} ===", flush=True)
    res = run(models, x0, caps, a.size, a.lower, a.idxs, a.ks, a.rollers, a.bs, dev, tok, txt)
    verdict, call = decide(res)
    print(f"--- T1 DECISION: {call}", flush=True)
    for r, v in verdict.items():
        print(f"    {r}: frac(w*>=1.3 | k>=5)={v['frac_ge_1p3']:.2f}  max w*={v['max_w']:.3f}  all<1.1={v['all_below_1p1']}")
    Path(a.out).write_text(json.dumps(dict(size=a.size, lower=a.lower, per_point=res, verdict=verdict, decision=call),
                                      indent=2))
    print("CRSC_T1_DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
