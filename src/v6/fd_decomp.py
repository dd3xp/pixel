"""Decompose the fair FD-DINOv2@R gap: mean vs covariance term, kNN precision/recall
(Kynkaanniemi 2019), density/coverage (Naeem 2020), intra-set diversity, and nearest-real
distance.  Same feature pipeline as fd_fair.py / fd_dino.py (dinov2-small CLS, NEAREST 224).

Usage:
  python src/v6/fd_decomp.py --size 16 --gen runs_out/probe_tv_eval/s16 runs_out/probe_tv_q16/s16
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from v6 import fd_dino as F  # noqa: E402
from fd_fair import real_split, dump_totensor  # noqa: E402


def cdist(a, b):
    a = torch.from_numpy(a).cuda()
    b = torch.from_numpy(b).cuda()
    return torch.cdist(a, b).cpu().numpy()


def knn_radius(x, k):
    d = cdist(x, x)
    return np.sort(d, axis=1)[:, k]  # k-th neighbour excluding self (col 0 is self=0)


def prdc(real, fake, k=5):
    """precision/recall (Kynkaanniemi) + density/coverage (Naeem)."""
    rr = knn_radius(real, k)
    rf = knn_radius(fake, k)
    d_rf = cdist(real, fake)  # (Nr, Nf)
    precision = (d_rf < rr[:, None]).any(0).mean()
    recall = (d_rf < rf[None, :]).any(1).mean()
    density = (1.0 / k) * (d_rf < rr[:, None]).sum(0).mean()
    coverage = (d_rf.min(1) < rr).mean()
    return dict(precision=float(precision), recall=float(recall), density=float(density), coverage=float(coverage))


def report(name, fr, fg, mr, sr):
    mg, sg = F.stats(fg)
    fd = F.frechet(mg, sg, mr, sr)
    mean_term = float(((mg - mr) ** 2).sum())
    out = dict(fd=round(fd, 2), mean_term=round(mean_term, 2), cov_term=round(fd - mean_term, 2),
               tr_cov_gen=round(float(np.trace(sg)), 1), tr_cov_ref=round(float(np.trace(sr)), 1))
    out.update({k: round(v, 3) for k, v in prdc(fr, fg).items()})
    d = cdist(fg, fr)
    out["nn_gen2ref"] = round(float(d.min(1).mean()), 2)  # fidelity-ish
    out["nn_ref2gen"] = round(float(d.min(0).mean()), 2)  # coverage-ish
    dgg = cdist(fg, fg)
    out["intra_div"] = round(float(dgg.sum() / (len(fg) * (len(fg) - 1))), 2)
    print(f"{name:45s} " + " ".join(f"{k}={v}" for k, v in out.items()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--gen", nargs="*", default=[])
    ap.add_argument("--out", default="runs_out/fd_decomp.json")
    args = ap.parse_args()
    R = args.size
    ref, held = real_split()
    ref_files = dump_totensor(ref, f"runs_out/ref_totensor_s{R}", R)
    held_files = dump_totensor(held[:3000], f"runs_out/heldout3000_totensor_s{R}", R)
    fr = F.embed(ref_files, R)
    mr, sr = F.stats(fr)
    res = {}
    drr = cdist(fr, fr)
    print(f"ref intra_div={drr.sum() / (len(fr) * (len(fr) - 1)):.2f}", flush=True)
    res["heldout3000"] = report("heldout3000 (floor)", fr, F.embed(held_files, R), mr, sr)
    for g in args.gen:
        files = sorted(glob.glob(f"{g}/*.png"))
        res[g] = report(g, fr, F.embed(files, R), mr, sr)
    old = json.load(open(args.out)) if Path(args.out).exists() else {}
    old.update(res)
    json.dump(old, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
