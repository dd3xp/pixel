"""A third instrument: can a classifier tell a sprite drawn at 16 px from artwork shrunk to 16 px?

FD against a reference we defined is one measurement, and the vision--language judge we tried failed its
own control. This is an attempt at something objective: the two populations are separated by a fact
about the file, not by taste --- a sprite either was authored at 16 px or it was not --- so a classifier
trained on that distinction has a ground truth, and its accuracy on held-out real sprites says whether
it works before we ask it anything about generated samples.

Features are the DINOv2-small CLS used by the FD, so this is not a new encoder, only a new question put
to it: instead of "how far is this distribution from that one", "what fraction of these images does a
trained classifier call natively drawn".

Usage:
  python src/v6/native_probe.py --dirs ours=<dir> gpt=<dir> ... --out runs_out/native_probe.json
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fd_dino as FD  # noqa: E402
from fd_fair import native_sizes  # noqa: E402
from train_cond import to_tensor  # noqa: E402


def render(paths, R, out):
    """Push real sprites through the same to_tensor the training pipeline uses, so a downscaled
    positive and a downscaled negative differ only in provenance."""
    from PIL import Image
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    have = sorted(out.glob("*.png"))
    if len(have) == len(paths):
        return [str(p) for p in have]
    for i, p in enumerate(paths):
        t = to_tensor(Image.open(p).convert("RGBA"), R)
        a = ((t + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        Image.fromarray(a, "RGBA").save(out / f"{i:05d}.png")
    return [str(p) for p in sorted(out.glob("*.png"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--dirs", nargs="*", default=[], help="name=path of generated samples to score")
    ap.add_argument("--n", type=int, default=1200, help="per class, for training plus test")
    ap.add_argument("--out", default="runs_out/native_probe.json")
    a = ap.parse_args()
    R = a.size

    real = sorted({p.replace("\\", "/") for p in
                   glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")})
    sz = native_sizes(real)
    # the probe is scored on runs_out/held_native{R}, the held half of the native pool, so that half has
    # to be kept out of its training set -- otherwise the control it passes is a control it has seen
    from fd_fair import real_split
    _, held = real_split(native_R=R, pool="old")
    held = {q.replace("\\", "/") for q in held}
    pos = [p for p in real if sz[p] <= R and p not in held]
    neg = [p for p in real if 25 <= sz[p] <= 64]
    print(f"native pool minus the held half: {len(pos)}", flush=True)
    rng = random.Random(0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n = min(a.n, len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]
    print(f"{n} natively <= {R}px, {n} artwork 25-64px", flush=True)

    fp = FD.embed(render(pos, R, f"runs_out/_probe_pos_s{R}"), R)
    fn = FD.embed(render(neg, R, f"runs_out/_probe_neg_s{R}"), R)
    X = np.concatenate([fp, fn])
    y = np.concatenate([np.ones(len(fp)), np.zeros(len(fn))])
    idx = np.arange(len(X))
    np.random.default_rng(0).shuffle(idx)
    cut = int(0.75 * len(idx))
    tr, te = idx[:cut], idx[cut:]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
    acc = float(clf.score(sc.transform(X[te]), y[te]))
    base = float(max(y[te].mean(), 1 - y[te].mean()))
    print(f"held-out accuracy {acc:.3f} (majority class {base:.3f})", flush=True)

    res = {"size": R, "n_per_class": n, "held_out_accuracy": acc, "majority_class": base, "sets": {}}
    if acc < 0.8:
        print("probe is too weak to ask anything of; reporting accuracy only", flush=True)
    else:
        for spec in a.dirs:
            name, d = spec.split("=", 1)
            files = sorted(glob.glob(str(Path(d) / "*.png")))
            if not files:
                print(f"{name}: no images at {d}", flush=True)
                continue
            f = FD.embed(files, R)
            p = clf.predict_proba(sc.transform(f))[:, 1]
            res["sets"][name] = {"n": len(files), "frac_called_native": float((p > 0.5).mean()),
                                 "mean_prob": float(p.mean())}
            print(f"{name:26s} n={len(files):5d}  called native {(p > 0.5).mean():6.1%}  "
                  f"mean p {p.mean():.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
