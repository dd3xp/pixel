"""Show the samples the provenance probe is least willing to call natively drawn.

The probe says 67.5% of our 16 px samples look authored at that size. The interesting third is the rest:
rather than eyeballing a random sheet and reporting whatever I notice, this uses the classifier as the
selector, so the failures shown are chosen by the instrument and not by me.

Usage: python scripts/probe_failures.py --gen <dir> --out runs_out/probe_failures.png
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "src/v6")
import fd_dino as FD  # noqa: E402
from fd_fair import native_sizes, real_split  # noqa: E402
from train_cond import to_tensor  # noqa: E402

CHECK = ((235, 235, 235), (205, 205, 205))


def render(paths, R, out):
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


def board(path, scale, cell):
    im = Image.open(path).convert("RGBA")
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    bg = Image.new("RGB", im.size)
    px = bg.load()
    for y in range(bg.height):
        for x in range(bg.width):
            px[x, y] = CHECK[((x // 4) + (y // 4)) % 2]
    bg.paste(im, (0, 0), im)
    return bg.resize((cell, cell), Image.NEAREST) if bg.size != (cell, cell) else bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="runs_out/v8n_icg2_pal4_matched_eval/s16")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="runs_out/probe_failures.png")
    a = ap.parse_args()
    R = a.size

    real = sorted({p.replace("\\", "/") for p in
                   glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")})
    sz = native_sizes(real)
    _, held = real_split(native_R=R, pool="old")
    held = {q.replace("\\", "/") for q in held}
    pos = [p for p in real if sz[p] <= R and p not in held][:1200]
    neg = [p for p in real if 25 <= sz[p] <= 64][:1200]
    fp = FD.embed(render(pos, R, f"runs_out/_probe_pos_s{R}"), R)
    fn = FD.embed(render(neg, R, f"runs_out/_probe_neg_s{R}"), R)
    X = np.concatenate([fp, fn])
    y = np.concatenate([np.ones(len(fp)), np.zeros(len(fn))])

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)

    files = sorted(glob.glob(str(Path(a.gen) / "*.png")))
    f = FD.embed(files, R)
    p = clf.predict_proba(sc.transform(f))[:, 1]
    order = np.argsort(p)
    worst, best = order[:a.n], order[-a.n:][::-1]
    print(f"{len(files)} samples, mean p {p.mean():.3f}; worst {p[worst[0]]:.3f}, best {p[best[0]]:.3f}")

    scale, cell, pad, lab = 8, 128, 4, 92
    W = lab + a.n * (cell + pad)
    sheet = Image.new("RGB", (W, 2 * (cell + pad)), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for r, (name, idxs) in enumerate((("least native", worst), ("most native", best))):
        dr.text((4, r * (cell + pad) + cell // 2), name, fill=(0, 0, 0))
        for c, i in enumerate(idxs):
            sheet.paste(board(files[i], scale, cell), (lab + c * (cell + pad), r * (cell + pad)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
