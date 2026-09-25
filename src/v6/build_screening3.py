"""Choose screening pairs with the provenance probe instead of a hand-picked pixel statistic.

Two selection rules have been tried for the control pairs.

The first (build_screening.py) drew random pairs of real sprites, one authored at 16 px and one larger
picture reduced to it. Eleven raters scored 3.36 of 6 on it against a chance of 3, and on five of the
twenty pairs every rater answered against the label.

The second (build_screening2.py) required the native side to be crisp and the shrunk side soft by the
paper's own flat-neighbour and palette statistics. It was checked against those eleven raters before
being used and it failed: 58% correct on the pairs it keeps against 55% on the pairs it drops. The
clearest counter-example is screen00, a perfect pair by the statistic -- 0.60 flat and 6 colours against
0.00 flat and 93 colours -- that every rater got wrong, because the crisp side is a fragment of grey
rubble and the soft side is a glowing blue burst, and a glow reads as deliberate at any size.

So the property that decides a pair is not how flat the native side is; it is whether the shrunk side
visibly shows detail that dissolved. That is what the provenance probe of native_probe.py is trained on
-- DINOv2-small CLS features, logistic regression, ground truth a fact about the file -- so this selects
pairs the probe gets right with margin, on the theory that a pair carrying evidence a trained classifier
can find is a pair carrying evidence a person can find.

That theory is not assumed. --report scores the first twenty pairs and prints probe margin beside the
raters' accuracy on each, so the rule is checked the same way the last one was, and is only worth
shipping if margin predicts accuracy.

Usage (GPU box; PYTHONPATH must carry the private huggingface_hub overlay):
  python src/v6/build_screening3.py --report runs/study_web/replies.json
  python src/v6/build_screening3.py --n 20 --out runs_out/human_study4
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import fd_dino as FD  # noqa: E402
from sprite_stats import at_size, native_sizes, real_split  # noqa: E402

CHECK = ((235, 235, 235), (205, 205, 205))


def at_R(path, R):
    return at_size(path, R)


def board(im, scale=16, cell=256):
    im = im.convert("RGBA")
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    bg = Image.new("RGB", im.size)
    px = bg.load()
    for y in range(bg.height):
        for x in range(bg.width):
            px[x, y] = CHECK[((x // 8) + (y // 8)) % 2]
    bg.paste(im, (0, 0), im)
    return bg.resize((cell, cell), Image.NEAREST) if bg.size != (cell, cell) else bg


def render_all(paths, R, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for i, p in enumerate(paths):
        f = out / f"{i:05d}.png"
        if not f.exists():
            at_R(p, R).save(f)
        files.append(str(f))
    return files


def train_probe(R, n_per_class=1200):
    """The probe of native_probe.py, trained here so candidates can be scored in the same call."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    real = sorted({p.replace("\\", "/") for p in
                   glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")})
    sz = native_sizes(real)
    _, held = real_split(native_R=R)
    held = {q.replace("\\", "/") for q in held}
    pos = [p for p in real if sz[p] <= R and p not in held]
    neg = [p for p in real if 25 <= sz[p] <= 64]
    rng = random.Random(0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n = min(n_per_class, len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]
    print(f"probe training on {n} native + {n} shrunk (held half excluded)", flush=True)

    fp = FD.embed(render_all(pos, R, f"runs_out/_probe_pos_s{R}"), R)
    fn = FD.embed(render_all(neg, R, f"runs_out/_probe_neg_s{R}"), R)
    X = np.concatenate([fp, fn])
    y = np.concatenate([np.ones(len(fp)), np.zeros(len(fn))])
    idx = np.arange(len(X))
    np.random.default_rng(0).shuffle(idx)
    cut = int(0.75 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
    acc = float(clf.score(sc.transform(X[te]), y[te]))
    print(f"probe held-out accuracy {acc:.3f}", flush=True)
    if acc < 0.8:
        raise SystemExit(f"probe too weak ({acc:.3f}) to select anything")
    return sc, clf, acc, pos, neg, real, sz, held


def prob_native(sc, clf, paths, R, cache):
    f = FD.embed(render_all(paths, R, cache), R)
    return clf.predict_proba(sc.transform(f))[:, 1]


def report(replies, R, sc, clf, seed=7, n=20):
    """Probe margin on the first twenty pairs, beside how the eleven raters did on each."""
    real = sorted({p.replace("\\", "/") for p in
                   glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")})
    sz = native_sizes(real)
    native = [p for p in real if sz[p] <= R]
    shrunk = [p for p in real if 25 <= sz[p] <= 64]
    rng = random.Random(seed)
    rng.shuffle(native)
    rng.shuffle(shrunk)
    pn, ps = native[:n], shrunk[:n]

    p_nat = prob_native(sc, clf, pn, R, f"runs_out/_rep_nat_s{R}")
    p_shr = prob_native(sc, clf, ps, R, f"runs_out/_rep_shr_s{R}")

    got = {}
    for p in json.loads(Path(replies).read_text(encoding="utf-8"))["raters"].values():
        for a in p["answers"]:
            if a["kind"] == "screen":
                w, t = got.get(a["id"], (0, 0))
                got[a["id"]] = (w + bool(a["correct"]), t + 1)

    rows = []
    print(f"{'item':9s} {'raters':>9s}  {'p(native)':>9s} {'p(shrunk)':>9s} {'margin':>7s}  probe")
    for i in range(n):
        m = float(p_nat[i] - p_shr[i])          # positive = probe separates the pair the right way
        ok = p_nat[i] > 0.5 and p_shr[i] < 0.5
        w, t = got.get(f"screen{i:02d}", (0, 0))
        if t:
            rows.append((m, w / t, w, t))
        print(f"screen{i:02d}  {w:3d}/{t:<5d} {p_nat[i]:9.3f} {p_shr[i]:9.3f} {m:7.3f}  "
              f"{'right' if ok else 'WRONG'}")
    if len(rows) > 2:
        ms = np.array([r[0] for r in rows])
        ac = np.array([r[1] for r in rows])
        r = float(np.corrcoef(ms, ac)[0, 1])
        hi = [x for x in rows if x[0] > 0]
        lo = [x for x in rows if x[0] <= 0]
        hw, hn = sum(x[2] for x in hi), sum(x[3] for x in hi)
        lw, ln = sum(x[2] for x in lo), sum(x[3] for x in lo)
        print(f"\ncorrelation(probe margin, rater accuracy) = {r:+.3f} over {len(rows)} pairs")
        print(f"raters on pairs with positive margin: {hw}/{hn}"
              f"{f' = {100*hw/hn:.0f}%' if hn else ''}")
        print(f"raters on pairs with negative margin: {lw}/{ln}"
              f"{f' = {100*lw/ln:.0f}%' if ln else ''}   (chance = 50%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--margin", type=float, default=0.8, help="p(native side) - p(shrunk side)")
    ap.add_argument("--pool", type=int, default=600, help="candidates scored per class")
    ap.add_argument("--out", default="runs_out/human_study4")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    R = a.size
    sc, clf, acc, pos, neg, real, sz, held = train_probe(R)
    if a.report:
        return report(a.report, R, sc, clf)

    rng = random.Random(a.seed)
    cand_n = [p for p in real if sz[p] <= R]
    cand_s = [p for p in real if 25 <= sz[p] <= 64]
    rng.shuffle(cand_n)
    rng.shuffle(cand_s)
    cand_n, cand_s = cand_n[:a.pool], cand_s[:a.pool]
    pn = prob_native(sc, clf, cand_n, R, f"runs_out/_cand_nat_s{R}")
    psh = prob_native(sc, clf, cand_s, R, f"runs_out/_cand_shr_s{R}")
    good_n = [c for c, p in sorted(zip(cand_n, pn), key=lambda kv: -kv[1]) if p > 0.5]
    good_s = [c for c, p in sorted(zip(cand_s, psh), key=lambda kv: kv[1]) if p < 0.5]
    pmap = dict(zip(cand_n, pn))
    smap = dict(zip(cand_s, psh))
    print(f"{len(good_n)}/{len(cand_n)} native and {len(good_s)}/{len(cand_s)} shrunk candidates "
          f"the probe calls correctly", flush=True)

    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    pairs = []
    i = j = k = 0
    while len(pairs) < a.n and i < len(good_n) and j < len(good_s):
        p, q = good_n[i], good_s[j]
        if pmap[p] - smap[q] < a.margin:
            i += 1
            continue
        board(at_R(p, R)).save(out / "img" / f"screen{k:02d}_native.png")
        board(at_R(q, R)).save(out / "img" / f"screen{k:02d}_shrunk.png")
        left_is_native = rng.random() < 0.5
        pairs.append({"id": f"screen{k:02d}",
                      "left_img": f"img/screen{k:02d}_{'native' if left_is_native else 'shrunk'}.png",
                      "right_img": f"img/screen{k:02d}_{'shrunk' if left_is_native else 'native'}.png",
                      "answer": "left" if left_is_native else "right",
                      "native_src": p, "shrunk_src": q,
                      "p_native": float(pmap[p]), "p_shrunk": float(smap[q])})
        i += 1
        j += 1
        k += 1
    if len(pairs) < a.n:
        raise SystemExit(f"only {len(pairs)} pairs cleared margin {a.margin}")
    (out / "screening.json").write_text(json.dumps(
        {"size": R, "seed": a.seed, "selector": "native_probe", "probe_accuracy": acc,
         "margin": a.margin, "pairs": pairs}, indent=1), encoding="utf-8")
    print(f"wrote {out}/screening.json with {len(pairs)} pairs", flush=True)


if __name__ == "__main__":
    main()
