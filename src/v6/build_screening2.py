"""Screening pairs whose two sides actually differ in the property the question asks about.

The first screening set (build_screening.py) drew twenty random pairs of real sprites, one authored at
16 px and one larger picture reduced to it. Eleven raters scored 3.36 of 6 on it, against a chance of
3, and the failure was not noise: on five of the twenty pairs *every* rater answered against the label
and on nine *every* rater answered with it. The labels were checked and are correct. Looking at the
images says why -- a good share of the natively small pool is soft-edged terrain fragments, drawn with
anti-aliasing and gradients, so the pair differed in provenance but not in anything visible.

A screening item is only a control if a rater who knows what to look for can pass it. So the sides are
now required to sit on opposite sides of the medium's own statistic rather than merely to come from
different pools:

    native side   flat >= 0.45, colours <= 16     (the paper's real-sprite average is 0.46 flat)
    shrunk side   flat <= 0.20, colours >= 32     (BOX averaging invents a colour per pixel)
    both sides    0.15 <= coverage <= 0.90        (neither is near-empty nor a full-bleed block)

These thresholds come from the medium's statistics as reported in the paper, not from which items the
eleven raters happened to pass; --report prints the old twenty scored against the rule so the two can
be compared. Provenance is still the ground truth: the rule decides which pairs are *asked*, never
what the right answer is.

Usage: python src/v6/build_screening2.py --n 20 --out runs_out/human_study3 --report runs/study_web/replies.json
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from sprite_stats import at_size, native_sizes, stats  # noqa: E402

CHECK = ((235, 235, 235), (205, 205, 205))
NATIVE_OK = {"flat_min": 0.45, "colours_max": 16}
SHRUNK_OK = {"flat_max": 0.20, "colours_min": 32}
COVER = (0.15, 0.90)


def keep_native(s):
    return (s["flat"] >= NATIVE_OK["flat_min"] and s["colours"] <= NATIVE_OK["colours_max"]
            and COVER[0] <= s["coverage"] <= COVER[1])


def keep_shrunk(s):
    return (s["flat"] <= SHRUNK_OK["flat_max"] and s["colours"] >= SHRUNK_OK["colours_min"]
            and COVER[0] <= s["coverage"] <= COVER[1])


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


def report_old(replies, R, seed=7, n=20):
    """Score the first screening set against the rule, beside how the raters actually did on it."""
    real = sorted({p.replace("\\", "/") for p in
                   glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")})
    sz = native_sizes(real)
    native = [p for p in real if sz[p] <= R]
    shrunk = [p for p in real if 25 <= sz[p] <= 64]
    rng = random.Random(seed)
    rng.shuffle(native)
    rng.shuffle(shrunk)

    got = {}
    for p in json.loads(Path(replies).read_text(encoding="utf-8"))["raters"].values():
        for a in p["answers"]:
            if a["kind"] == "screen":
                w, t = got.get(a["id"], (0, 0))
                got[a["id"]] = (w + bool(a["correct"]), t + 1)

    print(f"{'item':9s} {'raters':>8s}  {'kept':>5s}   native(flat/col/cov)      shrunk(flat/col/cov)")
    kept_w = kept_n = drop_w = drop_n = 0
    for i in range(n):
        sn, ss = stats(at_size(native[i], R)), stats(at_size(shrunk[i], R))
        ok = keep_native(sn) and keep_shrunk(ss)
        w, t = got.get(f"screen{i:02d}", (0, 0))
        if t:
            if ok:
                kept_w += w; kept_n += t
            else:
                drop_w += w; drop_n += t
        print(f"screen{i:02d}  {w:3d}/{t:<4d} {'keep' if ok else 'drop':>5s}   "
              f"{sn['flat']:.2f} {sn['colours']:3d} {sn['coverage']:.2f}          "
              f"{ss['flat']:.2f} {ss['colours']:3d} {ss['coverage']:.2f}")
    print(f"\nrater accuracy on pairs the rule keeps: {kept_w}/{kept_n}"
          f"{f' = {100*kept_w/kept_n:.0f}%' if kept_n else ''}")
    print(f"rater accuracy on pairs the rule drops: {drop_w}/{drop_n}"
          f"{f' = {100*drop_w/drop_n:.0f}%' if drop_n else ''}   (chance = 50%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--out", default="runs_out/human_study3")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--report", default=None, help="score the first screening set against the rule and exit")
    a = ap.parse_args()
    R = a.size
    if a.report:
        return report_old(a.report, R)

    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    real = sorted({p.replace("\\", "/") for p in
                   glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")})
    sz = native_sizes(real)
    pool_n = [p for p in real if sz[p] <= R]
    pool_s = [p for p in real if 25 <= sz[p] <= 64]

    native = [p for p in pool_n if keep_native(stats(at_size(p, R)))]
    shrunk = [p for p in pool_s if keep_shrunk(stats(at_size(p, R)))]
    print(f"native  {len(native):5d} of {len(pool_n):5d} pass the crisp rule  ({100*len(native)/max(1,len(pool_n)):.1f}%)")
    print(f"shrunk  {len(shrunk):5d} of {len(pool_s):5d} pass the soft rule   ({100*len(shrunk)/max(1,len(pool_s)):.1f}%)")
    if min(len(native), len(shrunk)) < a.n:
        raise SystemExit(f"only {min(len(native), len(shrunk))} candidates, need {a.n}")

    rng = random.Random(a.seed)
    rng.shuffle(native)
    rng.shuffle(shrunk)
    pairs = []
    for i in range(a.n):
        pn, ps = native[i], shrunk[i]
        board(at_size(pn, R)).save(out / "img" / f"screen{i:02d}_native.png")
        board(at_size(ps, R)).save(out / "img" / f"screen{i:02d}_shrunk.png")
        left_is_native = rng.random() < 0.5
        pairs.append({
            "id": f"screen{i:02d}",
            "left_img": f"img/screen{i:02d}_{'native' if left_is_native else 'shrunk'}.png",
            "right_img": f"img/screen{i:02d}_{'shrunk' if left_is_native else 'native'}.png",
            "answer": "left" if left_is_native else "right",
            "native_src": pn, "shrunk_src": ps,
        })
    (out / "screening.json").write_text(json.dumps(
        {"size": R, "seed": a.seed, "rule": {"native": NATIVE_OK, "shrunk": SHRUNK_OK, "coverage": COVER},
         "pairs": pairs}, indent=1), encoding="utf-8")
    print(f"wrote {out}/screening.json with {len(pairs)} pairs")


if __name__ == "__main__":
    main()
