"""Verify a new sprite bundle: clean, dedup against everything we already have, report, contact sheet.

Why (09-20): the 09-16 rebuild cut three OGA licence bundles (CC-BY-4.0, CC-BY-SA-3.0, OGA-BY-3.0) but not
the CC0 one, which is both the largest 2D bundle and the cleanest licence for a released corpus.  This is
the same pipeline as build_corpus_v8.py, with two differences: the "already have" set now includes the v8
corpus and the Kenney sprites, and nothing about the old corpus is rewritten -- the CC0 material is purely
additive, so data/corpus_v8_dupdrop.txt is left alone.

Outputs: data/oga4_clean/*.png, data/oga4_manifest.csv (path,src,zip,member,licence,side,palette,flat),
runs_out/corpus_v9_report.json (the verification checklist: native size distribution, palette and hard-edge
statistics, cross-corpus duplicate count, licence coverage), runs_out/corpus_v9_sheet.png (random sample of
the natively tiny survivors, for eyeballing).

Usage: python src/v6/build_corpus_v9.py [--new dir:licence ...] [--out DIR] [--tag NAME] [--limit N]
"""
import argparse
import csv
import glob
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_NEW = ["data/oga4_cut_cc0:CC0-1.0"]
OLD = ["data/oga_clean", "data/extra_all", "data/oga3_clean", "data/kenney_cut", "data/oga4_clean"]


def pixhash(a):
    return hashlib.md5(a.tobytes()).hexdigest() + f"_{a.shape[0]}x{a.shape[1]}"


def keep(a):
    """clean_cut.py's filters, applied to an RGBA array."""
    h, w = a.shape[:2]
    if min(h, w) < 10 or max(h, w) / min(h, w) > 3:
        return False
    fill = (a[:, :, 3] > 16).mean()
    if not (0.15 <= fill <= 0.97):
        return False
    vis = a[a[:, :, 3] > 16][:, :3]
    if len(vis) == 0:
        return False
    q = vis // 32
    if len(np.unique(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2])) < 4:
        return False
    return vis.std() >= 18


def palette(a):
    vis = a[a[:, :, 3] > 0][:, :3]
    return len(np.unique(vis.reshape(-1, 3), axis=0)) if len(vis) else 0


def flat_ratio(a):
    """Fraction of opaque 4-neighbour pairs with identical RGB: the hard-edge / flat-region statistic."""
    op = a[:, :, 3] >= 128
    rgb = a[:, :, :3].astype(np.int16)
    num = den = 0
    for s in ((slice(None, -1), slice(None)), (slice(None), slice(None, -1))):
        s2 = (slice(1, None), slice(None)) if s[0] != slice(None) else (slice(None), slice(1, None))
        m = op[s] & op[s2]
        if not m.any():
            continue
        d = np.abs(rgb[s] - rgb[s2]).sum(-1)[m]
        num += int((d == 0).sum())
        den += int(m.sum())
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", nargs="+", default=DEFAULT_NEW, help="cut_dir:licence, one per bundle")
    ap.add_argument("--out", default="data/oga4_clean")
    ap.add_argument("--tag", default="v9", help="names runs_out/corpus_<tag>_report.json and _sheet.png")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    NEW = [tuple(x.rsplit(":", 1)) for x in args.new]
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- hashes of everything we already have, so the CC0 material cannot duplicate it
    seen = {}
    for d in [x for x in OLD if x != str(OUT).replace(chr(92), "/")]:
        n0 = len(seen)
        for p in sorted({q.replace("\\", "/") for q in glob.glob(f"{d}/**/*.png", recursive=True)}):
            try:
                hh = pixhash(np.array(Image.open(p).convert("RGBA")))
            except Exception:
                continue
            seen.setdefault(hh, p)
        print(f"{d}: {len(seen) - n0} new hashes ({len(seen)} total)", flush=True)

    prov = {}
    for d, lic in NEW:
        f = Path(d) / "provenance.csv"
        if f.exists():
            for r in csv.DictReader(open(f, encoding="utf-8")):
                prov[f"{d}/{r['path']}"] = (r["zip"], r["member"], r["licence"])
    print(f"provenance rows: {len(prov)}", flush=True)

    man = open(f"{args.out}_manifest.csv", "w", newline="", encoding="utf-8")
    mw = csv.writer(man)
    mw.writerow(["path", "src", "zip", "member", "licence", "side", "palette", "flat"])
    n_kept = n_dup = n_junk = n_selfdup = 0
    sides, pals, flats, kept_paths, lic_count = [], [], [], [], Counter()
    for d, lic in NEW:
        files = sorted(glob.glob(f"{d}/**/*.png", recursive=True))
        if args.limit:
            files = files[:args.limit]
        print(f"{d}: {len(files)} candidates", flush=True)
        for i, p in enumerate(files):
            p = p.replace("\\", "/")
            try:
                a = np.array(Image.open(p).convert("RGBA"))
            except Exception:
                n_junk += 1
                continue
            if not keep(a):
                n_junk += 1
                continue
            hh = pixhash(a)
            if hh in seen:
                if seen[hh].startswith(d):
                    n_selfdup += 1
                else:
                    n_dup += 1
                continue
            seen[hh] = p
            n_kept += 1
            name = f"{n_kept:07d}.png"
            Image.fromarray(a, "RGBA").save(OUT / name)
            z, m, li = prov.get(p, ("", "", lic))
            side, pal, fl = max(a.shape[:2]), palette(a), flat_ratio(a)
            mw.writerow([name, d, z, m, li, side, pal, round(fl, 4)])
            lic_count[li] += 1
            sides.append(side)
            pals.append(pal)
            flats.append(fl)
            kept_paths.append(OUT / name)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(files)}: kept {n_kept}, dup {n_dup}, selfdup {n_selfdup}, junk {n_junk}",
                      flush=True)
    man.close()

    sides = np.array(sides)
    small = sides <= 24
    rep = {"kept": n_kept, "dropped_cross_corpus_duplicate": n_dup, "dropped_internal_duplicate": n_selfdup,
           "dropped_filter": n_junk, "existing_corpus_unique": len(seen) - n_kept,
           "native": {str(R): int((sides <= R).sum()) for R in (12, 16, 20, 24, 32, 64)},
           "palette_median_all": float(np.median(pals)) if pals else 0,
           "palette_median_native24": float(np.median(np.array(pals)[small])) if small.any() else 0,
           "flat_median_all": round(float(np.median(flats)), 4) if flats else 0,
           "flat_median_native24": round(float(np.median(np.array(flats)[small])), 4) if small.any() else 0,
           "licences": dict(lic_count)}
    Path("runs_out").mkdir(exist_ok=True)
    json.dump(rep, open(f"runs_out/corpus_{args.tag}_report.json", "w"), indent=2)
    print(json.dumps(rep, indent=2), flush=True)

    # ---- contact sheet of natively tiny survivors: the material the 16 px bucket actually needs
    tiny = [p for p, s in zip(kept_paths, sides) if s <= 24]
    random.seed(0)
    random.shuffle(tiny)
    cols, cell = 20, 48
    rows = max(1, min(10, (len(tiny) + cols - 1) // cols))
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for k, p in enumerate(tiny[:cols * rows]):
        im = Image.open(p).convert("RGBA")
        f = max(1, cell // max(im.size))
        im = im.resize((im.width * f, im.height * f), Image.NEAREST)
        bg = Image.new("RGB", im.size, (240, 240, 240))
        bg.paste(im, (0, 0), im)
        sheet.paste(bg, ((k % cols) * cell, (k // cols) * cell))
    sheet.save(f"runs_out/corpus_{args.tag}_sheet.png")
    print(f"sheet: {len(tiny)} sprites natively <= 24 px", flush=True)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
