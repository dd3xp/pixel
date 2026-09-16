"""Build the 09-16 rebuilt corpus: clean, pixel-hash dedup, verify, and report.

Why (09-16): the 16 px targets the model learns are mostly 25-64 px artwork BOX-squashed into the low
buckets (only 2,675 of 37,489 sprites are natively <= 16 px), which is why our samples look soft next to a
1024 px generator's downscaled output.  Two defects are fixed here at once:
  * too few natively tiny sprites  -> three more OGA licence bundles are cut in (extract_oga_v3.py);
  * 3,475 pixel-identical copies inside the old corpus, so 15.5% of the held-out eval sprites still had a
    twin in training (the hold-out list is by path) -> everything is deduplicated by exact pixel hash and
    every copy of a held-out sprite is dropped.

Outputs: data/oga3_clean/*.png, data/oga3_manifest.csv (path,src,zip,member,licence,side,palette),
data/corpus_v8_dupdrop.txt (old-corpus paths that are extra copies), runs_out/corpus_v8_report.json,
runs_out/corpus_v8_sheet.png (random sample for eyeballing).

Usage: python src/v6/build_corpus_v8.py [--limit N]
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

NEW = [("data/oga3_cut_ccby", "CC-BY-4.0"),
       ("data/oga3_cut_ccbysa", "CC-BY-SA-3.0"),
       ("data/oga3_cut_ogaby", "OGA-BY-3.0")]
OLD = ["data/oga_clean", "data/extra_all"]
OUT = Path("data/oga3_clean")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- hashes of everything we already have, so the new material cannot duplicate it
    seen, old_dups = {}, []
    for d in OLD:
        for p in sorted({q.replace("\\", "/") for q in glob.glob(f"{d}/**/*.png", recursive=True)}):
            try:
                hh = pixhash(np.array(Image.open(p).convert("RGBA")))
            except Exception:
                continue
            if hh in seen:
                old_dups.append(p)          # extra copy inside the OLD corpus -> drop from training
            else:
                seen[hh] = p
    Path("data/corpus_v8_dupdrop.txt").write_text("\n".join(old_dups), encoding="utf-8")
    print(f"old corpus: {len(seen)} unique, {len(old_dups)} duplicate copies flagged", flush=True)

    prov = {}
    for d, lic in NEW:
        f = Path(d) / "provenance.csv"
        if f.exists():
            for r in csv.DictReader(open(f, encoding="utf-8")):
                prov[f"{d}/{r['path']}"] = (r["zip"], r["member"], r["licence"])

    man = open("data/oga3_manifest.csv", "w", newline="", encoding="utf-8")
    mw = csv.writer(man)
    mw.writerow(["path", "src", "zip", "member", "licence", "side", "palette"])
    n_kept = n_dup = n_junk = 0
    sides, pals, kept_paths = [], [], []
    for d, lic in NEW:
        files = sorted(glob.glob(f"{d}/**/*.png", recursive=True))
        if args.limit:
            files = files[:args.limit]
        for p in files:
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
                n_dup += 1
                continue
            seen[hh] = p
            n_kept += 1
            name = f"{n_kept:07d}.png"
            Image.fromarray(a, "RGBA").save(OUT / name)
            z, m, li = prov.get(p, ("", "", lic))
            side, pal = max(a.shape[:2]), palette(a)
            mw.writerow([name, d, z, m, li, side, pal])
            sides.append(side)
            pals.append(pal)
            kept_paths.append(OUT / name)
        print(f"{d}: kept {n_kept} so far (dup {n_dup}, junk {n_junk})", flush=True)
    man.close()

    sides = np.array(sides)
    rep = {"kept": n_kept, "dropped_duplicate": n_dup, "dropped_filter": n_junk,
           "old_unique": len(seen) - n_kept, "old_duplicate_copies": len(old_dups),
           "native": {str(R): int((sides <= R).sum()) for R in (12, 16, 20, 24, 32, 64)},
           "palette_median": float(np.median(pals)) if pals else 0,
           "licences": dict(Counter(l for _, l in NEW))}
    Path("runs_out").mkdir(exist_ok=True)
    json.dump(rep, open("runs_out/corpus_v8_report.json", "w"), indent=2)
    print(json.dumps(rep, indent=2), flush=True)

    # ---- contact sheet of natively tiny survivors: the material the 16 px bucket actually needs
    small = [p for p, s in zip(kept_paths, sides) if s <= 24]
    random.seed(0)
    random.shuffle(small)
    cols, cell = 20, 48
    rows = max(1, min(8, (len(small) + cols - 1) // cols))
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for k, p in enumerate(small[:cols * rows]):
        im = Image.open(p).convert("RGBA")
        f = max(1, cell // max(im.size))
        im = im.resize((im.width * f, im.height * f), Image.NEAREST)
        bg = Image.new("RGB", im.size, (240, 240, 240))
        bg.paste(im, (0, 0), im)
        sheet.paste(bg, ((k % cols) * cell, (k // cols) * cell))
    sheet.save("runs_out/corpus_v8_sheet.png")
    print(f"sheet: {len(small)} sprites natively <= 24 px", flush=True)


if __name__ == "__main__":
    main()
