"""FD-DINOv2@R with a *pipeline-matched* reference set.

The original fd_dino.py reference set composites full-res real sprites on white and
BOX-squashes them to RxR (aspect NOT preserved, soft blended edges), while every
generator is trained on train_v7/train_cond `to_tensor` (aspect preserved, centred
on transparent canvas, hard alpha).  Real data pushed through `to_tensor` scores
FD 58.95 against that reference (2026-09-06), so the old metric's headroom was mostly
pipeline mismatch.  Here the reference is REAL[:3000] -> to_tensor(R) -> RGBA png,
cached under runs_out/ref_totensor_s{R}/, and then scored with fd_dino.fd unchanged.

Usage:
  python src/v6/fd_fair.py --size 16 --gen runs_out/derisk_v7sweep/s16 [--gen ...] [--floor]
  --floor also scores 3000 *held-out* real sprites (disjoint from the reference) via
  to_tensor, i.e. the real-vs-real floor at matched n.
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from v6 import fd_dino as F  # noqa: E402
from train_cond import to_tensor  # noqa: E402
from sample_cond import to_rgba  # noqa: E402


def real_split():
    """Same seed-0 shuffle as eval_probe.sh; returns (reference paths, held-out unique paths)."""
    real = glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")
    random.seed(0)
    random.shuffle(real)
    ref = real[:3000]
    refset = {p.replace("\\", "/") for p in ref}
    held = sorted({p.replace("\\", "/") for p in real} - refset)
    random.seed(2)
    random.shuffle(held)
    return ref, held


def dump_totensor(paths, out, R):
    out = Path(out)
    files = sorted(glob.glob(str(out / "*.png")))
    if len(files) == len(paths):
        return files
    out.mkdir(parents=True, exist_ok=True)
    for k, p in enumerate(paths):
        x = to_tensor(Image.open(p).convert("RGBA"), R)
        to_rgba((x + 1) / 2).save(out / f"{k:05d}.png")
    return sorted(glob.glob(str(out / "*.png")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--gen", nargs="*", default=[])
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--out", default=None, help="json to append results into")
    args = ap.parse_args()
    R = args.size
    ref, held = real_split()
    ref_files = dump_totensor(ref, f"runs_out/ref_totensor_s{R}", R)
    res = {}
    if args.floor:
        held_files = dump_totensor(held[:3000], f"runs_out/heldout3000_totensor_s{R}", R)
        res["floor_heldout3000"] = round(F.fd(held_files, ref_files, R), 2)
        print(f"FAIR FD@{R} floor (3000 held-out real via to_tensor) = {res['floor_heldout3000']}", flush=True)
    for g in args.gen:
        files = sorted(glob.glob(f"{g}/*.png"))
        fd = F.fd(files, ref_files, R)
        res[g] = round(fd, 2)
        print(f"FAIR FD@{R} {g} (n={len(files)}) = {fd:.2f}", flush=True)
    if args.out:
        old = json.load(open(args.out)) if Path(args.out).exists() else {}
        old.update(res)
        json.dump(old, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
