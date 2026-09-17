"""FD-DINOv2 against an explicit reference directory.

fd_fair.py builds its reference from the corpus; the object-only protocol (09-18) needs a reference that has already
been filtered by content (a vision model labels every reference sprite as object / effect / fragment / glyph / junk, and
only objects are kept), so the reference is passed as a directory of ready PNGs.

Usage: python src/v6/fd_ref.py --size 16 --ref runs_out/ref_object_s16 --gen dirA dirB ...
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from v6 import fd_dino as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ref = sorted(glob.glob(a.ref + "/*.png"))
    print(f"reference: {len(ref)} images from {a.ref}", flush=True)
    res = {}
    for g in a.gen:
        files = sorted(glob.glob(g + "/*.png"))
        fd = F.fd(files, ref, a.size)
        res[g] = round(fd, 2)
        print(f"FD@{a.size} {g} (n={len(files)}) = {fd:.2f}", flush=True)
    if a.out:
        old = json.load(open(a.out)) if Path(a.out).exists() else {}
        old.update(res)
        json.dump(old, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
