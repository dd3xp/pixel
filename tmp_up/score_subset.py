"""Score any sprite directory against the SAME reference, on the SAME prompt subset our own model is scored on.

Small-n FD is biased upward, so the only fair reading is like-for-like: every row here uses the identical
reference set and the identical number of samples.
"""
import argparse, glob, sys
from pathlib import Path
sys.path.insert(0, "src/v6")
import fd_dino
from fd_fair import real_split, dump_totensor

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=16)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--dirs", nargs="+", required=True, help="name=dir")
a = ap.parse_args()

ref, _ = real_split()
refdir = f"runs_out/ref_totensor_s{a.size}"
rf = dump_totensor(ref, refdir, a.size)
print(f"reference: {len(rf)} real sprites at {a.size} px (full set, unchanged)")

for spec in a.dirs:
    name, d = spec.split("=", 1)
    fs = sorted(glob.glob(d + "/*.png"))[:a.n]
    if not fs:
        print(f"{name:28s} NO FILES in {d}"); continue
    fd = fd_dino.fd(fs, rf, a.size)
    print(f"{name:28s} n={len(fs):4d}  FD-DINOv2@{a.size} = {fd:.2f}", flush=True)
