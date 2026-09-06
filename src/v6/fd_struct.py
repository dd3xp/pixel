"""Structure-only fair FD: score generated S-as-RGBA sprites against the *structure* of the
fair reference set (REAL[:3000] -> to_tensor -> make_struct -> S-as-RGBA), plus the held-out
structure floor.  Isolates stage-1 (structure generator) quality from the colouriser.

Usage:
  python src/v6/fd_struct.py --size 16 --gen runs_out/probe_sgen_eval/struct16
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from v6 import fd_dino as F  # noqa: E402
from fd_fair import dump_totensor, real_split  # noqa: E402
from sample_cond import to_rgba  # noqa: E402
from train_cond import make_struct  # noqa: E402
from train_sgen import struct_to_rgba  # noqa: E402


def dump_struct(src_dir, out):
    out = Path(out)
    src = sorted(glob.glob(f"{src_dir}/*.png"))
    files = sorted(glob.glob(str(out / "*.png")))
    if len(files) == len(src):
        return files
    out.mkdir(parents=True, exist_ok=True)
    for k, p in enumerate(src):
        x = torch.from_numpy(np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = x * 2 - 1
        s = struct_to_rgba(make_struct(x[None]))[0]
        to_rgba((s + 1) / 2).save(out / f"{k:05d}.png")
    return sorted(glob.glob(str(out / "*.png")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--gen", nargs="*", default=[], help="dirs of S-as-RGBA sprites (stage-1 output)")
    ap.add_argument("--struct_of", nargs="*", default=[], help="dirs of RGBA sprites; scored after make_struct")
    ap.add_argument("--out", default="runs_out/fair_fd16_struct.json")
    args = ap.parse_args()
    R = args.size
    ref, held = real_split()
    dump_totensor(ref, f"runs_out/ref_totensor_s{R}", R)
    dump_totensor(held[:3000], f"runs_out/heldout3000_totensor_s{R}", R)
    ref_s = dump_struct(f"runs_out/ref_totensor_s{R}", f"runs_out/ref_struct_s{R}")
    held_s = dump_struct(f"runs_out/heldout3000_totensor_s{R}", f"runs_out/heldout3000_struct_s{R}")
    res = {"floor_heldout3000_struct": round(F.fd(held_s, ref_s, R), 2)}
    print(f"STRUCT FD@{R} floor = {res['floor_heldout3000_struct']}", flush=True)
    for g in args.struct_of:  # RGBA sample dirs -> quantise to S first (structure of a one-stage model)
        files = dump_struct(g, g.rstrip("/") + "_structq")
        fd = F.fd(files, ref_s, R)
        res[f"struct_of:{g}"] = round(fd, 2)
        print(f"STRUCT FD@{R} struct_of {g} (n={len(files)}) = {fd:.2f}", flush=True)
    for g in args.gen:
        files = sorted(glob.glob(f"{g}/*.png"))
        fd = F.fd(files, ref_s, R)
        res[g] = round(fd, 2)
        print(f"STRUCT FD@{R} {g} (n={len(files)}) = {fd:.2f}", flush=True)
    old = json.load(open(args.out)) if Path(args.out).exists() else {}
    old.update(res)
    json.dump(old, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
