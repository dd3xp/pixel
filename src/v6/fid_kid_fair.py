"""Second metric family on the SAME saved samples as fd_fair.py: Inception-v3 clean-FID and KID (clean-fid, Parmar 2022).

Reference = runs_out/ref_totensor_s{R} (the fd_fair reference, real -> to_tensor(R)); floor = heldout3000_totensor_s{R}.
Every RGBA png is composited on white and NEAREST-upscaled to --up (64) into runs_out/_incep_s{R}/<tag>/ so the Inception
resize (to 299, clean mode) sees identical inputs for real and generated.  Reference features are extracted once per R.

Usage: python src/v6/fid_kid_fair.py --size 16 --gen runs_out/v7h_matched_eval/s16 [...] [--floor] [--out json]
Prints:  INCEP@{R} <dir> (n=...) FID=..  KID=..e-3
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def prep_dir(src, R, up):
    tag = Path(src).parent.name if Path(src).name.startswith("s") else Path(src).name
    dst = Path(f"runs_out/_incep_s{R}") / tag
    files = sorted(glob.glob(str(Path(src) / "*.png")))
    if len(list(dst.glob("*.png"))) != len(files):
        dst.mkdir(parents=True, exist_ok=True)
        for k, f in enumerate(files):
            im = Image.open(f).convert("RGBA")
            assert im.size == (R, R), (f, im.size)
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            Image.alpha_composite(bg, im).convert("RGB").resize((up, up), Image.NEAREST).save(dst / f"{k:05d}.png")
    return dst, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--gen", nargs="*", default=[])
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--up", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    R = args.size
    from cleanfid import fid
    from cleanfid.features import build_feature_extractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_model = build_feature_extractor("clean", device)

    def feats(d):
        return fid.get_folder_features(str(d), feat_model, num_workers=4, batch_size=128, device=device, mode="clean",
                                       description=f"{Path(d).name}: ")

    ref_dir, n_ref = prep_dir(f"runs_out/ref_totensor_s{R}", R, args.up)
    fr = feats(ref_dir)
    mu_r, sig_r = np.mean(fr, axis=0), np.cov(fr, rowvar=False)
    todo = ([f"runs_out/heldout3000_totensor_s{R}"] if args.floor else []) + args.gen
    res = []
    for g in todo:
        d, n = prep_dir(g, R, args.up)
        if n < 100:
            print(f"INCEP@{R} {g} skipped (n={n})", flush=True)
            continue
        fg = feats(d)
        f_ = fid.frechet_distance(np.mean(fg, axis=0), np.cov(fg, rowvar=False), mu_r, sig_r)
        k_ = fid.kernel_distance(fr, fg)
        line = f"INCEP@{R} {g} (n={n}) FID={f_:.2f}  KID={k_ * 1000:.2f}e-3"
        print(line, flush=True)
        res.append({"size": R, "gen": g, "n": n, "fid": float(f_), "kid": float(k_)})
    if args.out:
        p = Path(args.out)
        old = json.loads(p.read_text()) if p.exists() else []
        p.write_text(json.dumps(old + res, indent=1))


if __name__ == "__main__":
    main()
