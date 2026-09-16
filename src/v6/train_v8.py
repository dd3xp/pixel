"""v8 training wrapper: same network and schedule as train_v7, different TARGET DISTRIBUTION.

09-16 finding: our 16 px samples look soft next to a 1024 px generator's downscaled output because that is
what the targets look like -- only 2,675 of 37,489 sprites are natively <= 16 px, the rest is 25-64 px art
BOX-squashed into the low buckets, and on prompts whose ground truth IS native pixel art we emit 53 colours
where the truth has 13.  Three changes, all on the data side:

  --max_down R   a sprite may only feed a bucket at most R x smaller than its native size (v7: 1/1.25, i.e.
                 48 px art trained the 16 px bucket).  Default 1.5.
  --quant K      quantize each target to <= K colours (median cut, alpha untouched), so flat regions are
                 actually flat.  0 disables.
  --dupdrop F    drop paths listed in F (data/corpus_v8_dupdrop.txt): 4,888 sprites in the old corpus are
                 pixel-identical copies, which is why 15.5% of the held-out eval sprites still had a twin
                 in training even though the hold-out list is by path.

Everything else (architecture, optimiser, buckets, CLIP text encoder, EMA) is train_v7's, so a v8 run is
directly comparable to a v7r run of the same length.

Usage: python src/v6/train_v8.py --out workdir/v8pilot --steps 25000 --csv_suffix _recap [--quant 16]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import train_v7 as T  # noqa: E402

MAX_DOWN = 1.5
QUANT = 0


def quantize(im, k):
    """Median-cut to <= k colours on the opaque pixels; alpha is left alone."""
    a = np.array(im)
    rgb = Image.fromarray(a[:, :, :3], "RGB").quantize(colors=k, method=Image.MEDIANCUT,
                                                       dither=Image.Dither.NONE).convert("RGB")
    out = np.dstack([np.array(rgb), a[:, :, 3]])
    out[a[:, :, 3] == 0] = 0
    return Image.fromarray(out, "RGBA")


_orig_to_tensor = T.to_tensor


def to_tensor(im, side):
    if QUANT and max(im.size) <= side:      # quantize before any padding, after any downscale below
        im = quantize(im, QUANT)
    elif QUANT:
        im = quantize(T.downscale_rgba(im, side), QUANT)
    return _orig_to_tensor(im, side)


class NativeSpritesV8(T.NativeSprites):
    """train_v7.NativeSprites, but a sprite only feeds a lower bucket it is within MAX_DOWN of."""

    def __init__(self, sources, exclude=()):
        super().__init__(sources, exclude)
        keep_rows, keep_b = [], []
        for (path, text), b in zip(self.rows, self.bucket_of):
            s = max(Image.open(path).size)
            if s <= T.BUCKETS[b] * MAX_DOWN:
                keep_rows.append((path, text))
                keep_b.append(b)
        dropped = len(self.rows) - len(keep_rows)
        self.rows, self.bucket_of = keep_rows, keep_b
        print(f"v8: dropped {dropped} over-downscaled copies, kept {len(self.rows)}", flush=True)


def main():
    global MAX_DOWN, QUANT
    argv = []
    it = iter(sys.argv[1:])
    dupdrop = None
    for a in it:
        if a == "--max_down":
            MAX_DOWN = float(next(it))
        elif a == "--quant":
            QUANT = int(next(it))
        elif a == "--dupdrop":
            dupdrop = next(it)
        else:
            argv.append(a)
    if dupdrop:
        drop = [l.strip() for l in open(dupdrop, encoding="utf-8") if l.strip()]
        print(f"v8: excluding {len(drop)} duplicate copies", flush=True)
        tmp = Path("runs_out/v8_exclude.txt")
        base = []
        if "--exclude" in argv:
            base = [l.strip() for l in open(argv[argv.index("--exclude") + 1], encoding="utf-8") if l.strip()]
        tmp.write_text("\n".join(base + drop), encoding="utf-8")
        if "--exclude" in argv:
            argv[argv.index("--exclude") + 1] = str(tmp)
        else:
            argv += ["--exclude", str(tmp)]
    T.to_tensor = to_tensor
    T.NativeSprites = NativeSpritesV8
    sys.argv = [sys.argv[0]] + argv
    print(f"v8: max_down={MAX_DOWN} quant={QUANT or 'off'}", flush=True)
    T.main()


if __name__ == "__main__":
    main()
