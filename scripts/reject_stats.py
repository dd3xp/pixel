"""Is the third of our samples the probe rejects badly drawn, or just not drawn at all?

The extreme tail of the sheet looks like solid slabs, but that sheet shows the fourteen lowest
probabilities, which is the tail by construction. This measures the whole rejected group against the
accepted one, so the impression either survives or does not.
"""
import glob
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "src/v6")
import fd_dino as FD  # noqa: E402
from fd_fair import native_sizes, real_split  # noqa: E402
from train_cond import to_tensor  # noqa: E402

R = 16
GEN = sys.argv[1] if len(sys.argv) > 1 else "runs_out/v8n_icg2_pal4_matched_eval/s16"


def render(paths, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    have = sorted(out.glob("*.png"))
    if len(have) == len(paths):
        return [str(p) for p in have]
    for i, p in enumerate(paths):
        t = to_tensor(Image.open(p).convert("RGBA"), R)
        a = ((t + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        Image.fromarray(a, "RGBA").save(out / f"{i:05d}.png")
    return [str(p) for p in sorted(out.glob("*.png"))]


def stats(path):
    a = np.array(Image.open(path).convert("RGBA"))
    op = a[:, :, 3] >= 128
    if op.sum() < 4:
        return 1.0, 0.0, 0
    cols = Counter(map(tuple, a[:, :, :3][op]))
    modal = cols.most_common(1)[0][1] / op.sum()
    return modal, op.mean(), len(cols)


real = sorted({p.replace("\\", "/") for p in
               glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")})
sz = native_sizes(real)
_, held = real_split(native_R=R, pool="old")
held = {q.replace("\\", "/") for q in held}
pos = [p for p in real if sz[p] <= R and p not in held][:1200]
neg = [p for p in real if 25 <= sz[p] <= 64][:1200]
fp = FD.embed(render(pos, f"runs_out/_probe_pos_s{R}"), R)
fn = FD.embed(render(neg, f"runs_out/_probe_neg_s{R}"), R)
X = np.concatenate([fp, fn])
y = np.concatenate([np.ones(len(fp)), np.zeros(len(fn))])

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X)
clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)

files = sorted(glob.glob(str(Path(GEN) / "*.png")))
p = clf.predict_proba(sc.transform(FD.embed(files, R)))[:, 1]
S = np.array([stats(f) for f in files])          # modal fraction, coverage, colours
for name, m in (("rejected (p<0.5)", p < 0.5), ("accepted (p>=0.5)", p >= 0.5)):
    print(f"{name:20s} n={m.sum():5d}  modal colour {S[m, 0].mean():.3f}  "
          f"coverage {S[m, 1].mean():.3f}  colours {S[m, 2].mean():.1f}")
# how many rejected sprites are essentially one slab of colour?
slab = (S[:, 0] > 0.8)
print(f"near-single-colour (modal > 0.8): {100 * slab[p < 0.5].mean():.1f}% of rejected, "
      f"{100 * slab[p >= 0.5].mean():.1f}% of accepted")
# and for real sprites, as the yardstick
Sr = np.array([stats(f) for f in render(pos, f"runs_out/_probe_pos_s{R}")])
print(f"real natively-small sprites: modal {Sr[:, 0].mean():.3f}, "
      f"near-single-colour {100 * (Sr[:, 0] > 0.8).mean():.1f}%")
