"""1b (train_gft_res.py) on the 8-rung model v7r8, so 12 px gets a lower bucket (8 px) for the internalised reference.

Same trick as train_v7r8.py: grow train_v7's bucket list in place before train_gft_res is imported, and set
LOWER[12] = 8. The training reference is still "early snapshot under the lower bucket"; the snapshot must have 8 rows,
so pass a grown copy of workdir/v7r_snap10k (8 px row copied from its 12 px row, made by --make_grown_snap).
Usage:
  python src/v6/train_gft_res8.py --make_grown_snap workdir/v7r_snap10k/model_latest.pt workdir/v7r_snap10k/model_grown8.pt
  python src/v6/train_gft_res8.py --init workdir/v7r8/model_latest.pt --snap workdir/v7r_snap10k/model_grown8.pt \
         --steps 10000 --out workdir/probe_gft_v7r8 --csv_suffix _recap
"""
import sys

import torch

sys.path.insert(0, "src/v6")
import train_v7

train_v7.BUCKETS[:] = [12, 16, 20, 24, 32, 48, 64, 8]
train_v7.LOW[:] = [0, 1, 2, 3, 7]
train_v7.BATCH[8] = 384
import train_gft_res  # noqa: E402  (must see the grown bucket list)

train_gft_res.LOWER[12] = 8
train_gft_res.LOWER[8] = None

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--make_grown_snap":
        sd = torch.load(sys.argv[2], map_location="cpu")
        old = sd["class_embedding.weight"]
        assert old.shape[0] == 7, old.shape
        sd["class_embedding.weight"] = torch.cat([old, old[0:1]], 0)
        torch.save(sd, sys.argv[3])
        print("grown snapshot ->", sys.argv[3])
    else:
        train_gft_res.main()
