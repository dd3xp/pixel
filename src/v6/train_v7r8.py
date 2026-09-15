"""v7r8 = v7r + an 8 px rung, so that 12 px (the bottom of the user's 12-24 px range) has a LOWER bucket and every
reference-based method -- label guidance, composed guidance, 1b -- becomes defined there (arch_crossres_module.md 6.2).

Everything is train_v7.py; this wrapper only
  * appends 8 to BUCKETS (index 7, so the seven existing rows keep their indices and meaning),
  * adds 8 px to the low-bucket augmentation (every sprite of at least 10 px also appears box-downsampled to 8 px),
  * loads v7r with the class embedding grown from 7 to 8 rows, the new row initialised from the 12 px row.
Usage (fine-tune): python src/v6/train_v7r8.py --init_v6 workdir/v7r/model_latest.pt --steps 20000 --out workdir/v7r8 \
                   --exclude runs_out/holdout_exclude.txt --csv_suffix _recap --snap_every 5000
Sampling: sample_e.py --buckets 12,16,20,24,32,48,64,8
"""
import sys

import torch

sys.path.insert(0, "src/v6")
import train_v7

train_v7.BUCKETS[:] = [12, 16, 20, 24, 32, 48, 64, 8]    # in place: modules that imported BUCKETS see it too
train_v7.LOW[:] = [0, 1, 2, 3, 7]
train_v7.BATCH[8] = 384


def load_grow_8(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    key = "class_embedding.weight"
    old = sd[key]
    assert old.shape[0] == 7, old.shape
    sd[key] = torch.cat([old, old[0:1]], 0)                # 8 px row <- 12 px row
    model.load_state_dict(sd, strict=True)
    print(f"v7r8 init from {ckpt_path}: class embedding 7 -> 8 rows (8 px <- 12 px)", flush=True)


train_v7.surgical_load = load_grow_8

if __name__ == "__main__":
    train_v7.main()
