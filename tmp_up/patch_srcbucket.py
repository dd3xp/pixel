import sys
p = "src/v6/train_srcbucket.py"
s = open(p, encoding="utf-8").read()
def sub(a, b):
    global s
    assert s.count(a) == 1, (a[:60], s.count(a))
    s = s.replace(a, b)

sub("""                s = max(Image.open(path).size)
                b = next((i for i, bb in enumerate(BUCKETS) if bb >= s), len(BUCKETS) - 1)
                self.rows.extend([(path, text)] * repeat)
                self.bucket_of.extend([b] * repeat)
                for t in LOW:
                    if t != b and s >= BUCKETS[t] * 1.25:
                        self.rows.extend([(path, text)] * repeat)
                        self.bucket_of.extend([t] * repeat)
                        n_aug += repeat""",
"""                s = max(Image.open(path).size)
                b = next((i for i, bb in enumerate(BUCKETS) if bb >= s), len(BUCKETS) - 1)
                self.rows.extend([(path, text)] * repeat)
                self.bucket_of.extend([b] * repeat)
                self.src_of.extend([b] * repeat)          # native: source bucket == target bucket
                for t in LOW:
                    if t != b and s >= BUCKETS[t] * 1.25:
                        self.rows.extend([(path, text)] * repeat)
                        self.bucket_of.extend([t] * repeat)
                        self.src_of.extend([b] * repeat)  # downscaled: remember what it was drawn at
                        n_aug += repeat""")
sub("        return to_tensor(im, BUCKETS[b]), text, b\n",
    "        return to_tensor(im, BUCKETS[b]), text, b * len(BUCKETS) + self.src_of[i]\n")
sub("num_class_embeds=len(BUCKETS),",
    "num_class_embeds=len(BUCKETS) * len(BUCKETS),  # (target bucket, source bucket) pair")
sub('bucket={BUCKETS[int(b[0])]}"',
    'bucket={BUCKETS[int(b[0]) // len(BUCKETS)]}<-src{BUCKETS[int(b[0]) % len(BUCKETS)]}"')
# the sampler and any other consumer index bucket_of, which keeps its old meaning (target bucket)
for cand in ["self.rows, self.bucket_of = [], []", "self.rows = []\n        self.bucket_of = []"]:
    if cand in s:
        s = s.replace(cand, cand.replace("self.bucket_of = []", "self.bucket_of, self.src_of = [], []")
                      if "\n" in cand else "self.rows, self.bucket_of, self.src_of = [], [], []")
        break
else:
    raise SystemExit("could not find the list initialisation")
# training-time eval grid: sample the native mode (src == target)
sub("""    lab = torch.full((n,), BUCKETS.index(size), device=device, dtype=torch.long)""",
    """    bi = BUCKETS.index(size)
    lab = torch.full((n,), bi * len(BUCKETS) + bi, device=device, dtype=torch.long)  # native mode""")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("OK src_of refs:", s.count("src_of"), "| pair table:", "len(BUCKETS) * len(BUCKETS)" in s)
