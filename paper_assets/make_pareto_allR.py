"""FD-DINOv2 vs CLIP alignment at 12 / 20 / 24 / 32 px (v7h, seed 0, matched protocol): the CFG weight sweep traces the
frontier at each resolution, the reference-guidance rows sit below it; composed∘bucketu (unconditional-text wrong-bucket
reference) recovers alignment at ~unchanged FD.  Numbers from experiment_log 09-08 §10:10 / §11:45 / §12:25 (diag_review2-6)
and the CLIP table in paper_outline (g).  Output: paper_assets/fig_pareto_allR.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# R -> dict(CFG={w:(FD,CLIP)}, rows={name:(FD,CLIP)}, real=(FD_floor, CLIP_real))
D = {
    12: dict(CFG={1.5: (10.44, 29.54), 2: (9.71, 29.65), 3: (11.09, 29.76), 4: (13.02, 29.76)},
             rows={"autoguidance 10k  w=1.5": (8.54, 29.41), "autog$\\circ$bucketu:16  w=1.5": (9.28, 29.58)},
             real=(3.37, 29.55)),
    20: dict(CFG={1.5: (42.61, 29.75), 2: (40.57, 29.92), 2.5: (39.53, 30.02), 4: (45.92, 30.16)},
             rows={"bucket:16  w=2": (32.89, 29.34), "autoguidance 10k  w=1.5": (31.78, 29.45),
                   "composed  w=1.5": (29.66, 29.48), "composed$\\circ$bucketu:16  w=1.5": (29.81, 29.70)},
             real=(12.14, 29.86)),
    24: dict(CFG={1.5: (71.22, 29.70), 2: (67.34, 29.89), 3: (72.93, 30.00), 4: (79.80, 30.11)},
             rows={"bucket:16  w=2": (60.41, 29.09), "autoguidance 10k  w=1.5": (53.36, 29.35),
                   "composed  w=1.5": (48.70, 29.33), "composed$\\circ$bucketu:16  w=1.5": (49.78, 29.60)},
             real=(12.96, 29.81)),
    32: dict(CFG={1.5: (92.21, 29.15), 2: (83.65, 29.43), 3: (86.99, 29.60), 4: (96.85, 29.72)},
             rows={"bucket:24  w=2": (77.45, 28.49), "autoguidance 10k  w=1.5": (75.23, 28.69),
                   "composed  w=1.5": (69.27, 28.78), "composed$\\circ$bucketu:24  w=1.5": (65.27, 29.18)},
             real=(13.77, 29.65)),
}
MK = [("composed$", ("D", "tab:red")), ("autog$", ("D", "tab:green")), ("bucket:", ("^", "tab:blue")),
      ("autoguidance", ("v", "tab:green")), ("composed  ", ("*", "tab:red"))]  # ordered: the bucketu rows first

fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, (R, d) in zip(axes, sorted(D.items())):
    ws = sorted(d["CFG"])
    ax.plot([d["CFG"][w][1] for w in ws], [d["CFG"][w][0] for w in ws], "-o", color="0.35", label="CFG sweep")
    for w in ws:
        ax.annotate(f"w={w:g}", d["CFG"][w][::-1], textcoords="offset points", xytext=(4, 2), fontsize=6, color="0.35")
    for k, (fd, cl) in d["rows"].items():
        m, c = next(v for p, v in MK if p in k)
        ax.plot([cl], [fd], m, color=c, ms=10 if m == "*" else 6, mfc=c if m != "D" else "none", label=k)
    ax.axvline(d["real"][1], color="k", lw=0.5, ls=":")
    ax.axhline(d["real"][0], color="k", lw=0.5, ls=":")
    ax.set_title(f"{R} px  (floor {d['real'][0]:.2f}, real CLIP {d['real'][1]:.2f})", fontsize=9)
    ax.set_xlabel("CLIP 100·cos (↑)", fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.5, loc="lower right", framealpha=0.9)
axes[0].set_ylabel("FD-DINOv2 (↓)", fontsize=8)
fig.suptitle("Alignment–fidelity frontier per resolution (v7h, seed 0): guidance rows lie below the CFG curve at every R", fontsize=9)
fig.tight_layout()
fig.savefig("paper_assets/fig_pareto_allR.png", dpi=200)
print("wrote paper_assets/fig_pareto_allR.png")
