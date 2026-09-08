"""FD-DINOv2 vs CLIP text alignment at 16 px (v7h, seed 0, matched protocol): the CFG weight sweep traces one frontier,
the reference-guidance rows sit strictly below it.  Numbers from experiment_log 09-08 02:15 / 04:10 / 07:52 / 08:05.
Output: paper_assets/fig_pareto_fd_clip.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG = {1: (16.39, 29.43), 1.5: (12.24, 29.74), 2: (12.98, 29.87), 2.5: (14.56, 29.95), 3: (16.67, 30.01), 4: (21.98, 30.06),
       7: (42.95, 30.04), 10: (62.12, 30.05)}                      # w -> (FD, CLIP 100cos)
PAG = {1.5: (12.57, 29.45), 2: (12.62, 29.41), 3: (12.90, 29.45)}  # perturbed-attention guidance, mid block
PAG_CT = (11.73, 29.74)                                            # PAG w2 + CFG term 1.5 (3 NFE)
OURS = {"bucket:12  w=2": (8.59, 29.37), "autoguidance 10k  w=1.5": (8.98, 29.58),
        "composed  w=1.5": (7.67, 29.51), "composed$\\circ$bucketu:12  w=1.5": (8.60, 29.77),
        "composed + CFG term 1.5 (3 NFE)": (9.19, 29.72), "composed + CFG term 2 (3 NFE)": (11.10, 29.89)}
REAL = (3.45, 29.80)

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ws = sorted(CFG)
ax.plot([CFG[w][1] for w in ws], [CFG[w][0] for w in ws], "-o", color="0.35", label="CFG, w = 1 … 10")
for w in ws:
    ax.annotate(f"w={w:g}", (CFG[w][1], CFG[w][0]), textcoords="offset points", xytext=(5, 2), fontsize=7, color="0.35")
ax.plot([v[1] for v in PAG.values()], [v[0] for v in PAG.values()], "s", color="tab:orange", ms=5, label="PAG (mid), w = 1.5 / 2 / 3")
ax.plot([PAG_CT[1]], [PAG_CT[0]], "s", mfc="none", color="tab:orange", ms=6, label="PAG w=2 + CFG term (3 NFE)")
mk = {"bucket:12  w=2": ("^", "tab:blue"), "autoguidance 10k  w=1.5": ("v", "tab:green"), "composed  w=1.5": ("*", "tab:red")}
for k, (fd, cl) in OURS.items():
    m, c = mk.get(k, ("D", "tab:red"))
    ax.plot([cl], [fd], m, color=c, ms=9 if m == "*" else 6, mfc=c if k in mk else "none", label=k)
ax.plot([REAL[1]], [REAL[0]], "P", color="k", ms=8, label="real held-out (floor)")
ax.axvline(REAL[1], color="k", lw=0.5, ls=":")
ax.set_xlabel("CLIP text alignment (100·cos, ↑)")
ax.set_ylabel("FD-DINOv2 @16 px (↓)")
ax.set_ylim(0, 66)
ax.set_xlim(29.3, 30.15)
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="upper left", ncol=1, framealpha=0.9)
ax.set_title("16 px, v7h, seed 0: alignment–fidelity frontier", fontsize=10)
fig.tight_layout()
fig.savefig("paper_assets/fig_pareto_fd_clip.png", dpi=200)
print("wrote paper_assets/fig_pareto_fd_clip.png")
