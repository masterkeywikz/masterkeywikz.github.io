"""Regenerate the article's SVG figures with Python, NumPy, and Matplotlib."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "images"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#79716b",
    "axes.labelcolor": "#302b25",
    "text.color": "#302b25",
    "xtick.color": "#79716b",
    "ytick.color": "#79716b",
    "figure.facecolor": "#fffdf7",
    "axes.facecolor": "#fffdf7",
    "svg.fonttype": "none",
    "svg.hashsalt": "inference-roofline",
})


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
intensity = np.geomspace(0.5, 300, 500)
ax.loglog(intensity, np.minimum(120, 2 * intensity), color="#326b77", lw=3)
ax.axvline(60, color="#b98548", ls="--", lw=1.4)
ax.set(xlim=(0.5, 300), ylim=(0.8, 250),
       xlabel="Arithmetic intensity (FLOPs/byte)",
       ylabel="Performance ceiling (TFLOP/s)")
ax.set_xticks([1, 4, 16, 60, 240], labels=["1", "4", "16", "60", "240"])
ax.set_yticks([1, 4, 16, 60, 120], labels=["1", "4", "16", "60", "120"])
ax.minorticks_off()
ax.grid(alpha=0.15)
ax.set_title("Two limits, one roof", loc="left", fontsize=20, pad=20)
ax.text(0.7, 160, "Illustrative accelerator · 120 TFLOP/s · 2 TB/s", fontsize=11)
for batch, label, offset in [(1, "B = 1", (12, -15)), (16, "B = 16", (12, -15))]:
    ax.scatter(batch, 2 * batch, s=65, color="#326b77", zorder=3)
    ax.annotate(label, (batch, 2 * batch), xytext=offset, textcoords="offset points")
ax.annotate("Ridge: 60 FLOPs/byte", (60, 120), xytext=(-200, -20),
            textcoords="offset points", arrowprops={"arrowstyle": "->", "color": "#b98548"})
ax.text(1.3, 12, "Bandwidth ceiling", color="#326b77", rotation=33)
ax.text(75, 155, "Compute ceiling", fontsize=11)
save(fig, "inference-roofline")

fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
contexts = np.array([1000, 4000, 8000, 16000])
kv_gb = 16 * contexts * (2 * 32 * 8 * 128 * 2) / 1e9
positions = np.arange(len(contexts))
ax.bar(positions, np.full(4, 16), width=0.58, color="#326b77", label="Weight reads")
ax.bar(positions, kv_gb, bottom=16, width=0.58, color="#d4a66c", label="KV reads")
for x, kv in zip(positions, kv_gb):
    ax.text(x, 8, "16 GB", ha="center", va="center", color="white")
    ax.text(x, 16 + kv + 0.8, f"+ {kv:.1f} GB KV", ha="center", fontsize=11)
ax.set_xticks(positions, labels=["1,000", "4,000", "8,000", "16,000"])
ax.set(xlabel="Cached tokens per sequence", ylabel="HBM read traffic per step (GB)", ylim=(0, 61))
ax.set_title("Longer context adds traffic at every step", loc="left", fontsize=18, pad=20)
ax.text(-0.42, 57, "Batch 16 · 32 layers · 8 KV heads · head dimension 128 · BF16", fontsize=10.5)
ax.legend(loc="upper left", bbox_to_anchor=(0, 0.89), frameon=False, fontsize=11)
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=0.15)
save(fig, "inference-kv-traffic")

for batch in [1, 16, 60, 120]:
    compute = 2 * 8e9 * batch / 120e12
    memory = 16e9 / 2e12
    step = max(compute, memory)
    print(f"B={batch}: compute={compute * 1000:.3f} ms, "
          f"step={step * 1000:.3f} ms, total={batch / step:.1f} tok/s, "
          f"per-sequence={1 / step:.1f} tok/s")
step = (16e9 + 16 * 8000 * 131072) / 2e12
print(f"With KV: {step * 1000:.3f} ms, {16 / step:.1f} total tok/s, "
      f"{1 / step:.1f} per-sequence tok/s")
