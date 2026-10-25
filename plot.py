#!/usr/bin/env python3
from pathlib import Path
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"


def save(fig, stem):
    fig.savefig(RESULTS / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(RESULTS / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


summary = pd.read_csv(RESULTS / "summary.csv")
sweep = summary[summary.experiment.str.startswith("sweep_")].copy()
sweep["tma_sms"] = sweep.tma_sms.astype(int)
sweep = sweep.sort_values("tma_sms")
sweep.to_csv(RESULTS / "sweep.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sweep.concurrent_bandwidth_GB_per_s, sweep.average_ns,
        marker="o", linewidth=1.8)
for _, row in sweep.iterrows():
    ax.annotate(str(row.tma_sms),
                (row.concurrent_bandwidth_GB_per_s, row.average_ns),
                textcoords="offset points", xytext=(4, 5), fontsize=8)
ax.set_xlabel("Measured concurrent TMA bandwidth (GB/s)")
ax.set_ylabel("Mean pointer-chasing latency (ns)")
ax.yaxis.set_major_locator(MultipleLocator(100))
gpu_name = os.environ.get("GPU_NAME", "RTX 5080")
ax.set_title(f"TMA bandwidth pressure vs. dependent-load latency ({gpu_name})")
ax.grid(True, alpha=0.3)
save(fig, "bandwidth_latency")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sweep.tma_sms, sweep.concurrent_bandwidth_GB_per_s, marker="o")
ax.set_xlabel("TMA SM count")
ax.set_ylabel("Measured concurrent bandwidth (GB/s)")
ax.set_title("TMA bandwidth scaling")
ax.grid(True, alpha=0.3)
save(fig, "bandwidth_scaling")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, name, title in zip(
        axes, ["separate", "concurrent"],
        ["Experiment 2: other SMs issue TMA", "Experiment 3: all SMs issue TMA"]):
    histogram = pd.read_csv(RESULTS / f"{name}_histogram.csv")
    centers = (histogram.lower_ns + histogram.upper_ns) / 2
    widths = histogram.upper_ns - histogram.lower_ns
    ax.bar(centers, histogram["count"], width=widths, align="center",
           edgecolor="none")
    ax.set_xlabel("Latency per dependent load, 8-load samples (ns)")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
fig.suptitle("Full-load latency histograms (512 linear bins, pretest min-P99 range)")
fig.tight_layout()
save(fig, "full_load_histograms")

half = summary[summary.experiment == "half_bandwidth"].iloc[0]
full = summary[summary.experiment == "concurrent"].iloc[0]
histogram = pd.read_csv(RESULTS / "half_bandwidth_histogram.csv")
centers = (histogram.lower_ns + histogram.upper_ns) / 2
widths = histogram.upper_ns - histogram.lower_ns
percentage = 100 * half.concurrent_bandwidth_GB_per_s / full.concurrent_bandwidth_GB_per_s
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(centers, histogram["count"], width=widths, align="center", edgecolor="none")
ax.set_xlabel("Latency per dependent load, 8-load samples (ns)")
ax.set_ylabel("Samples")
ax.set_title(f"50% bandwidth target: {int(half.tma_sms)} TMA SMs, "
             f"{half.concurrent_bandwidth_GB_per_s:.1f} GB/s ({percentage:.1f}% of full)")
ax.grid(True, axis="y", alpha=0.25)
fig.suptitle("Pointer-chasing latency histogram (512 linear bins, pretest min-P99 range)")
fig.tight_layout()
save(fig, "half_bandwidth_histogram")
