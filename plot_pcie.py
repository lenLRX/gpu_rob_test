#!/usr/bin/env python3
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_b300/pcie")
summary = pd.read_csv(RESULTS / "pcie_summary.csv")
formal = summary[summary.experiment.isin(["pcie_baseline", "pcie_host"])].copy()
statistics = []
for name in ["pcie_baseline", "pcie_host"]:
    values = pd.read_csv(RESULTS / f"{name}_samples.csv").latency_ns
    row = formal[formal.experiment == name].iloc[0]
    statistics.append({
        "experiment": name,
        "average_ns": values.mean(),
        "median_ns": values.median(),
        "p95_ns": values.quantile(0.95),
        "p99_ns": values.quantile(0.99),
        "max_ns": values.max(),
        "host_concurrent_bandwidth_GB_per_s": row.host_concurrent_bandwidth_GB_per_s,
    })
pd.DataFrame(statistics).to_csv(RESULTS / "pcie_statistics.csv", index=False)

histograms = {
    name: pd.read_csv(RESULTS / f"{name}_histogram.csv")
    for name in ["pcie_baseline", "pcie_host"]
}
x_min = min(h.lower_ns.min() for h in histograms.values())
x_max = max(h.upper_ns.max() for h in histograms.values())
y_max = max(h["count"].max() for h in histograms.values()) * 1.1
host = formal[formal.experiment == "pcie_host"].iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
for ax, name, title in zip(
        axes, ["pcie_baseline", "pcie_host"],
        ["Without host-read warp", f"With host-read warp ({host.host_concurrent_bandwidth_GB_per_s:.3f} GB/s)"]):
    histogram = histograms[name]
    centers = (histogram.lower_ns + histogram.upper_ns) / 2
    widths = histogram.upper_ns - histogram.lower_ns
    ax.bar(centers, histogram["count"], width=widths, edgecolor="none")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Pointer-chasing latency (ns)")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
fig.suptitle("Single-SM PCIe host-read comparison (512 linear bins, pretest min-P99)")
fig.tight_layout()
fig.savefig(RESULTS / "pcie_latency_histograms.png", dpi=180, bbox_inches="tight")
fig.savefig(RESULTS / "pcie_latency_histograms.svg", bbox_inches="tight")
plt.close(fig)
