#!/usr/bin/env python3
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_b300/hbm_host_sweep")
data = pd.read_csv(RESULTS / "hbm_host_sweep.csv").sort_values("host_warps")
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(data.host_warps, data.average_ns, marker="o", markersize=4, linewidth=1.8)
ax.set_xlabel("Concurrent PCIe host-read warp count on the same SM")
ax.set_ylabel("Mean HBM lane-0 pointer-chasing latency (ns)")
ax.set_title("One HBM lane vs. 0-31 host-read warps (NVIDIA B300)")
ax.set_xticks(range(0, 32, 2))
ax.set_xlim(-0.5, 31.5)
ax.set_ylim(0, (data.average_ns.max() // 500 + 1) * 500)
ax.yaxis.set_major_locator(MultipleLocator(500))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / "hbm_latency_vs_host_warps.png", dpi=180, bbox_inches="tight")
fig.savefig(RESULTS / "hbm_latency_vs_host_warps.svg", bbox_inches="tight")
plt.close(fig)
