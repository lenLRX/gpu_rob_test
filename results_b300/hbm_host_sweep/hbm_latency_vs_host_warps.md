# HBM lane 0 延迟—host warp 数折线图

对应图：`hbm_latency_vs_host_warps.png`、`hbm_latency_vs_host_warps.svg`。横轴为同一 SM 上的 host PCIe warp 数，纵轴为 HBM lane 0 的平均延迟（ns）。0 个 host warp 是额外基线，1–31 个为正式扫描。数据源为 `hbm_host_sweep.csv`，说明见 `dataset_sweep.md`。
