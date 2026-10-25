# 单 HBM lane 对 0–31 个 host warp 数据集

单 SM 中仅测量 warp 的 lane 0 执行完整 512 MiB HBM pointer chasing，扫描 0–31 个 host PCIe warp，共 32 个配置。每配置 4096 个样本，每样本 8 次串行加载。

基线平均 377.348 ns，31 个 host warp 时 3351.900 ns。`hbm_host_sweep.csv` 包含平均/最小/最大延迟和 host 带宽；`hbm_lane_averages.csv` 每配置仅含 lane 0；`full_run.log` 为实际运行日志。
