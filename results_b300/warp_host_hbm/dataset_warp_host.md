# 单 warp 数据集：32 lane 全部访问 host

仅一个 warp。host 组统计全部 32 lane；mixed 组 lane 0–15 访问 HBM、lane 16–31 访问 host，仅统计前 16 lane。测量循环使用统一加载，没有 lane 控制流分裂。每 lane 4096 个样本，每样本 8 次串行加载。

本组统计 32 lane，共 131072 个正式样本；平均 **1534.378 ns**，中位数 1524 ns，P95 1568 ns，P99 1592.0 ns。

文件：`warp_host_samples.csv`、`warp_host_pretest_samples.csv`、`warp_host_histogram.csv`、`warp_host_histogram_metadata.csv`；SASS 为 `warp_host_hbm.sass`。
