# 有无 host-read warp 的延迟直方图

对应图：`pcie_latency_histograms.png`、`pcie_latency_histograms.svg`。左图无 host warp，右图有一个持续读取 mapped host memory 的 warp；两图使用相同坐标范围和 512 个线性桶。host 组平均延迟比 baseline 高约 2.1%，主体略右移，没有新增超长尾。

数据说明：`dataset_pcie_baseline.md`、`dataset_pcie_host.md`。
