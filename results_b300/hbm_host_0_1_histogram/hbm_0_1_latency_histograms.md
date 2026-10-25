# 0 与 1 个 host warp 的 HBM 延迟直方图

对应图：`hbm_0_1_latency_histograms.png`、`hbm_0_1_latency_histograms.svg`。左图无 host warp，右图有一个连续读取 host memory 的 warp；host warp 不是 pointer chasing。两组均采集 65536 个正式样本，使用预测试 min–P99 的 512 个线性桶和相同显示坐标。数据说明：`dataset_hbm0.md`、`dataset_hbm1.md`。
