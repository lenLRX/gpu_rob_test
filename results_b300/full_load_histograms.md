# NVIDIA B300 两份满负载直方图

对应图：`full_load_histograms.png`、`full_load_histograms.svg`。左图是其他 SM 执行 TMA 的分离结构，右图是所有 SM 执行 TMA 且测量 SM 增加测量 warp 的结构。两图均使用预测试 min–P99 范围内的 512 个线性桶，横轴为 ns；极端长尾仅裁出显示，原始样本和平均值不裁剪。

数据说明：`dataset_separate.md`、`dataset_concurrent.md`。计数文件：`separate_histogram.csv`、`concurrent_histogram.csv`。
