# NVIDIA B300 分离满载数据集

实验结构：测量 SM 不执行 TMA，其余 SM 持续执行 TMA。预测试和正式测量各 4096 个样本，实际 TMA 进度与测量窗口重叠。

正式结果：平均 **988.697 ns**，同期带宽 **7442.870 GB/s**，重叠 32.500 ms。预测试平均 997.544 ns。

直方图使用预测试 min–P99 的 512 个线性桶，范围 480.0–1416.0 ns，桶宽 1.830 ns；桶内 4048.0 个，低于下限 3.0 个，高于 P99 45.0 个。裁剪不影响原始数据和平均值。

文件：`separate_samples.csv`、`separate_pretest_samples.csv`、`separate_blocks.csv`、`separate_histogram.csv`、`separate_histogram_metadata.csv`。
