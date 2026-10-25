# RTX 5080 分离满载数据集

实验结构：测量 SM 不执行 TMA，其余 SM 持续执行 TMA。预测试和正式测量各 4096 个样本，实际 TMA 进度与测量窗口重叠。

正式结果：平均 **1839.381 ns**，同期带宽 **882.486 GB/s**，重叠 60.411 ms。预测试平均 1817.693 ns。

直方图使用预测试 min–P99 的 512 个线性桶，范围 580.0–3196.0 ns，桶宽 5.111 ns；桶内 4059.0 个，低于下限 0.0 个，高于 P99 37.0 个。裁剪不影响原始数据和平均值。

文件：`separate_samples.csv`、`separate_pretest_samples.csv`、`separate_blocks.csv`、`separate_histogram.csv`、`separate_histogram_metadata.csv`。
