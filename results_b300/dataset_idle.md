# NVIDIA B300 空载 pointer-chasing 数据

仅一个 SM 的 lane 0 执行 512 MiB 随机单循环链访问，无 TMA 或 host 背景流量。正式样本 4096 个，每样本为 8 次串行加载的平均值。平均延迟 **412.457 ns**，最小 280 ns，最大 564 ns。

对应文件：`idle_samples.csv`（全部样本）、`idle_blocks.csv`（SM 和时间戳）、`summary.csv`。
