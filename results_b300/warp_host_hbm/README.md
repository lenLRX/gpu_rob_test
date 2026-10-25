# B300 单 warp host / HBM 混合延迟实验

本目录包含两组仅使用一个 warp 的真实 B300 延迟实验。host 内存均为 256 MiB mapped pinned memory，HBM 链为 256 MiB。

## 结果

| 实验 | 统计 lane | 平均延迟 | 中位数 | P95 | P99 | 最大值 |
|---|---:|---:|---:|---:|---:|---:|
| 全部 lane 访问 host | 32 | 1534.378 ns | 1524 ns | 1568 ns | 1592 ns | 20432 ns |
| lane 0–15 访问 HBM，lane 16–31 访问 host | 仅 lane 0–15 | 1449.564 ns | 1440 ns | 1496 ns | 1550.6 ns | 15352 ns |

混合组只统计前 16 个 HBM lane，但这些 lane 仍落在约 1.45 μs 的 host 延迟范围。原因是测量循环没有控制流分裂，所有 lane 执行同一条加载；warp 需要等待较慢的 host lane 完成后才能继续后续指令，因此前 16 个 HBM lane 也观察到接近 host 的延迟。

图表：`warp_host_hbm_histograms.png` / `.svg`。两组使用相同的显示坐标范围，便于比较。

## 无 divergence 的访问实现

- 每个 lane 在测量循环前计算独立基址。
- host-only：lane 0–31 分别指向 32 个独立 host chain slice。
- mixed：lane 0–15 指向 HBM chain slice，lane 16–31 指向 host chain slice。
- 测量循环内所有 lane 执行相同的 `ld.global.cv.u32` 和相同的 8 次串行依赖展开，没有 `if (lane < 16)` 访问分支。
- 初始基址选择不位于测量访问路径中；SASS 已保存为 `warp_host_hbm.sass`，其中测量循环的分支为 uniform 循环控制，不是 lane 分支。
- 每个 lane 使用独立单循环随机置换链，避免不同 lane 之间共享地址。每次运行更换链起点，减少缓存复用。

## 采样和直方图

- 每 lane 4096 个样本，每样本为 8 次串行依赖加载的平均值。
- host-only 正式原始样本数：32 × 4096 = 131072。
- mixed 正式原始样本数：16 × 4096 = 65536。
- 原始 CSV 包含 `sample,lane,latency_ns`；mixed 仅包含 lane 0–15。
- 所有计时使用 `%globaltimer`，分辨率 32 ns。
- 两组各自预测试并正式测量，直方图使用预测试 min–P99 范围内的 512 个线性桶。
- host-only 范围 1424–1660 ns，桶宽 0.463 ns；显示 130592 个，32 个低于下限、448 个高于 P99。
- mixed 范围 1340–1536 ns，桶宽 0.385 ns；显示 64752 个，16 个低于下限、768 个高于 P99。
- 直方图裁剪不影响原始数据和平均值。

## 文件与校验

- `warp_summary.csv`：两次预测试和两次正式运行汇总。
- `warp_statistics.csv`：均值、中位数、P95、P99 和最大值。
- `warp_host_samples.csv`、`warp_mixed_samples.csv`：逐 sample、逐 lane 原始数据。
- `warp_host_histogram.csv`、`warp_mixed_histogram.csv` 及对应 metadata。
- `full_run.log`：B300 实际运行日志。
- `warp_host_hbm.sass`：最终设备汇编。

已验证样本数、lane 范围、正延迟、每份直方图 512 桶，以及桶内和上下尾计数之和等于原始样本数。两组实际均运行在 SM 142。

## 复现

```bash
cd /root/gpu_rob_test_b300
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_103 -lineinfo \
  warp_host_hbm.cu -o warp_host_hbm
mkdir -p results/warp_host_hbm
./warp_host_hbm --output results/warp_host_hbm \
  2>&1 | tee results/warp_host_hbm/full_run.log
```

本地绘图：

```bash
cd /home/cc/gpu_rob_test
python3 plot_warp_host_hbm.py results_b300/warp_host_hbm
```

源码副本：`../source/warp_host_hbm.cu` 和 `../source/plot_warp_host_hbm.py`。
