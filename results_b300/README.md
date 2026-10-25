# B300 TMA 带宽压力与 pointer-chasing 延迟

本目录只保存 B300 远程实测结果，与 RTX 5080 的 `results/` 分离。全部最终数据来自 NVIDIA B300 SXM6 AC，未使用模拟结果。

## 最终实测结果

| 实验 | TMA SM | 平均延迟 | 同期实测带宽 | 总 TMA 时间窗带宽 |
|---|---:|---:|---:|---:|
| 空载，仅测量 SM | 0 | 412.457 ns | 0 | 0 |
| 测量 SM 无 TMA，其他 SM 执行 TMA | 147 | 988.697 ns | 7442.870 GB/s | 6821.000 GB/s |
| 所有 SM 执行 TMA，测量 SM 增加一个测量 warp | 148 | 983.909 ns | 7441.360 GB/s | 6827.270 GB/s |
| 约 50% 带宽，同 SM 并发 | 36 | 435.188 ns | 3826.358 GB/s | 3804.520 GB/s |

半带宽正式测量为同轮满负载同期带宽的约 51.4%。带宽扫描包含 19 个配置：`1, 2, 4, 5, 6, 7, 8, 12, 16, 24, 32, 36, 40, 48, 56, 64, 72, 80, 148` 个 TMA SM。所有点的数据见 `sweep.csv`。

图表：

- `bandwidth_latency.png` / `.svg`：实测带宽—平均延迟曲线；纵轴为 ns，主刻度间隔 100 ns。
- `bandwidth_scaling.png` / `.svg`：TMA SM 数—实测带宽。
- `full_load_histograms.png` / `.svg`：两种满负载结构的延迟直方图。
- `half_bandwidth_histogram.png` / `.svg`：36 SM、约 51.4% 满负载带宽时的延迟直方图。

## 环境和配置

- GPU：NVIDIA B300 SXM6 AC，compute capability 10.3 / `sm_103`，148 SM。
- 显存：287421038592 字节；L2：132644864 字节（约 126.5 MiB）。
- NVIDIA-SMI / 驱动：595.91.07；CUDA runtime 和 `nvcc`：13.2。
- 远程 OS：Ubuntu，Linux kernel `6.8.0-147-generic`。
- 指针链：512 MiB、134217728 个 32 位索引、单个随机置换循环。
- 每次运行 4096 个样本，每样本为 8 次串行加载的平均延迟；原始样本全部保留。
- 加载：`ld.global.cv.u32`；计时：`%globaltimer`，最小非零分辨率 32 ns。
- 实际设备等待：5024 ns，目标为 5000 ns。
- TMA 源：1 GiB；二维 tensor map；每个活动 SM 一个 TMA warp。
- 最终 TMA 配置：**3 stages × 64 KiB**，使用 mbarrier 初始化 fence。
- 每 SM 262144 次传输，即 16 GiB，保证整个 pointer-chasing 测量窗口内 TMA 持续运行。

参考文件 `tma_load_bench.cu` 的独立扫描结果为最佳 3 stages × 64 KiB、7.071 TB/s，见 `tma_load_bench.log`。据此将原 4×16 KiB 配置的约 4.06 TB/s 提升到最终约 7.44 TB/s。

## 直方图

三份直方图均使用预测试 min–P99 范围内的 512 个线性桶；裁剪只影响显示，不影响原始样本和平均值。

- 分离满载：480–1416 ns，桶宽 1.830 ns；显示 4048 个样本，3 个低于下限、45 个高于 P99。
- 同 SM 满载：512–1416 ns，桶宽 1.768 ns；显示 4057 个，2 个低于下限、37 个高于 P99。
- 半带宽：312–544 ns，桶宽 0.455 ns；显示 4058 个，4 个低于下限、34 个高于 P99。

各 `*_histogram_metadata.csv` 保存边界、桶宽及排除数量。

## 原始数据与校验

`summary.csv` 包含 26 次运行：空载、两组满载预测试和正式测量、19 点扫描、半带宽预测试和正式测量。

每个 `*_samples.csv` 有 4096 个真实样本；每个 `*_blocks.csv` 保存逐块 SM、时间戳、TMA 字节数、轮数和 checksum。已验证：

- 所有样本数正确且延迟大于零。
- 每次运行的 block 均映射到不同 SM。
- 每个 TMA block 精确完成 262144 × 65536 字节。
- 所有负载实验均有非零 TMA progress 和并发重叠。
- 分离实验的测量块无 TMA 流量；同 SM 实验的 148 个块均有 TMA 流量。
- 三份直方图的桶内计数与上下尾计数之和均为 4096。
- PNG/SVG 均已生成并检查。

## 复现

远程主机和工作目录：`root@195.26.233.146 -p 13443`，`/root/gpu_rob_test_b300`。

```bash
cd /root/gpu_rob_test_b300
make -B benchmark SM=103 TMA_SLOTS=3 TMA_TILE_BYTES=65536
./benchmark --mode all --iterations 262144 --output results 2>&1 | tee results/full_run.log
```

本地重新绘图：

```bash
cd /home/cc/gpu_rob_test
GPU_NAME='NVIDIA B300' python3 plot.py results_b300
```

最终源码副本位于 `source/`，远程连接与执行记录位于 `remote_connection.log`。早期另一地址的连接失败仅作历史记录，见 `remote_connection_initial_failure.log`。

## 追加：单 SM PCIe host-read 对照

`pcie/` 中另有单 SM 对照实验：无 host-read warp 时平均延迟为 374.548 ns，有一个持续读取 mapped host memory 的 warp 时为 382.431 ns，增加 2.105%。同期 host-read 请求带宽为 1.060 GB/s，未观察到新增超长尾。

详见 `pcie/README.md` 和 `pcie/pcie_latency_histograms.png` / `.svg`。

## 追加：单 warp host 与 HBM/host 混合实验

`warp_host_hbm/` 中新增两组单 warp 实验：32 lane 全部访问 host 时平均延迟为 1534.378 ns；lane 0–15 访问 HBM、lane 16–31 访问 host 时，仅统计前 16 lane，平均延迟为 1449.564 ns。访问循环使用统一加载指令，没有按 lane 分裂的控制流。

详见 `warp_host_hbm/README.md` 和 `warp_host_hbm/warp_host_hbm_histograms.png` / `.svg`。

## 追加：单 HBM lane 对 0–31 个 host warp 扫描

`hbm_host_sweep/` 中新增单 SM 扫描：仅测量 warp 的 lane 0 执行完整 512 MiB HBM pointer chasing，host warp 从 0 到 31 个变化。平均延迟从 377.348 ns 总体上升至 20 个以上时的约 3.35–3.45 μs；31 个 host warp 时为 3351.900 ns。

详见 `hbm_host_sweep/README.md`、`hbm_host_sweep/hbm_host_sweep.csv` 和 `hbm_host_sweep/hbm_latency_vs_host_warps.png` / `.svg`。

## 追加：31 个 host warp 下的大样本 HBM 直方图

`hbm_host_histogram/` 中新增 31 个 host warp 下的 lane 0 HBM 延迟分布。正式测量包含 65536 个样本、524288 次串行加载，平均延迟 3387.470 ns，中位数 3356 ns，P99 为 4000 ns。

详见 `hbm_host_histogram/README.md`、`hbm_host_histogram/hbm31_samples.csv` 和 `hbm_host_histogram/hbm31_latency_histogram.png` / `.svg`。

## 追加：0 与 1 个连续 host-read warp 的大样本直方图

`hbm_host_0_1_histogram/` 中新增两组各 65536 个样本的对照。无 host warp 时 HBM lane 0 平均延迟为 382.304 ns；有一个连续读取 host memory 的 warp 时为 455.403 ns。host warp 不执行 pointer chasing。

详见 `hbm_host_0_1_histogram/README.md` 和 `hbm_host_0_1_histogram/hbm_0_1_latency_histograms.png` / `.svg`。
