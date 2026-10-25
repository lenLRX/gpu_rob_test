# RTX 5080 TMA 带宽压力与 pointer-chasing 延迟

本目录包含已在真实 GPU 上编译、运行的 CUDA microbenchmark。测量目标是 GDDR7 片外显存；没有使用模拟数据或主机时间替代设备计时。

## 实测结果

最终完整运行日志为 `results/full_run.log`，汇总数据为 `results/summary.csv`。

| 实验 | TMA SM | 平均延迟 | 同期实测带宽 | 总 TMA 时间窗带宽 |
|---|---:|---:|---:|---:|
| 1. 空载，仅测量 SM | 0 | 316.752 ns | 0 | 0 |
| 2. 测量 SM 无 TMA，其他 SM 执行 TMA | 83 | 1839.380 ns | 882.486 GB/s | 878.084 GB/s |
| 3. 所有 SM 执行 TMA，测量 SM 增加一个测量 warp | 84 | 1805.150 ns | 870.541 GB/s | 877.961 GB/s |
| 50% 带宽目标，同 SM 并发 | 6 | 371.997 ns | 422.548 GB/s | 416.079 GB/s |

空载中位延迟为 316 ns，P99 为 344 ns。所有平均值均直接对全部原始样本求算术平均，未剔除长尾；直方图的显示范围裁剪不影响平均值。50% 带宽实验的正式实测带宽为同轮实验 3 的 48.54%。

带宽扫描有 18 个配置：`1, 2, 4, 5, 6, 7, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 84` 个 TMA SM。每点的 SM 数、实际同期带宽、平均延迟、最小/最大延迟、字节数和时间窗均在 `results/sweep.csv`。例如：

- 1 SM：69.375 GB/s，382.580 ns。
- 16 SM：836.612 GB/s，503.294 ns。
- 32 SM：868.475 GB/s，881.652 ns。
- 84 SM：864.019 GB/s，1980.560 ns。

图表：

- `results/bandwidth_latency.png` / `.svg`：实测带宽—平均延迟曲线，点标注为 TMA SM 数。
- `results/bandwidth_scaling.png` / `.svg`：TMA SM 数—实测带宽。
- `results/full_load_histograms.png` / `.svg`：实验 2、3 的满负载直方图。
- `results/half_bandwidth_histogram.png` / `.svg`：接近 50% 带宽时的延迟直方图。

## 环境与配置

- GPU：NVIDIA GeForce RTX 5080，compute capability 12.0 / `sm_120`，84 SM，16 GiB GDDR7。
- L2：64 MiB；指针链工作集为 512 MiB，即 L2 的 8 倍。
- OS：Windows 上的 WSL2，Linux kernel `6.18.26.1-microsoft-standard-WSL2`。
- `nvidia-smi`：615.65.07；KMD 616.64；CUDA UMD 13.4。
- CUDA runtime 12.9，实际编译器 `/usr/local/cuda/bin/nvcc` 12.9.86。
- 系统默认 `nvcc` 为 11.5，不能用于 sm_120；Makefile 显式使用已有 CUDA 12.9，不修改系统 CUDA 配置。

环境原始记录：`results/nvidia_smi.txt`、`results/environment_runtime.txt`、`results/nvcc_version.txt`、`results/os.txt`。

### Pointer chasing

- `134217728` 个 32 位索引；固定种子的 Fisher–Yates 随机置换被连接成单个循环依赖链。
- 每次正式运行有 4096 个样本，每个样本计时 8 次真实串行依赖加载，保存每次加载的平均 ns；即每个正式实验 32768 次依赖加载。
- 下一次地址由前一次返回值计算。结束计时对加载值执行非恒等依赖运算后恢复索引，防止 timer 在数据返回前读取；另有 128 次预热和一次不计入样本的计时预热。
- 加载指令为 `ld.global.cv.u32`，不使用 L1/L2 缓存结果。最终 SASS 保存在 `results/benchmark_sass.txt`。

### TMA

- 全局源为 1 GiB，大于 64 MiB L2；源按活动 TMA SM 分片并顺序流式读取。
- 使用二维 `CUtensorMap`（256 字节 × 64 行）和 `cp.async.bulk.tensor.2d`，每次传输 16 KiB，L2 promotion 为 NONE。
- 每个活动 SM 一个 TMA warp，4 个 shared-memory slot 流水传输；每 SM 65536 次传输，共 1 GiB。
- 每个 slot 的有效数据由 warp 读取并归约 checksum；主机检查所有 TMA 块完成规定轮数且 checksum 非零，防止流量被消除。
- 同期带宽使用测量窗口前后各块已完成 TMA tile 数之差 × 16 KiB，再除以同一 `%globaltimer` 时间窗。`total_bandwidth_GB_per_s` 则使用整个 TMA kernel 的实际字节数和总时长。
- 每个块记录 `%smid`；主机验证同一次运行中的块占据不同 SM，并检查所有并发运行的重叠时间和 TMA progress 均非零。

### 计时、等待与直方图

- 所有设备时间均通过 PTX `%globaltimer` 读取，单位 ns；未使用 `clock`、`clock64` 或主机计时。
- 连续 10000 次读数测得最小非零间隔为 32 ns。每样本合并 8 次加载以降低量化及强负载计时异常影响。
- 测量 warp 在全局块启动屏障后设备自旋等待目标 5000 ns；本次所有运行实际等待均为 5024 ns。
- 实验 2、3 和 50% 带宽实验各自先独立预测试，再正式测量。直方图在预测试 `[min, P99]` 主体范围内设置 512 个线性等宽桶，坐标轴也使用线性刻度，以提高主体分布的区分度。
- 本次实验 2 使用 580–3196 ns，桶宽 5.111 ns，显示 4059 个正式样本，37 个高于 P99；实验 3 使用 736–3464 ns，桶宽 5.330 ns，显示 4060 个，另有 10 个低于预测试最小值、26 个高于 P99。
- 50% 带宽直方图使用 288–420 ns，桶宽 0.260 ns，显示 4054 个样本，另有 1 个低于最小值、41 个高于 P99。低延迟样本受 globaltimer 量化影响，因此部分桶为空。
- 裁剪仅用于直方图显示；所有原始样本和平均值均保留完整长尾。各 `*_histogram_metadata.csv` 记录范围、桶宽及上下尾排除数量。
- 正式重叠时间：实验 2 为 60.411 ms，实验 3 为 59.287 ms，50% 带宽实验为 12.240 ms。

## 原始数据

`results/summary.csv` 包含全部 23 次运行（空载、两次预测试、两次正式满载、18 点扫描、50% 带宽预测试与正式测量）。

每个 `results/<experiment>_samples.csv` 保存全部 4096 个原始样本；每个 `results/<experiment>_blocks.csv` 保存逐块 SM、起止时间、TMA 起止时间、实际字节数、轮数和 checksum。直方图边界与计数分别在 `results/separate_histogram.csv`、`results/concurrent_histogram.csv` 和 `results/half_bandwidth_histogram.csv`。

## 构建与复现

```bash
cd /home/cc/gpu_rob_test
make -j2
make run
```

`make run` 会重新记录环境、执行全部实验并生成 CSV 和 PNG/SVG。绘图需要 Python 3、pandas 和 matplotlib。

也可单独运行：

```bash
./benchmark --mode idle --output results
./benchmark --mode separate --output results
./benchmark --mode concurrent --output results
./benchmark --mode sweep --output results
python3 plot.py
```

注意：单独模式会重写 `results/summary.csv`；完整交付汇总由 `--mode all` 生成。可调参数包括 `--samples`、`--iterations`、`--chain-mb`、`--source-mb`，默认值即上述已验证配置。
