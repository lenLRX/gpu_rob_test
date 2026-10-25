#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
created = []


def write(directory, filename, title, body):
    path = directory / filename
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")
    created.append(path)


def row(summary, name):
    return summary[summary.experiment == name].iloc[0]


def metadata(directory, name):
    return pd.read_csv(directory / f"{name}_histogram_metadata.csv").iloc[0]


def main_documents(directory, device):
    summary = pd.read_csv(directory / "summary.csv")
    full = row(summary, "concurrent")
    docs = [
        ("dataset_summary.md", "汇总数据集"),
        ("dataset_idle.md", "空载数据集"),
        ("dataset_separate.md", "分离满载数据集"),
        ("dataset_concurrent.md", "同 SM 满载数据集"),
        ("dataset_half_bandwidth.md", "半带宽数据集"),
        ("dataset_sweep.md", "带宽扫描数据集"),
        ("bandwidth_latency.md", "带宽—平均延迟图"),
        ("bandwidth_scaling.md", "带宽 scaling 图"),
        ("full_load_histograms.md", "满负载直方图"),
        ("half_bandwidth_histogram.md", "半带宽直方图"),
    ]
    write(directory, "dataset_summary.md", f"{device} 全部实验汇总",
          f"`summary.csv` 汇总 {len(summary)} 次实际运行，包含空载、预测试、正式满载、带宽扫描和半带宽实验。每行记录 SM 数、平均/最小/最大延迟、同期实测带宽、TMA 字节数、globaltimer 时间窗及并发重叠。逐次原始样本和逐块统计由同名 `*_samples.csv`、`*_blocks.csv` 提供。环境记录见 `environment_runtime.txt`、`nvidia_smi.txt` 和 `full_run.log`。")
    idle = row(summary, "idle")
    write(directory, "dataset_idle.md", f"{device} 空载 pointer-chasing 数据",
          f"仅一个 SM 的 lane 0 执行 512 MiB 随机单循环链访问，无 TMA 或 host 背景流量。正式样本 4096 个，每样本为 8 次串行加载的平均值。平均延迟 **{idle.average_ns:.3f} ns**，最小 {idle.min_ns} ns，最大 {idle.max_ns} ns。\n\n对应文件：`idle_samples.csv`（全部样本）、`idle_blocks.csv`（SM 和时间戳）、`summary.csv`。")
    for name, title, description in [
        ("separate", "分离满载", "测量 SM 不执行 TMA，其余 SM 持续执行 TMA"),
        ("concurrent", "同 SM 满载", "所有 SM 执行 TMA，测量 SM 额外运行一个测量 warp"),
    ]:
        pre = row(summary, f"{name}_pretest")
        formal = row(summary, name)
        meta = metadata(directory, name)
        write(directory, f"dataset_{name}.md", f"{device} {title}数据集",
              f"实验结构：{description}。预测试和正式测量各 4096 个样本，实际 TMA 进度与测量窗口重叠。\n\n正式结果：平均 **{formal.average_ns:.3f} ns**，同期带宽 **{formal.concurrent_bandwidth_GB_per_s:.3f} GB/s**，重叠 {formal.overlap_ns / 1e6:.3f} ms。预测试平均 {pre.average_ns:.3f} ns。\n\n直方图使用预测试 min–P99 的 512 个线性桶，范围 {meta.pretest_min_ns}–{meta.pretest_p99_ns} ns，桶宽 {meta.bin_width_ns:.3f} ns；桶内 {meta.formal_in_range} 个，低于下限 {meta.formal_below_min} 个，高于 P99 {meta.formal_above_p99} 个。裁剪不影响原始数据和平均值。\n\n文件：`{name}_samples.csv`、`{name}_pretest_samples.csv`、`{name}_blocks.csv`、`{name}_histogram.csv`、`{name}_histogram_metadata.csv`。")
    half = row(summary, "half_bandwidth")
    meta = metadata(directory, "half_bandwidth")
    ratio = 100 * half.concurrent_bandwidth_GB_per_s / full.concurrent_bandwidth_GB_per_s
    write(directory, "dataset_half_bandwidth.md", f"{device} 半带宽数据集",
          f"采用同 SM 并发结构，从扫描点中自动选择实测带宽最接近满载 50% 的配置。选中 **{int(half.tma_sms)} 个 TMA SM**，正式平均延迟 **{half.average_ns:.3f} ns**，同期带宽 **{half.concurrent_bandwidth_GB_per_s:.3f} GB/s**，为同轮满载的 **{ratio:.2f}%**。\n\n预测试和正式测量各 4096 个样本。直方图为预测试 min–P99 的 512 个线性桶，范围 {meta.pretest_min_ns}–{meta.pretest_p99_ns} ns，桶宽 {meta.bin_width_ns:.3f} ns。\n\n文件：`half_bandwidth_samples.csv`、`half_bandwidth_pretest_samples.csv`、`half_bandwidth_blocks.csv`、`half_bandwidth_histogram.csv`、`half_bandwidth_histogram_metadata.csv`。")
    sweep = summary[summary.experiment.str.startswith("sweep_")].sort_values("tma_sms")
    points = ", ".join(str(int(x)) for x in sweep.tma_sms)
    write(directory, "dataset_sweep.md", f"{device} TMA 带宽扫描数据集",
          f"采用同 SM 并发结构，扫描 {len(sweep)} 个 TMA SM 配置：{points}。每个配置有 4096 个测量样本，并使用同期实际完成的 TMA 字节数计算带宽。\n\n`sweep.csv` 是曲线直接数据源，包含每个配置的 SM 数、平均延迟和实测带宽。每个 `sweep_N_samples.csv` 保存原始样本，每个 `sweep_N_blocks.csv` 保存逐块 TMA 统计。")
    write(directory, "bandwidth_latency.md", f"{device} 带宽—平均延迟图",
          "对应图：`bandwidth_latency.png`、`bandwidth_latency.svg`。横轴为同期实测 TMA 带宽（GB/s），纵轴为平均 pointer-chasing 延迟（ns），主刻度间隔 100 ns；点标签为 TMA SM 数。数据完全来自 `sweep.csv`，说明见 `dataset_sweep.md`。曲线不使用理论带宽或 SM 数代替实测带宽。")
    write(directory, "bandwidth_scaling.md", f"{device} TMA SM 数—带宽图",
          "对应图：`bandwidth_scaling.png`、`bandwidth_scaling.svg`。横轴为参与 TMA 的 SM 数，纵轴为同期实测带宽（GB/s），用于观察带宽爬坡和平台区。数据源为 `sweep.csv`，各配置的样本和逐块统计说明见 `dataset_sweep.md`。")
    write(directory, "full_load_histograms.md", f"{device} 两份满负载直方图",
          "对应图：`full_load_histograms.png`、`full_load_histograms.svg`。左图是其他 SM 执行 TMA 的分离结构，右图是所有 SM 执行 TMA 且测量 SM 增加测量 warp 的结构。两图均使用预测试 min–P99 范围内的 512 个线性桶，横轴为 ns；极端长尾仅裁出显示，原始样本和平均值不裁剪。\n\n数据说明：`dataset_separate.md`、`dataset_concurrent.md`。计数文件：`separate_histogram.csv`、`concurrent_histogram.csv`。")
    write(directory, "half_bandwidth_histogram.md", f"{device} 半带宽延迟直方图",
          f"对应图：`half_bandwidth_histogram.png`、`half_bandwidth_histogram.svg`。展示 {int(half.tma_sms)} 个 TMA SM、实际为满载带宽 {ratio:.2f}% 时的延迟分布。使用预测试 min–P99 的 512 个线性桶，保留原始长尾。\n\n数据说明：`dataset_half_bandwidth.md`；计数文件：`half_bandwidth_histogram.csv`、`half_bandwidth_histogram_metadata.csv`。")
    index = "# 结果文档索引\n\n" + "\n".join(f"- [{title}]({name})" for name, title in docs)
    write(directory, "RESULTS_INDEX.md", f"{device} 结果文档索引", index.split("\n", 2)[2])


def pcie_documents(directory):
    summary = pd.read_csv(directory / "pcie_summary.csv")
    stats = pd.read_csv(directory / "pcie_statistics.csv").set_index("experiment")
    for name, title, enabled in [("pcie_baseline", "无 host warp 基线", False),
                                 ("pcie_host", "单 host warp 对照", True)]:
        r = row(summary, name)
        s = stats.loc[name]
        traffic = f"同期 host 请求带宽 {r.host_concurrent_bandwidth_GB_per_s:.3f} GB/s，重叠 {r.overlap_ns / 1e6:.3f} ms" if enabled else "无 host 背景流量"
        write(directory, f"dataset_{name}.md", f"PCIe 对照：{title}",
              f"单 SM、仅测量 warp 的 lane 0 做 512 MiB HBM pointer chasing。{traffic}。正式样本 4096 个，每样本 8 次串行加载。\n\n平均 **{s.average_ns:.3f} ns**，中位数 {s.median_ns:.0f} ns，P95 {s.p95_ns:.0f} ns，P99 {s.p99_ns:.1f} ns，最大 {s.max_ns} ns。\n\n文件：`{name}_samples.csv`、`{name}_pretest_samples.csv`、`{name}_histogram.csv`、`{name}_histogram_metadata.csv`。")
    write(directory, "dataset_summary.md", "PCIe 对照汇总数据集",
          "`pcie_summary.csv` 包含 baseline 和 host 两组的预测试与正式运行；`pcie_statistics.csv` 包含正式样本的均值、中位数、P95、P99 和最大值。两组各有 4096 个正式样本，512 桶直方图的显示裁剪不影响统计。")
    write(directory, "pcie_latency_histograms.md", "有无 host-read warp 的延迟直方图",
          "对应图：`pcie_latency_histograms.png`、`pcie_latency_histograms.svg`。左图无 host warp，右图有一个持续读取 mapped host memory 的 warp；两图使用相同坐标范围和 512 个线性桶。host 组平均延迟比 baseline 高约 2.1%，主体略右移，没有新增超长尾。\n\n数据说明：`dataset_pcie_baseline.md`、`dataset_pcie_host.md`。")
    write(directory, "RESULTS_INDEX.md", "PCIe 对照文档索引",
          "- [汇总数据](dataset_summary.md)\n- [无 host warp 基线](dataset_pcie_baseline.md)\n- [单 host warp 对照](dataset_pcie_host.md)\n- [对比直方图](pcie_latency_histograms.md)")


def warp_documents(directory):
    summary = pd.read_csv(directory / "warp_summary.csv")
    stats = pd.read_csv(directory / "warp_statistics.csv").set_index("experiment")
    for name, title, lanes in [("warp_host", "32 lane 全部访问 host", 32),
                               ("warp_mixed", "16 lane HBM 与 16 lane host 混合", 16)]:
        s = stats.loc[name]
        write(directory, f"dataset_{name}.md", f"单 warp 数据集：{title}",
              f"仅一个 warp。host 组统计全部 32 lane；mixed 组 lane 0–15 访问 HBM、lane 16–31 访问 host，仅统计前 16 lane。测量循环使用统一加载，没有 lane 控制流分裂。每 lane 4096 个样本，每样本 8 次串行加载。\n\n本组统计 {lanes} lane，共 {lanes * 4096} 个正式样本；平均 **{s.average_ns:.3f} ns**，中位数 {s.median_ns:.0f} ns，P95 {s.p95_ns:.0f} ns，P99 {s.p99_ns:.1f} ns。\n\n文件：`{name}_samples.csv`、`{name}_pretest_samples.csv`、`{name}_histogram.csv`、`{name}_histogram_metadata.csv`；SASS 为 `warp_host_hbm.sass`。")
    write(directory, "dataset_summary.md", "单 warp host/HBM 汇总数据集",
          "`warp_summary.csv` 包含两组预测试和正式测量；`warp_statistics.csv` 包含均值和分位数。原始 CSV 逐 sample、逐 lane 保存，混合组仅包含 lane 0–15。")
    write(directory, "warp_host_hbm_histograms.md", "单 warp host 与混合访问直方图",
          "对应图：`warp_host_hbm_histograms.png`、`warp_host_hbm_histograms.svg`。左图统计 32 个 host lane，右图仅统计混合 warp 的前 16 个 HBM lane。两组均为预测试 min–P99 的 512 个线性桶，使用相同显示范围。混合组 HBM lane 也表现出接近 host 的延迟，因为统一加载需等待较慢 host lane。\n\n数据说明：`dataset_warp_host.md`、`dataset_warp_mixed.md`。")
    write(directory, "RESULTS_INDEX.md", "单 warp host/HBM 文档索引",
          "- [汇总数据](dataset_summary.md)\n- [全部 host lane](dataset_warp_host.md)\n- [HBM/host 混合](dataset_warp_mixed.md)\n- [对比直方图](warp_host_hbm_histograms.md)")


def sweep_documents(directory):
    data = pd.read_csv(directory / "hbm_host_sweep.csv")
    write(directory, "dataset_sweep.md", "单 HBM lane 对 0–31 个 host warp 数据集",
          f"单 SM 中仅测量 warp 的 lane 0 执行完整 512 MiB HBM pointer chasing，扫描 0–31 个 host PCIe warp，共 {len(data)} 个配置。每配置 4096 个样本，每样本 8 次串行加载。\n\n基线平均 {data.iloc[0].average_ns:.3f} ns，31 个 host warp 时 {data.iloc[-1].average_ns:.3f} ns。`hbm_host_sweep.csv` 包含平均/最小/最大延迟和 host 带宽；`hbm_lane_averages.csv` 每配置仅含 lane 0；`full_run.log` 为实际运行日志。")
    write(directory, "hbm_latency_vs_host_warps.md", "HBM lane 0 延迟—host warp 数折线图",
          "对应图：`hbm_latency_vs_host_warps.png`、`hbm_latency_vs_host_warps.svg`。横轴为同一 SM 上的 host PCIe warp 数，纵轴为 HBM lane 0 的平均延迟（ns）。0 个 host warp 是额外基线，1–31 个为正式扫描。数据源为 `hbm_host_sweep.csv`，说明见 `dataset_sweep.md`。")
    write(directory, "RESULTS_INDEX.md", "HBM/host warp 扫描文档索引",
          "- [扫描数据集](dataset_sweep.md)\n- [延迟折线图](hbm_latency_vs_host_warps.md)")


def histogram_documents(directory):
    values = pd.read_csv(directory / "hbm31_samples.csv").latency_ns
    stats = pd.read_csv(directory / "hbm31_statistics.csv").iloc[0]
    meta = pd.read_csv(directory / "hbm31_histogram_metadata.csv").iloc[0]
    write(directory, "dataset_hbm31.md", "31 个 host warp 下的大样本 HBM 数据",
          f"单 SM、31 个 host PCIe warp，仅测量 warp 的 lane 0 执行完整 512 MiB HBM pointer chasing。预测试和正式测量各 **65536 个样本**，每样本 8 次串行加载，正式共 524288 次。\n\n正式平均 **{stats.average_ns:.3f} ns**，中位数 {stats.median_ns:.0f} ns，P95 {stats.p95_ns:.0f} ns，P99 {stats.p99_ns:.0f} ns，最大 {stats.max_ns} ns。host 请求带宽 {stats.host_bandwidth_GB_per_s:.3f} GB/s。\n\n文件：`hbm31_samples.csv`、`hbm31_pretest_samples.csv`、`hbm31_summary.csv`、`hbm31_statistics.csv`、`hbm31_histogram.csv`、`hbm31_histogram_metadata.csv`。")
    write(directory, "hbm31_latency_histogram.md", "31 个 host warp 下的 HBM 延迟直方图",
          f"对应图：`hbm31_latency_histogram.png`、`hbm31_latency_histogram.svg`。图中展示 65536 个正式样本的主体分布，使用预测试 min–P99 的 512 个线性桶，范围 {meta.pretest_min_ns}–{meta.pretest_p99_ns} ns，桶宽 {meta.bin_width_ns:.3f} ns。桶内 {meta.formal_in_range} 个，上下尾分别 {meta.formal_below_min}、{meta.formal_above_p99} 个；原始样本不裁剪。\n\n数据说明：`dataset_hbm31.md`。")
    write(directory, "RESULTS_INDEX.md", "31 host warp 大样本实验文档索引",
          "- [大样本数据集](dataset_hbm31.md)\n- [HBM 延迟直方图](hbm31_latency_histogram.md)")


def hbm01_documents(directory):
    stats = pd.read_csv(directory / "hbm_0_1_statistics.csv").set_index("host_warps")
    for count in [0, 1]:
        s = stats.loc[count]
        meta = pd.read_csv(directory / f"hbm{count}_histogram_metadata.csv").iloc[0]
        structure = "无 host 背景流量" if count == 0 else "一个 host warp 连续读取 mapped host memory，不执行 pointer chasing"
        write(directory, f"dataset_hbm{count}.md", f"HBM lane 0：{count} 个 host warp 数据集",
              f"单 SM 中仅测量 warp 的 lane 0 执行完整 512 MiB HBM pointer chasing；{structure}。预测试和正式测量各 65536 个样本，每样本 8 次串行加载。\n\n正式平均 **{s.average_ns:.3f} ns**，中位数 {s.median_ns:.0f} ns，P95 {s.p95_ns:.0f} ns，P99 {s.p99_ns:.0f} ns，最大 {s.max_ns} ns。host 请求带宽 {s.host_bandwidth_GB_per_s:.3f} GB/s。\n\n直方图范围 {meta.pretest_min_ns}–{meta.pretest_p99_ns} ns，512 个线性桶，桶宽 {meta.bin_width_ns:.3f} ns。文件：`hbm{count}_samples.csv`、`hbm{count}_pretest_samples.csv`、`hbm{count}_histogram.csv`、`hbm{count}_histogram_metadata.csv`。")
    write(directory, "dataset_summary.md", "0/1 host warp 大样本汇总数据集",
          "`hbm_0_1_summary.csv` 包含 0 和 1 个 host warp 两组的预测试及正式运行；`hbm_0_1_statistics.csv` 包含每组 65536 个正式样本的均值、中位数、P95、P99 和最大值。host warp 仅做连续 host 读取。")
    write(directory, "hbm_0_1_latency_histograms.md", "0 与 1 个 host warp 的 HBM 延迟直方图",
          "对应图：`hbm_0_1_latency_histograms.png`、`hbm_0_1_latency_histograms.svg`。左图无 host warp，右图有一个连续读取 host memory 的 warp；host warp 不是 pointer chasing。两组均采集 65536 个正式样本，使用预测试 min–P99 的 512 个线性桶和相同显示坐标。数据说明：`dataset_hbm0.md`、`dataset_hbm1.md`。")
    write(directory, "RESULTS_INDEX.md", "0/1 host warp 大样本实验文档索引",
          "- [汇总数据](dataset_summary.md)\n- [0 host warp 数据集](dataset_hbm0.md)\n- [1 host warp 数据集](dataset_hbm1.md)\n- [对比直方图](hbm_0_1_latency_histograms.md)")


for directory, device in [(ROOT / "results", "RTX 5080"), (ROOT / "results_b300", "NVIDIA B300")]:
    if (directory / "summary.csv").exists():
        main_documents(directory, device)
if (ROOT / "results_b300/pcie/pcie_summary.csv").exists():
    pcie_documents(ROOT / "results_b300/pcie")
if (ROOT / "results_b300/warp_host_hbm/warp_summary.csv").exists():
    warp_documents(ROOT / "results_b300/warp_host_hbm")
if (ROOT / "results_b300/hbm_host_sweep/hbm_host_sweep.csv").exists():
    sweep_documents(ROOT / "results_b300/hbm_host_sweep")
if (ROOT / "results_b300/hbm_host_histogram/hbm31_summary.csv").exists():
    histogram_documents(ROOT / "results_b300/hbm_host_histogram")
if (ROOT / "results_b300/hbm_host_0_1_histogram/hbm_0_1_summary.csv").exists():
    hbm01_documents(ROOT / "results_b300/hbm_host_0_1_histogram")

index = ROOT / "RESULTS_DOCUMENTATION.md"
index.write_text("""# 实验图与数据集文档总索引

- [RTX 5080 TMA 实验](results/RESULTS_INDEX.md)
- [B300 TMA 实验](results_b300/RESULTS_INDEX.md)
- [B300 单 SM PCIe 对照](results_b300/pcie/RESULTS_INDEX.md)
- [B300 单 warp host/HBM 混合实验](results_b300/warp_host_hbm/RESULTS_INDEX.md)
- [B300 单 HBM lane 对 0–31 host warp 扫描](results_b300/hbm_host_sweep/RESULTS_INDEX.md)
- [B300 31 host warp 大样本 HBM 直方图](results_b300/hbm_host_histogram/RESULTS_INDEX.md)
- [B300 0/1 host warp 大样本 HBM 直方图](results_b300/hbm_host_0_1_histogram/RESULTS_INDEX.md)

每张 PNG/SVG 均有同名 Markdown 说明；每组数据均有独立 `dataset_*.md` 或汇总说明。说明包含实验结构、关键结果、样本数、坐标或分桶设置及对应数据文件。
""", encoding="utf-8")
created.append(index)
print(f"Created {len(created)} documentation files")
