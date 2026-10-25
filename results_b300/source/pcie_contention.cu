#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kWarmupLoads = 128;
constexpr uint32_t kLoadsPerSample = 8;
constexpr uint64_t kSleepNs = 5000;
constexpr uint32_t kBins = 512;
constexpr uint32_t kWarpBytes = 128;
constexpr uint64_t kHostLaneStride = 16384;
constexpr uint32_t kHostBatch = 64;

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t status_ = (call);                                                          \
        if (status_ != cudaSuccess) {                                                          \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(status_)); \
        }                                                                                      \
    } while (false)

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
    return value;
}

__device__ __forceinline__ uint64_t globaltimer_after(uint32_t& value) {
    uint64_t time;
    asm volatile("add.u32 %1, %1, 1; mov.u64 %0, %%globaltimer; sub.u32 %1, %1, 1;"
                 : "=l"(time), "+r"(value) :: "memory");
    return time;
}

__device__ __forceinline__ uint32_t load_cv(const uint32_t* address) {
    uint32_t value;
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ uint64_t load_cv_u64(const uint64_t* address) {
    uint64_t value;
    asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ void store_cv_u64(uint64_t* address, uint64_t value) {
    asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(address), "l"(value) : "memory");
}

struct HostStats {
    uint64_t first_ns;
    uint64_t last_ns;
    uint64_t iterations;
    uint64_t checksum;
    uint32_t smid;
};

struct MeasureStats {
    uint64_t wait_start_ns;
    uint64_t wait_end_ns;
    uint64_t chase_start_ns;
    uint64_t chase_end_ns;
    uint64_t progress_before;
    uint64_t progress_after;
    uint64_t final_index;
};

__global__ void timer_resolution_kernel(uint64_t* result) {
    uint64_t previous = globaltimer();
    uint64_t minimum = ~uint64_t{0};
    for (int i = 0; i < 10000; ++i) {
        uint64_t current = globaltimer();
        uint64_t delta = current - previous;
        if (delta != 0 && delta < minimum) minimum = delta;
        previous = current;
    }
    *result = minimum;
}

__global__ void __launch_bounds__(64, 1) pcie_kernel(
    const uint32_t* chain, uint32_t chain_start, const uint32_t* host_source,
    uint64_t host_bytes, uint32_t samples, uint64_t* latencies, uint64_t* progress,
    uint32_t* done, HostStats* host_stats, MeasureStats* measure_stats,
    uint32_t host_enabled) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    uint32_t index = chain_start;
    if (warp == 0 && lane == 0) {
        for (uint32_t i = 0; i < kWarmupLoads; ++i) index = load_cv(chain + index);
        host_stats->smid = [] {
            uint32_t value;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
            return value;
        }();
    }
    __syncthreads();

    if (host_enabled && warp == 1) {
        if (lane == 0) host_stats->first_ns = globaltimer();
        uint64_t completed = 0;
        uint64_t checksum = 0;
        bool stop = false;
        while (!stop) {
            uint32_t values[kHostBatch];
            #pragma unroll
            for (uint32_t batch = 0; batch < kHostBatch; ++batch) {
                uint64_t byte_offset = ((completed + batch) * kWarpBytes +
                                        lane * kHostLaneStride) % host_bytes;
                values[batch] = load_cv(reinterpret_cast<const uint32_t*>(
                    reinterpret_cast<const unsigned char*>(host_source) + byte_offset));
            }
            #pragma unroll
            for (uint32_t batch = 0; batch < kHostBatch; ++batch) checksum += values[batch];
            completed += kHostBatch;
            if (lane == 0) {
                store_cv_u64(progress, completed);
                stop = *done != 0;
            }
            stop = __shfl_sync(0xffffffff, stop, 0);
        }
        for (uint32_t offset = 16; offset != 0; offset >>= 1) {
            checksum += __shfl_down_sync(0xffffffff, checksum, offset);
        }
        if (lane == 0) {
            host_stats->last_ns = globaltimer();
            host_stats->iterations = completed;
            host_stats->checksum = checksum;
        }
    }

    if (warp == 0 && lane == 0) {
        MeasureStats stats{};
        stats.wait_start_ns = globaltimer();
        uint64_t now;
        do {
            now = globaltimer();
        } while (now - stats.wait_start_ns < kSleepNs);
        stats.wait_end_ns = now;
        index = load_cv(chain + index);
        stats.chase_start_ns = globaltimer_after(index);
        stats.progress_before = load_cv_u64(progress);
        uint64_t warmup_before = globaltimer();
        index = load_cv(chain + index);
        uint64_t warmup_after = globaltimer_after(index);
        if (warmup_after < warmup_before) latencies[0] = 0;
        for (uint32_t sample = 0; sample < samples; ++sample) {
            uint64_t before = globaltimer();
            for (uint32_t load = 0; load < kLoadsPerSample; ++load) {
                index = load_cv(chain + index);
            }
            uint64_t after = globaltimer_after(index);
            latencies[sample] = (after - before) / kLoadsPerSample;
        }
        stats.progress_after = load_cv_u64(progress);
        stats.chase_end_ns = globaltimer_after(index);
        stats.final_index = index;
        __threadfence();
        atomicExch(done, 1);
        *measure_stats = stats;
    }
    __syncthreads();
}

struct RunResult {
    std::string name;
    uint32_t host_enabled = 0;
    uint32_t samples = 0;
    double average_ns = 0;
    uint64_t minimum_ns = 0;
    uint64_t maximum_ns = 0;
    uint64_t wait_ns = 0;
    uint64_t chase_ns = 0;
    uint64_t overlap_ns = 0;
    uint64_t host_iterations = 0;
    uint64_t host_total_bytes = 0;
    uint64_t host_concurrent_bytes = 0;
    double host_total_bandwidth_gbs = 0;
    double host_concurrent_bandwidth_gbs = 0;
    uint64_t host_checksum = 0;
    uint32_t smid = 0;
    std::vector<uint64_t> latencies;
};

void write_samples(const std::string& path, const std::vector<uint64_t>& values) {
    std::ofstream output(path);
    output << "sample,latency_ns\n";
    for (size_t i = 0; i < values.size(); ++i) output << i << ',' << values[i] << '\n';
}

void build_histogram(const std::string& path, const RunResult& formal,
                     const RunResult& pretest) {
    std::vector<uint64_t> sorted = pretest.latencies;
    std::sort(sorted.begin(), sorted.end());
    uint64_t lower = sorted.front();
    size_t index = static_cast<size_t>(std::ceil(0.99 * (sorted.size() - 1)));
    uint64_t p99 = sorted[index];
    uint64_t upper = p99 + 1;
    if (lower == 0 || upper <= lower) throw std::runtime_error("Invalid histogram range");
    double width = static_cast<double>(upper - lower) / kBins;
    std::vector<uint64_t> counts(kBins, 0);
    uint64_t below = 0;
    uint64_t above = 0;
    for (uint64_t latency : formal.latencies) {
        if (latency < lower) ++below;
        else if (latency >= upper) ++above;
        else ++counts[static_cast<uint32_t>((latency - lower) / width)];
    }
    std::ofstream output(path);
    output << "bin,lower_ns,upper_ns,count\n";
    for (uint32_t bin = 0; bin < kBins; ++bin) {
        output << bin << ',' << std::fixed << std::setprecision(6) << lower + bin * width << ','
               << lower + (bin + 1) * width << ',' << counts[bin] << '\n';
    }
    std::ofstream metadata(path.substr(0, path.size() - 4) + "_metadata.csv");
    metadata << "bins,pretest_min_ns,pretest_p99_ns,pretest_max_ns,bin_width_ns,"
                "formal_in_range,formal_below_min,formal_above_p99\n";
    metadata << kBins << ',' << lower << ',' << p99 << ',' << pretest.maximum_ns << ','
             << std::setprecision(17) << width << ',' << formal.latencies.size() - below - above
             << ',' << below << ',' << above << '\n';
}

class Benchmark {
  public:
    Benchmark(uint32_t samples, uint32_t chain_mb, uint32_t host_mb)
        : samples_(samples), chain_count_(static_cast<uint64_t>(chain_mb) * 1024 * 1024 / 4),
          host_bytes_(static_cast<uint64_t>(host_mb) * 1024 * 1024) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (!prop.canMapHostMemory) throw std::runtime_error("Mapped host memory unsupported");
        std::cout << "GPU=" << prop.name << " sm_count=" << prop.multiProcessorCount
                  << " l2_bytes=" << prop.l2CacheSize << '\n';
        CUDA_CHECK(cudaMalloc(&chain_, chain_count_ * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocManaged(&latencies_, samples_ * sizeof(uint64_t)));
        CUDA_CHECK(cudaMallocManaged(&progress_, sizeof(uint64_t)));
        CUDA_CHECK(cudaMallocManaged(&done_, sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocManaged(&host_stats_, sizeof(HostStats)));
        CUDA_CHECK(cudaMallocManaged(&measure_stats_, sizeof(MeasureStats)));
        CUDA_CHECK(cudaHostAlloc(&host_source_, host_bytes_, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&device_host_source_, host_source_, 0));
        std::memset(host_source_, 0x5a, host_bytes_);
        initialize_chain();
        timer_resolution_ns_ = timer_resolution();
        std::cout << "mapped_host_bytes=" << host_bytes_
                  << " globaltimer_resolution_ns=" << timer_resolution_ns_ << '\n';
    }

    ~Benchmark() {
        cudaFree(chain_);
        cudaFree(latencies_);
        cudaFree(progress_);
        cudaFree(done_);
        cudaFree(host_stats_);
        cudaFree(measure_stats_);
        cudaFreeHost(host_source_);
    }

    RunResult run(const std::string& name, bool host_enabled) {
        *progress_ = 0;
        *done_ = 0;
        *host_stats_ = HostStats{};
        *measure_stats_ = MeasureStats{};
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t chain_start = static_cast<uint32_t>(
            (run_serial_++ * 1048576ull) & (chain_count_ - 1));
        pcie_kernel<<<1, host_enabled ? 64 : 32>>>(
            chain_, chain_start, device_host_source_, host_bytes_, samples_, latencies_, progress_,
            done_, host_stats_, measure_stats_, host_enabled ? 1 : 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        RunResult result;
        result.name = name;
        result.host_enabled = host_enabled;
        result.samples = samples_;
        result.latencies.assign(latencies_, latencies_ + samples_);
        auto extrema = std::minmax_element(result.latencies.begin(), result.latencies.end());
        result.minimum_ns = *extrema.first;
        result.maximum_ns = *extrema.second;
        result.average_ns = static_cast<double>(std::accumulate(
            result.latencies.begin(), result.latencies.end(), uint64_t{0})) / samples_;
        result.wait_ns = measure_stats_->wait_end_ns - measure_stats_->wait_start_ns;
        result.chase_ns = measure_stats_->chase_end_ns - measure_stats_->chase_start_ns;
        result.smid = host_stats_->smid;
        if (host_enabled) {
            result.host_iterations = host_stats_->iterations;
            result.host_total_bytes = result.host_iterations * kWarpBytes;
            result.host_concurrent_bytes =
                (measure_stats_->progress_after - measure_stats_->progress_before) * kWarpBytes;
            uint64_t host_ns = host_stats_->last_ns - host_stats_->first_ns;
            result.host_total_bandwidth_gbs = static_cast<double>(result.host_total_bytes) / host_ns;
            result.host_concurrent_bandwidth_gbs =
                static_cast<double>(result.host_concurrent_bytes) / result.chase_ns;
            uint64_t overlap_begin = std::max(host_stats_->first_ns, measure_stats_->chase_start_ns);
            uint64_t overlap_end = std::min(host_stats_->last_ns, measure_stats_->chase_end_ns);
            if (overlap_end > overlap_begin) result.overlap_ns = overlap_end - overlap_begin;
            result.host_checksum = host_stats_->checksum;
            if (!result.overlap_ns || !result.host_concurrent_bytes || !result.host_checksum) {
                throw std::runtime_error("Host-read warp did not overlap measurement correctly");
            }
        }
        std::cout << name << " avg_ns=" << result.average_ns << " min_ns=" << result.minimum_ns
                  << " max_ns=" << result.maximum_ns << " wait_ns=" << result.wait_ns;
        if (host_enabled) {
            std::cout << " host_read_GB/s=" << result.host_concurrent_bandwidth_gbs
                      << " overlap_ns=" << result.overlap_ns;
        }
        std::cout << " smid=" << result.smid << '\n';
        write_samples(output_ + "/" + name + "_samples.csv", result.latencies);
        return result;
    }

    uint64_t timer_resolution_ns() const { return timer_resolution_ns_; }
    void set_output(const std::string& output) { output_ = output; }

  private:
    void initialize_chain() {
        std::vector<uint32_t> order(chain_count_);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937_64 random(0xb300c1e);
        std::shuffle(order.begin(), order.end(), random);
        std::vector<uint32_t> chain(chain_count_);
        for (uint64_t i = 0; i < chain_count_; ++i) chain[order[i]] = order[(i + 1) % chain_count_];
        CUDA_CHECK(cudaMemcpy(chain_, chain.data(), chain_count_ * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    }

    uint64_t timer_resolution() {
        uint64_t* value;
        CUDA_CHECK(cudaMallocManaged(&value, sizeof(uint64_t)));
        timer_resolution_kernel<<<1, 1>>>(value);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t result = *value;
        cudaFree(value);
        return result;
    }

    uint32_t samples_;
    uint64_t run_serial_ = 0;
    uint64_t chain_count_;
    uint64_t host_bytes_;
    uint32_t* chain_ = nullptr;
    uint64_t* latencies_ = nullptr;
    uint64_t* progress_ = nullptr;
    uint32_t* done_ = nullptr;
    HostStats* host_stats_ = nullptr;
    MeasureStats* measure_stats_ = nullptr;
    void* host_source_ = nullptr;
    uint32_t* device_host_source_ = nullptr;
    uint64_t timer_resolution_ns_ = 0;
    std::string output_ = "results";
};

void write_summary(const std::string& path, const std::vector<RunResult>& results,
                   uint64_t timer_resolution) {
    std::ofstream output(path);
    output << "experiment,host_enabled,samples,average_ns,min_ns,max_ns,wait_ns,chase_ns,"
              "overlap_ns,host_iterations,host_total_bytes,host_concurrent_bytes,"
              "host_total_bandwidth_GB_per_s,host_concurrent_bandwidth_GB_per_s,"
              "host_checksum,smid,globaltimer_resolution_ns\n";
    for (const auto& result : results) {
        output << result.name << ',' << result.host_enabled << ',' << result.samples << ','
               << std::setprecision(17) << result.average_ns << ',' << result.minimum_ns << ','
               << result.maximum_ns << ',' << result.wait_ns << ',' << result.chase_ns << ','
               << result.overlap_ns << ',' << result.host_iterations << ','
               << result.host_total_bytes << ',' << result.host_concurrent_bytes << ','
               << result.host_total_bandwidth_gbs << ',' << result.host_concurrent_bandwidth_gbs
               << ',' << result.host_checksum << ',' << result.smid << ',' << timer_resolution
               << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        uint32_t samples = 4096;
        uint32_t chain_mb = 512;
        uint32_t host_mb = 256;
        std::string output = "results";
        for (int i = 1; i < argc; ++i) {
            std::string argument = argv[i];
            if (++i >= argc) throw std::runtime_error("Missing value after " + argument);
            std::string value = argv[i];
            if (argument == "--samples") samples = std::stoul(value);
            else if (argument == "--chain-mb") chain_mb = std::stoul(value);
            else if (argument == "--host-mb") host_mb = std::stoul(value);
            else if (argument == "--output") output = value;
            else throw std::runtime_error("Unknown argument: " + argument);
        }
        Benchmark benchmark(samples, chain_mb, host_mb);
        benchmark.set_output(output);
        RunResult baseline_pretest = benchmark.run("pcie_baseline_pretest", false);
        RunResult baseline = benchmark.run("pcie_baseline", false);
        build_histogram(output + "/pcie_baseline_histogram.csv", baseline, baseline_pretest);
        RunResult host_pretest = benchmark.run("pcie_host_pretest", true);
        RunResult host = benchmark.run("pcie_host", true);
        build_histogram(output + "/pcie_host_histogram.csv", host, host_pretest);
        write_summary(output + "/pcie_summary.csv",
                      {baseline_pretest, baseline, host_pretest, host},
                      benchmark.timer_resolution_ns());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
