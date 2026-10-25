#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kLanes = 32;
constexpr uint32_t kWarmupLoads = 128;
constexpr uint32_t kLoadsPerSample = 8;
constexpr uint32_t kSamples = 4096;
constexpr uint32_t kHostBatch = 16;
constexpr uint32_t kWarpBytes = 128;
constexpr uint64_t kHostLaneStride = 16384;
constexpr uint64_t kHostWarpStride = 1048576;
constexpr uint64_t kHostBytes = 256ull * 1024 * 1024;
constexpr uint64_t kChainBytes = 512ull * 1024 * 1024;
constexpr uint64_t kSleepNs = 5000;

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

struct HostStats {
    uint64_t first_ns;
    uint64_t last_ns;
    uint64_t iterations;
    uint64_t checksum;
};

struct MeasureStats {
    uint64_t chase_start_ns;
    uint64_t chase_end_ns;
    uint64_t progress_before;
    uint64_t progress_after;
    uint64_t wait_actual_ns;
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

__global__ void __launch_bounds__(1024, 1) sweep_kernel(
    const uint32_t* hbm_chain, uint32_t slice_count, uint32_t start,
    const uint32_t* host_source, uint32_t host_warps, uint64_t* latencies,
    uint64_t* final_indices, uint64_t* progress, uint32_t* done, HostStats* host_stats,
    MeasureStats* measure_stats, uint32_t* smid_result) {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    if (warp == 0 && lane == 0) {
        const uint32_t* base = hbm_chain;
        uint32_t index = start;
        for (uint32_t i = 0; i < kWarmupLoads; ++i) index = load_cv(base + index);
        latencies[lane] = index;
    }
    __syncthreads();

    if (warp > 0 && warp <= host_warps) {
        if (lane == 0) host_stats[warp - 1].first_ns = globaltimer();
        uint64_t completed = 0;
        uint64_t checksum = 0;
        bool stop = false;
        while (!stop) {
            uint32_t values[kHostBatch];
            #pragma unroll
            for (uint32_t batch = 0; batch < kHostBatch; ++batch) {
                uint64_t byte_offset = ((completed + batch) * kWarpBytes +
                    lane * kHostLaneStride + (warp - 1) * kHostWarpStride) % kHostBytes;
                values[batch] = load_cv(reinterpret_cast<const uint32_t*>(
                    reinterpret_cast<const unsigned char*>(host_source) + byte_offset));
            }
            #pragma unroll
            for (uint32_t batch = 0; batch < kHostBatch; ++batch) checksum += values[batch];
            completed += kHostBatch;
            if (lane == 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(progress), 1ull);
                stop = *done != 0;
            }
            stop = __shfl_sync(0xffffffff, stop, 0);
        }
        for (uint32_t offset = 16; offset != 0; offset >>= 1) {
            checksum += __shfl_down_sync(0xffffffff, checksum, offset);
        }
        if (lane == 0) {
            host_stats[warp - 1].last_ns = globaltimer();
            host_stats[warp - 1].iterations = completed;
            host_stats[warp - 1].checksum = checksum;
        }
    }

    if (warp == 0 && lane == 0) {
        const uint32_t* base = hbm_chain;
        uint32_t index = static_cast<uint32_t>(latencies[0]);
        uint64_t wait_start = globaltimer();
        uint64_t now;
        do {
            now = globaltimer();
        } while (now - wait_start < kSleepNs);
        index = load_cv(base + index);
        measure_stats->wait_actual_ns = globaltimer_after(index) - wait_start;
        measure_stats->chase_start_ns = globaltimer_after(index);
        measure_stats->progress_before = *progress;
        *smid_result = [] {
            uint32_t value;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
            return value;
        }();
        for (uint32_t sample = 0; sample < kSamples; ++sample) {
            uint64_t before = globaltimer();
            #pragma unroll
            for (uint32_t load = 0; load < kLoadsPerSample; ++load) {
                index = load_cv(base + index);
            }
            uint64_t after = globaltimer_after(index);
            latencies[sample * kLanes] = (after - before) / kLoadsPerSample;
        }
        final_indices[0] = index;
        measure_stats->progress_after = *progress;
        measure_stats->chase_end_ns = globaltimer_after(index);
        __threadfence();
        atomicExch(done, 1);
    }
    __syncthreads();
}

struct RunResult {
    uint32_t host_warps = 0;
    double average_ns = 0;
    uint64_t minimum_ns = 0;
    uint64_t maximum_ns = 0;
    uint64_t wait_ns = 0;
    uint64_t chase_ns = 0;
    uint64_t overlap_ns = 0;
    uint64_t host_bytes = 0;
    double host_bandwidth_gbs = 0;
    uint64_t host_iterations = 0;
    uint32_t smid = 0;
    std::vector<double> lane_averages;
};

class Benchmark {
  public:
    Benchmark() : slice_count_(kChainBytes / sizeof(uint32_t)) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (!prop.canMapHostMemory) throw std::runtime_error("Mapped host memory unsupported");
        std::cout << "GPU=" << prop.name << " sm_count=" << prop.multiProcessorCount << '\n';
        CUDA_CHECK(cudaMalloc(&chain_, kChainBytes));
        CUDA_CHECK(cudaHostAlloc(&host_source_, kHostBytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&device_host_source_, host_source_, 0));
        std::memset(host_source_, 0x5a, kHostBytes);
        CUDA_CHECK(cudaMallocManaged(&latencies_, kSamples * kLanes * sizeof(uint64_t)));
        CUDA_CHECK(cudaMallocManaged(&final_indices_, kLanes * sizeof(uint64_t)));
        CUDA_CHECK(cudaMallocManaged(&progress_, sizeof(uint64_t)));
        CUDA_CHECK(cudaMallocManaged(&done_, sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocManaged(&host_stats_, 32 * sizeof(HostStats)));
        CUDA_CHECK(cudaMallocManaged(&measure_stats_, sizeof(MeasureStats)));
        CUDA_CHECK(cudaMallocManaged(&smid_, sizeof(uint32_t)));
        initialize_chain();
        timer_resolution_ns_ = timer_resolution();
        std::cout << "globaltimer_resolution_ns=" << timer_resolution_ns_ << '\n';
    }

    ~Benchmark() {
        cudaFree(chain_);
        cudaFreeHost(host_source_);
        cudaFree(latencies_);
        cudaFree(final_indices_);
        cudaFree(progress_);
        cudaFree(done_);
        cudaFree(host_stats_);
        cudaFree(measure_stats_);
        cudaFree(smid_);
    }

    RunResult run(uint32_t host_warps) {
        *progress_ = 0;
        *done_ = 0;
        *measure_stats_ = MeasureStats{};
        for (uint32_t i = 0; i < 32; ++i) host_stats_[i] = HostStats{};
        uint32_t start = (run_serial_++ * 131072u) % slice_count_;
        CUDA_CHECK(cudaDeviceSynchronize());
        sweep_kernel<<<1, (host_warps + 1) * 32>>>(
            chain_, slice_count_, start, device_host_source_, host_warps, latencies_, final_indices_,
            progress_, done_, host_stats_, measure_stats_, smid_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        RunResult result;
        result.host_warps = host_warps;
        result.wait_ns = measure_stats_->wait_actual_ns;
        result.chase_ns = measure_stats_->chase_end_ns - measure_stats_->chase_start_ns;
        result.smid = *smid_;
        uint64_t total = 0;
        result.minimum_ns = ~uint64_t{0};
        for (uint32_t sample = 0; sample < kSamples; ++sample) {
            uint64_t latency = latencies_[sample * kLanes];
            total += latency;
            result.minimum_ns = std::min(result.minimum_ns, latency);
            result.maximum_ns = std::max(result.maximum_ns, latency);
        }
        result.lane_averages.push_back(static_cast<double>(total) / kSamples);
        result.average_ns = static_cast<double>(total) / kSamples;
        if (host_warps) {
            uint64_t first = ~uint64_t{0};
            uint64_t last = 0;
            for (uint32_t warp = 0; warp < host_warps; ++warp) {
                const auto& stats = host_stats_[warp];
                if (!stats.iterations || !stats.checksum) {
                    throw std::runtime_error("A host warp exited early or produced no traffic");
                }
                first = std::min(first, stats.first_ns);
                last = std::max(last, stats.last_ns);
                result.host_iterations += stats.iterations;
            }
            result.host_bytes = (measure_stats_->progress_after - measure_stats_->progress_before) *
                                kHostBatch * kWarpBytes;
            result.host_bandwidth_gbs = static_cast<double>(result.host_bytes) / result.chase_ns;
            uint64_t overlap_begin = std::max(first, measure_stats_->chase_start_ns);
            uint64_t overlap_end = std::min(last, measure_stats_->chase_end_ns);
            if (overlap_end > overlap_begin) result.overlap_ns = overlap_end - overlap_begin;
            if (!result.overlap_ns || !result.host_bytes) {
                throw std::runtime_error("No host/HBM concurrency");
            }
        }
        std::cout << "host_warps=" << host_warps << " avg_ns=" << result.average_ns
                  << " min_ns=" << result.minimum_ns << " max_ns=" << result.maximum_ns
                  << " host_read_GB/s=" << result.host_bandwidth_gbs
                  << " overlap_ns=" << result.overlap_ns << " smid=" << result.smid << '\n';
        return result;
    }

    uint64_t timer_resolution_ns() const { return timer_resolution_ns_; }

  private:
    void initialize_chain() {
        std::vector<uint32_t> order(slice_count_);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937_64 random(0xb3005);
        std::shuffle(order.begin(), order.end(), random);
        std::vector<uint32_t> chain(slice_count_);
        for (uint32_t i = 0; i < slice_count_; ++i) chain[order[i]] = order[(i + 1) % slice_count_];
        CUDA_CHECK(cudaMemcpy(chain_, chain.data(), kChainBytes, cudaMemcpyHostToDevice));
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

    uint32_t slice_count_;
    uint32_t run_serial_ = 0;
    uint32_t* chain_ = nullptr;
    void* host_source_ = nullptr;
    uint32_t* device_host_source_ = nullptr;
    uint64_t* latencies_ = nullptr;
    uint64_t* final_indices_ = nullptr;
    uint64_t* progress_ = nullptr;
    uint32_t* done_ = nullptr;
    HostStats* host_stats_ = nullptr;
    MeasureStats* measure_stats_ = nullptr;
    uint32_t* smid_ = nullptr;
    uint64_t timer_resolution_ns_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string output = argc > 1 ? argv[1] : "results";
        Benchmark benchmark;
        std::vector<RunResult> results;
        std::ofstream lanes(output + "/hbm_lane_averages.csv");
        lanes << "host_warps,lane,average_ns\n";
        for (uint32_t host_warps = 0; host_warps <= 31; ++host_warps) {
            RunResult result = benchmark.run(host_warps);
            for (uint32_t lane = 0; lane < result.lane_averages.size(); ++lane) {
                lanes << host_warps << ',' << lane << ',' << std::setprecision(17)
                      << result.lane_averages[lane] << '\n';
            }
            results.push_back(std::move(result));
        }
        std::ofstream summary(output + "/hbm_host_sweep.csv");
        summary << "host_warps,average_ns,min_ns,max_ns,wait_ns,chase_ns,overlap_ns,"
                   "host_bytes,host_bandwidth_GB_per_s,host_iterations,smid,"
                   "globaltimer_resolution_ns\n";
        for (const auto& result : results) {
            summary << result.host_warps << ',' << std::setprecision(17) << result.average_ns << ','
                    << result.minimum_ns << ',' << result.maximum_ns << ',' << result.wait_ns << ','
                    << result.chase_ns << ',' << result.overlap_ns << ',' << result.host_bytes << ','
                    << result.host_bandwidth_gbs << ',' << result.host_iterations << ','
                    << result.smid << ',' << benchmark.timer_resolution_ns() << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
