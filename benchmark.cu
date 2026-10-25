#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

namespace {

#ifndef TMA_TILE_BYTES
#define TMA_TILE_BYTES 16384
#endif
constexpr uint32_t kTileBytes = TMA_TILE_BYTES;
#ifndef TMA_SLOTS
#define TMA_SLOTS 4
#endif
constexpr uint32_t kSlots = TMA_SLOTS;
constexpr uint32_t kWarmupLoads = 128;
constexpr uint32_t kLoadsPerSample = 8;
constexpr uint64_t kSleepNs = 5000;
constexpr uint32_t kHistogramBins = 512;

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t status_ = (call);                                                        \
        if (status_ != cudaSuccess) {                                                        \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(status_)); \
        }                                                                                    \
    } while (false)

#define CU_CHECK(call)                                                                       \
    do {                                                                                     \
        CUresult status_ = (call);                                                           \
        if (status_ != CUDA_SUCCESS) {                                                       \
            const char* message_ = nullptr;                                                  \
            cuGetErrorString(status_, &message_);                                            \
            throw std::runtime_error(std::string(#call) + ": " +                           \
                                     (message_ ? message_ : "unknown CUDA driver error"));    \
        }                                                                                    \
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

__device__ __forceinline__ uint32_t chase_load(const uint32_t* address) {
    uint32_t value;
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ uint32_t smid() {
    uint32_t value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier) {
    uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(address) : "memory");
}

__device__ __forceinline__ void mbarrier_init_fence() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void tma_issue(const CUtensorMap* map, uint32_t destination,
                                          uint64_t* barrier, uint32_t offset) {
    uint32_t barrier_address = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%2], [%3, {0, %4}], [%0];\n\t"
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(barrier_address), "r"(kTileBytes), "r"(destination), "l"(map), "r"(offset / 256)
        : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* barrier, uint32_t parity) {
    uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "wait_loop:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@!p bra wait_loop;\n\t"
        "}"
        : "+r"(address), "+r"(parity) :: "memory");
}

__device__ __forceinline__ void progress_store(uint64_t* progress, uint64_t value) {
    asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(progress), "l"(value) : "memory");
}

__device__ __forceinline__ uint64_t progress_load(const uint64_t* progress) {
    uint64_t value;
    asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(value) : "l"(progress) : "memory");
    return value;
}

struct DeviceBlockStats {
    uint32_t smid;
    uint32_t tma_enabled;
    uint64_t start_ns;
    uint64_t end_ns;
    uint64_t tma_first_ns;
    uint64_t tma_last_ns;
    uint64_t tma_bytes;
    uint64_t tma_iterations;
    uint64_t checksum;
};

struct DeviceMeasureStats {
    uint64_t block_start_ns;
    uint64_t wait_start_ns;
    uint64_t wait_end_ns;
    uint64_t chase_start_ns;
    uint64_t chase_end_ns;
    uint64_t progress_at_start;
    uint64_t progress_at_end;
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

__global__ void __launch_bounds__(64, 1) benchmark_kernel(
    const __grid_constant__ CUtensorMap tensor_map, const uint32_t* chain,
    uint32_t chain_mask, uint32_t samples, uint64_t* latencies,
    DeviceBlockStats* block_stats, DeviceMeasureStats* measure_stats,
    uint64_t* progress, uint32_t* arrival, uint32_t total_blocks,
    uint32_t tma_blocks, uint32_t tma_iterations, uint32_t source_chunk_tiles,
    uint32_t concurrent_mode, uint32_t measure_enabled) {
    extern __shared__ __align__(16) unsigned char shared_storage[];
    uint64_t* barriers = reinterpret_cast<uint64_t*>(shared_storage + kSlots * kTileBytes);
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t tma_index = concurrent_mode ? blockIdx.x : blockIdx.x - 1;
    const bool block_has_tma = concurrent_mode ? blockIdx.x < tma_blocks
                                                : blockIdx.x > 0 && blockIdx.x <= tma_blocks;
    const bool tma_warp = warp == 0 && block_has_tma;
    const bool measure_warp = blockIdx.x == 0 && measure_enabled &&
                              warp == (concurrent_mode ? 1u : 0u);
    uint32_t index = blockIdx.x * 2654435761u & chain_mask;

    if (measure_warp && lane == 0) {
        for (uint32_t i = 0; i < kWarmupLoads; ++i) index = chase_load(chain + index);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        block_stats[blockIdx.x].smid = smid();
        block_stats[blockIdx.x].tma_enabled = block_has_tma;
        __threadfence();
        atomicAdd(arrival, 1);
        while (atomicAdd(arrival, 0) < total_blocks) {
        }
    }
    __syncthreads();
    const uint64_t start = globaltimer();
    if (threadIdx.x == 0) block_stats[blockIdx.x].start_ns = start;

    if (tma_warp) {
        if (lane == 0) {
            for (uint32_t slot = 0; slot < kSlots; ++slot) mbarrier_init(barriers + slot);
            mbarrier_init_fence();
            block_stats[blockIdx.x].tma_first_ns = globaltimer();
        }
        __syncwarp();
        uint32_t destination[kSlots];
        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            destination[slot] = static_cast<uint32_t>(
                __cvta_generic_to_shared(shared_storage + slot * kTileBytes));
        }
        if (lane == 0) {
            for (uint32_t slot = 0; slot < kSlots && slot < tma_iterations; ++slot) {
                uint32_t tile = (tma_index * 977u + slot) % source_chunk_tiles;
                uint32_t offset = (tma_index * source_chunk_tiles + tile) * kTileBytes;
                tma_issue(&tensor_map, destination[slot], barriers + slot, offset);
            }
        }
        __syncwarp();
        uint64_t checksum = 0;
        uint64_t completed = 0;
        for (uint32_t iteration = 0; iteration < tma_iterations; ++iteration) {
            uint32_t slot = iteration % kSlots;
            uint32_t parity = (iteration / kSlots) & 1;
            if (lane == 0) mbarrier_wait(barriers + slot, parity);
            __syncwarp();
            const uint32_t* words = reinterpret_cast<const uint32_t*>(
                shared_storage + slot * kTileBytes);
            checksum += words[lane];
            __syncwarp();
            ++completed;
            if (lane == 0 && (completed & 15) == 0) {
                __threadfence();
                progress_store(progress + blockIdx.x, completed);
            }
            if (lane == 0 && iteration + kSlots < tma_iterations) {
                uint32_t next_iteration = iteration + kSlots;
                uint32_t tile = (tma_index * 977u + next_iteration) % source_chunk_tiles;
                uint32_t offset = (tma_index * source_chunk_tiles + tile) * kTileBytes;
                tma_issue(&tensor_map, destination[slot], barriers + slot, offset);
            }
            __syncwarp();
        }
        for (uint32_t offset = 16; offset != 0; offset >>= 1) {
            checksum += __shfl_down_sync(0xffffffff, checksum, offset);
        }
        if (lane == 0) {
            __threadfence();
            progress_store(progress + blockIdx.x, completed);
            DeviceBlockStats& stats = block_stats[blockIdx.x];
            stats.tma_last_ns = globaltimer();
            stats.tma_bytes = static_cast<uint64_t>(completed) * kTileBytes;
            stats.tma_iterations = completed;
            stats.checksum = checksum;
        }
    }

    if (measure_warp && lane == 0) {
        DeviceMeasureStats stats{};
        stats.block_start_ns = start;
        stats.wait_start_ns = globaltimer();
        uint64_t now;
        do {
            now = globaltimer();
        } while (now - stats.wait_start_ns < kSleepNs);
        stats.wait_end_ns = now;
        index = chase_load(chain + index);
        stats.chase_start_ns = globaltimer_after(index);
        uint64_t progress_before = 0;
        for (uint32_t block = 0; block < total_blocks; ++block) {
            progress_before += progress_load(progress + block);
        }
        stats.progress_at_start = progress_before;
        uint64_t warmup_before = globaltimer();
        index = chase_load(chain + index);
        uint64_t warmup_after = globaltimer_after(index);
        if (warmup_after < warmup_before) latencies[0] = 0;
        for (uint32_t sample = 0; sample < samples; ++sample) {
            uint64_t before = globaltimer();
            for (uint32_t load = 0; load < kLoadsPerSample; ++load) {
                index = chase_load(chain + index);
            }
            uint64_t after = globaltimer_after(index);
            latencies[sample] = (after - before) / kLoadsPerSample;
        }
        uint64_t progress_after = 0;
        for (uint32_t block = 0; block < total_blocks; ++block) {
            progress_after += progress_load(progress + block);
        }
        stats.progress_at_end = progress_after;
        stats.chase_end_ns = globaltimer_after(index);
        stats.final_index = index;
        *measure_stats = stats;
    }
    __syncthreads();
    if (threadIdx.x == 0) block_stats[blockIdx.x].end_ns = globaltimer();
}

struct Config {
    uint32_t chain_mb = 512;
    uint32_t source_mb = 1024;
    uint32_t samples = 4096;
    uint32_t iterations = 65536;
    std::string mode = "all";
    std::string output = "results";
};

struct RunResult {
    std::string name;
    uint32_t grid_blocks = 0;
    uint32_t tma_sms = 0;
    uint32_t samples = 0;
    double average_ns = 0;
    uint64_t minimum_ns = 0;
    uint64_t maximum_ns = 0;
    double total_bandwidth_gbs = 0;
    double concurrent_bandwidth_gbs = 0;
    uint64_t total_bytes = 0;
    uint64_t concurrent_bytes = 0;
    uint64_t elapsed_ns = 0;
    uint64_t chase_elapsed_ns = 0;
    uint64_t wait_actual_ns = 0;
    uint64_t overlap_ns = 0;
    uint64_t progress_delta = 0;
    std::vector<uint64_t> latencies;
    std::vector<DeviceBlockStats> blocks;
    DeviceMeasureStats measure{};
};

void write_samples(const std::string& path, const std::vector<uint64_t>& values) {
    std::ofstream output(path);
    output << "sample,latency_ns\n";
    for (size_t i = 0; i < values.size(); ++i) output << i << ',' << values[i] << '\n';
}

void write_blocks(const std::string& path, const RunResult& result) {
    std::ofstream output(path);
    output << "block,smid,tma_enabled,start_ns,end_ns,tma_first_ns,tma_last_ns,"
              "tma_bytes,tma_iterations,checksum\n";
    for (size_t i = 0; i < result.blocks.size(); ++i) {
        const auto& block = result.blocks[i];
        output << i << ',' << block.smid << ',' << block.tma_enabled << ','
               << block.start_ns << ',' << block.end_ns << ',' << block.tma_first_ns << ','
               << block.tma_last_ns << ',' << block.tma_bytes << ',' << block.tma_iterations
               << ',' << block.checksum << '\n';
    }
}

void write_histogram(const std::string& path, const std::vector<uint64_t>& counts,
                     uint64_t lower, double width) {
    std::ofstream output(path);
    output << "bin,lower_ns,upper_ns,count\n";
    for (size_t i = 0; i < counts.size(); ++i) {
        output << i << ',' << std::fixed << std::setprecision(6)
               << lower + i * width << ',' << lower + (i + 1) * width << ',' << counts[i] << '\n';
    }
}

void build_histogram(const std::string& path, const RunResult& formal,
                     const RunResult& pretest) {
    std::vector<uint64_t> sorted = pretest.latencies;
    std::sort(sorted.begin(), sorted.end());
    uint64_t lower = sorted.front();
    size_t p99_index = static_cast<size_t>(std::ceil(0.99 * (sorted.size() - 1)));
    uint64_t p99 = sorted[p99_index];
    uint64_t upper = p99 + 1;
    if (lower == 0 || upper <= lower) throw std::runtime_error("Invalid histogram range");
    double width = static_cast<double>(upper - lower) / kHistogramBins;
    std::vector<uint64_t> histogram(kHistogramBins, 0);
    uint64_t below = 0;
    uint64_t above = 0;
    for (uint64_t latency : formal.latencies) {
        if (latency < lower) ++below;
        else if (latency >= upper) ++above;
        else ++histogram[static_cast<uint32_t>((latency - lower) / width)];
    }
    write_histogram(path, histogram, lower, width);
    std::string metadata_path = path.substr(0, path.size() - 4) + "_metadata.csv";
    std::ofstream metadata(metadata_path);
    metadata << "bins,pretest_min_ns,pretest_p99_ns,pretest_max_ns,bin_width_ns,"
                "formal_in_range,formal_below_min,formal_above_p99\n";
    metadata << kHistogramBins << ',' << lower << ',' << p99 << ',' << pretest.maximum_ns
             << ',' << std::setprecision(17) << width << ','
             << formal.latencies.size() - below - above << ',' << below << ',' << above << '\n';
}

class Benchmark {
  public:
    explicit Benchmark(Config config) : config_(std::move(config)) {
        CUDA_CHECK(cudaGetDevice(&device_));
        CUDA_CHECK(cudaGetDeviceProperties(&properties_, device_));
        bool rtx_5080 = properties_.major == 12 && properties_.minor == 0 &&
                        std::string(properties_.name).find("RTX 5080") != std::string::npos;
        bool b300 = properties_.major == 10 &&
                    std::string(properties_.name).find("B300") != std::string::npos;
        if (!rtx_5080 && !b300) {
            throw std::runtime_error(std::string("Expected RTX 5080/sm_120 or B300/sm_10x, got: ") +
                                     properties_.name);
        }
        chain_count_ = static_cast<uint64_t>(config_.chain_mb) * 1024 * 1024 / sizeof(uint32_t);
        if (chain_count_ > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Pointer chain exceeds 32-bit indexing");
        }
        source_size_ = static_cast<uint64_t>(config_.source_mb) * 1024 * 1024;
        if (source_size_ % kTileBytes != 0) throw std::runtime_error("Invalid TMA source size");
        std::cout << "GPU=" << properties_.name << " sm_count=" << properties_.multiProcessorCount
                  << " l2_bytes=" << properties_.l2CacheSize << '\n';
        std::cout << "Allocating chain=" << config_.chain_mb << " MiB, TMA source="
                  << config_.source_mb << " MiB\n";
        CUDA_CHECK(cudaMalloc(&chain_, chain_count_ * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&source_, source_size_));
        CUDA_CHECK(cudaMalloc(&latencies_, config_.samples * sizeof(uint64_t)));
        CUDA_CHECK(cudaMallocManaged(&progress_, properties_.multiProcessorCount * sizeof(uint64_t)));
        CUDA_CHECK(cudaMallocManaged(&arrival_, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&blocks_, properties_.multiProcessorCount * sizeof(DeviceBlockStats)));
        CUDA_CHECK(cudaMallocManaged(&measure_, sizeof(DeviceMeasureStats)));
        CUDA_CHECK(cudaMemset(source_, 0x5a, source_size_));
        initialize_chain();
        create_tensor_map();
        CUDA_CHECK(cudaFuncSetAttribute(benchmark_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kSlots * kTileBytes + kSlots * sizeof(uint64_t)));
        timer_resolution_ns_ = measure_timer_resolution();
        std::cout << "globaltimer_minimum_nonzero_delta_ns=" << timer_resolution_ns_ << '\n';
    }

    ~Benchmark() {
        cudaFree(chain_);
        cudaFree(source_);
        cudaFree(latencies_);
        cudaFree(progress_);
        cudaFree(arrival_);
        cudaFree(blocks_);
        cudaFree(measure_);
    }

    uint64_t timer_resolution_ns() const { return timer_resolution_ns_; }

    RunResult run(const std::string& name, uint32_t tma_sms, bool concurrent,
                  bool measure = true) {
        if (tma_sms > static_cast<uint32_t>(properties_.multiProcessorCount)) {
            throw std::runtime_error("TMA SM count exceeds device SM count");
        }
        uint32_t grid = std::max(1u, tma_sms);
        if (!concurrent && tma_sms != 0) grid = tma_sms + 1;
        if (grid > static_cast<uint32_t>(properties_.multiProcessorCount)) {
            throw std::runtime_error("Grid would exceed SM count");
        }
        uint32_t threads = concurrent && tma_sms != 0 ? 64 : 32;
        uint32_t source_tiles = source_size_ / kTileBytes;
        uint32_t active_tma = std::max(1u, tma_sms);
        uint32_t chunk_tiles = source_tiles / active_tma;
        if (chunk_tiles == 0) throw std::runtime_error("TMA source chunk is empty");

        CUDA_CHECK(cudaMemset(blocks_, 0, grid * sizeof(DeviceBlockStats)));
        CUDA_CHECK(cudaMemset(measure_, 0, sizeof(DeviceMeasureStats)));
        CUDA_CHECK(cudaMemset(progress_, 0, grid * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(arrival_, 0, sizeof(uint32_t)));
        *arrival_ = 0;
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t shared_bytes = kSlots * kTileBytes + kSlots * sizeof(uint64_t);
        benchmark_kernel<<<grid, threads, shared_bytes>>>(
            tensor_map_, chain_, static_cast<uint32_t>(chain_count_ - 1),
            measure ? config_.samples : 0, latencies_, blocks_, measure_, progress_, arrival_,
            grid, tma_sms, config_.iterations, chunk_tiles, concurrent ? 1 : 0,
            measure ? 1 : 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        RunResult result;
        result.name = name;
        result.grid_blocks = grid;
        result.tma_sms = tma_sms;
        result.samples = measure ? config_.samples : 0;
        result.blocks.resize(grid);
        CUDA_CHECK(cudaMemcpy(result.blocks.data(), blocks_, grid * sizeof(DeviceBlockStats),
                              cudaMemcpyDeviceToHost));
        result.measure = *measure_;
        if (measure) {
            result.latencies.resize(config_.samples);
            CUDA_CHECK(cudaMemcpy(result.latencies.data(), latencies_,
                                  config_.samples * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            auto extrema = std::minmax_element(result.latencies.begin(), result.latencies.end());
            result.minimum_ns = *extrema.first;
            result.maximum_ns = *extrema.second;
            uint64_t sum = std::accumulate(result.latencies.begin(), result.latencies.end(), uint64_t{0});
            result.average_ns = static_cast<double>(sum) / result.latencies.size();
            result.wait_actual_ns = result.measure.wait_end_ns - result.measure.wait_start_ns;
            result.chase_elapsed_ns = result.measure.chase_end_ns - result.measure.chase_start_ns;
            result.progress_delta = result.measure.progress_at_end - result.measure.progress_at_start;
            result.concurrent_bytes = result.progress_delta * kTileBytes;
            result.concurrent_bandwidth_gbs = static_cast<double>(result.concurrent_bytes) /
                                               result.chase_elapsed_ns;
        }
        uint64_t start = std::numeric_limits<uint64_t>::max();
        uint64_t end = 0;
        uint64_t first_tma = std::numeric_limits<uint64_t>::max();
        uint64_t last_tma = 0;
        std::set<uint32_t> sms;
        for (const auto& block : result.blocks) {
            sms.insert(block.smid);
            start = std::min(start, block.start_ns);
            end = std::max(end, block.end_ns);
            result.total_bytes += block.tma_bytes;
            if (block.tma_enabled) {
                first_tma = std::min(first_tma, block.tma_first_ns);
                last_tma = std::max(last_tma, block.tma_last_ns);
            }
            if (block.tma_enabled && block.tma_iterations != config_.iterations) {
                throw std::runtime_error("TMA loop terminated early");
            }
            if (block.tma_enabled && block.checksum == 0) {
                throw std::runtime_error("Invalid TMA destination consumption");
            }
        }
        if (sms.size() != grid) throw std::runtime_error("Blocks did not occupy distinct SMs");
        result.elapsed_ns = end - start;
        if (result.total_bytes) {
            result.total_bandwidth_gbs = static_cast<double>(result.total_bytes) / result.elapsed_ns;
        }
        if (measure && tma_sms) {
            uint64_t overlap_begin = std::max(first_tma, result.measure.chase_start_ns);
            uint64_t overlap_end = std::min(last_tma, result.measure.chase_end_ns);
            if (overlap_end > overlap_begin) result.overlap_ns = overlap_end - overlap_begin;
            if (result.overlap_ns == 0 || result.progress_delta == 0) {
                throw std::runtime_error("No measured TMA/pointer-chasing concurrency");
            }
        }
        std::cout << name << ": grid=" << grid << " tma_sms=" << tma_sms;
        if (measure) {
            std::cout << " avg_ns=" << result.average_ns << " min_ns=" << result.minimum_ns
                      << " max_ns=" << result.maximum_ns << " wait_ns=" << result.wait_actual_ns;
        }
        if (tma_sms) {
            std::cout << " bandwidth_GB/s=" << result.concurrent_bandwidth_gbs
                      << " total_GB/s=" << result.total_bandwidth_gbs
                      << " overlap_ns=" << result.overlap_ns;
        }
        std::cout << '\n';
        write_samples(config_.output + "/" + name + "_samples.csv", result.latencies);
        write_blocks(config_.output + "/" + name + "_blocks.csv", result);
        return result;
    }

  private:
    void initialize_chain() {
        std::vector<uint32_t> order(chain_count_);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937_64 random(0x5080120);
        std::shuffle(order.begin(), order.end(), random);
        std::vector<uint32_t> chain(chain_count_);
        for (uint64_t i = 0; i < chain_count_; ++i) {
            chain[order[i]] = order[(i + 1) % chain_count_];
        }
        CUDA_CHECK(cudaMemcpy(chain_, chain.data(), chain_count_ * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        std::cout << "Initialized one random permutation cycle, entries=" << chain_count_ << '\n';
    }

    void create_tensor_map() {
        uint64_t global_dim[2] = {256, source_size_ / 256};
        uint64_t global_strides[1] = {256};
        uint32_t box_dim[2] = {256, kTileBytes / 256};
        uint32_t element_strides[2] = {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(
            &tensor_map_, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, source_, global_dim,
            global_strides, box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    uint64_t measure_timer_resolution() {
        uint64_t* device_result;
        CUDA_CHECK(cudaMallocManaged(&device_result, sizeof(uint64_t)));
        timer_resolution_kernel<<<1, 1>>>(device_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t result = *device_result;
        cudaFree(device_result);
        return result;
    }

    Config config_;
    int device_ = 0;
    cudaDeviceProp properties_{};
    uint64_t chain_count_ = 0;
    uint64_t source_size_ = 0;
    uint32_t* chain_ = nullptr;
    void* source_ = nullptr;
    uint64_t* latencies_ = nullptr;
    uint64_t* progress_ = nullptr;
    uint32_t* arrival_ = nullptr;
    DeviceBlockStats* blocks_ = nullptr;
    DeviceMeasureStats* measure_ = nullptr;
    CUtensorMap tensor_map_{};
    uint64_t timer_resolution_ns_ = 0;
};

Config parse_arguments(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        std::string argument = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("Missing argument after " + argument);
            return argv[i];
        };
        if (argument == "--mode") config.mode = next();
        else if (argument == "--chain-mb") config.chain_mb = std::stoul(next());
        else if (argument == "--source-mb") config.source_mb = std::stoul(next());
        else if (argument == "--samples") config.samples = std::stoul(next());
        else if (argument == "--iterations") config.iterations = std::stoul(next());
        else if (argument == "--output") config.output = next();
        else throw std::runtime_error("Unknown argument: " + argument);
    }
    if (config.samples == 0 || config.iterations < kSlots || config.chain_mb < 128 ||
        config.source_mb < 128) {
        throw std::runtime_error("Invalid configuration");
    }
    return config;
}

void write_summary(const std::string& path, const std::vector<RunResult>& results,
                   uint64_t timer_resolution) {
    std::ofstream output(path);
    output << "experiment,grid_blocks,tma_sms,samples,average_ns,min_ns,max_ns,"
              "concurrent_bandwidth_GB_per_s,total_bandwidth_GB_per_s,total_bytes,"
              "concurrent_bytes,elapsed_ns,chase_elapsed_ns,wait_actual_ns,overlap_ns,"
              "progress_delta_iterations,globaltimer_resolution_ns\n";
    for (const auto& result : results) {
        output << result.name << ',' << result.grid_blocks << ',' << result.tma_sms << ','
               << result.samples << ',' << std::setprecision(17) << result.average_ns << ','
               << result.minimum_ns << ',' << result.maximum_ns << ','
               << result.concurrent_bandwidth_gbs << ',' << result.total_bandwidth_gbs << ','
               << result.total_bytes << ',' << result.concurrent_bytes << ',' << result.elapsed_ns
               << ',' << result.chase_elapsed_ns << ',' << result.wait_actual_ns << ','
               << result.overlap_ns << ',' << result.progress_delta << ',' << timer_resolution
               << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config config = parse_arguments(argc, argv);
        Benchmark benchmark(config);
        std::vector<RunResult> results;
        if (config.mode == "idle" || config.mode == "all") {
            results.push_back(benchmark.run("idle", 0, false));
        }
        if (config.mode == "separate" || config.mode == "all") {
            uint32_t count = 84;
            cudaDeviceProp prop{};
            CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
            count = prop.multiProcessorCount - 1;
            RunResult pretest = benchmark.run("separate_pretest", count, false);
            results.push_back(pretest);
            RunResult formal = benchmark.run("separate", count, false);
            build_histogram(config.output + "/separate_histogram.csv", formal, pretest);
            results.push_back(formal);
        }
        if (config.mode == "concurrent" || config.mode == "all") {
            cudaDeviceProp prop{};
            CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
            uint32_t count = prop.multiProcessorCount;
            RunResult pretest = benchmark.run("concurrent_pretest", count, true);
            results.push_back(pretest);
            RunResult formal = benchmark.run("concurrent", count, true);
            build_histogram(config.output + "/concurrent_histogram.csv", formal, pretest);
            results.push_back(formal);
        }
        if (config.mode == "sweep" || config.mode == "all") {
            cudaDeviceProp prop{};
            CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
            uint32_t count = prop.multiProcessorCount;
            std::vector<uint32_t> points{1, 2, 4, 5, 6, 7, 8, 12, 16, 24, 32, 36, 40, 48, 56, 64, 72, 80};
            for (uint32_t tma_sms : points) {
                if (tma_sms < count) {
                    results.push_back(benchmark.run("sweep_" + std::to_string(tma_sms),
                                                    tma_sms, true));
                }
            }
            results.push_back(benchmark.run("sweep_" + std::to_string(count), count, true));
        }
        if (config.mode == "all") {
            double target_bandwidth = 0;
            for (const auto& result : results) {
                if (result.name == "concurrent") {
                    target_bandwidth = 0.5 * result.concurrent_bandwidth_gbs;
                }
            }
            uint32_t half_sms = 0;
            double nearest = std::numeric_limits<double>::max();
            for (const auto& result : results) {
                if (result.name.rfind("sweep_", 0) == 0) {
                    double distance = std::abs(result.concurrent_bandwidth_gbs - target_bandwidth);
                    if (distance < nearest) {
                        nearest = distance;
                        half_sms = result.tma_sms;
                    }
                }
            }
            if (half_sms == 0) throw std::runtime_error("No half-bandwidth sweep candidate");
            std::cout << "Half-bandwidth target_GB/s=" << target_bandwidth
                      << " selected_tma_sms=" << half_sms << '\n';
            RunResult pretest = benchmark.run("half_bandwidth_pretest", half_sms, true);
            results.push_back(pretest);
            RunResult formal = benchmark.run("half_bandwidth", half_sms, true);
            build_histogram(config.output + "/half_bandwidth_histogram.csv", formal, pretest);
            results.push_back(formal);
        }
        write_summary(config.output + "/summary.csv", results,
                      benchmark.timer_resolution_ns());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
