// =============================================================================
// tma_load_bench.cu
//
// Benchmark the *peak load bandwidth* achievable with the TMA bulk-copy engine
// (cp.async.bulk, a.k.a. UBLKCP / UTMALDG) on Blackwell (sm_100, B200).
//
// Design follows "DevTech Blackwell Tensor Core Training - Memory subsystem":
//   * Little's law:  bytes_in_flight = bandwidth * latency.
//     To saturate DRAM we must keep MANY bytes in flight per SM.
//   * TMA decouples bytes-in-flight from the thread count: a single warp can
//     keep dozens of large async copies outstanding (the slides use ~320
//     threads/SM instead of 2048).
//   * Multi-stage software pipeline. The slides recommend a total stage budget
//     of ~64 KB, "prefer > 2 stages", and report that 4-8 stages of 8-16 KB
//     each reach ~7 TB/s, vs ~6 TB/s for 2 stages of 32 KB.
//
// This kernel is a *pure load* micro-benchmark: each block streams through a
// large global buffer, issuing one TMA bulk-copy per pipeline stage into
// shared memory, and only lightly "touches" the data so nothing is optimised
// away. No store / no compute -> we measure the load path in isolation.
//
// The host driver sweeps (#stages x bytes/load), follows the slides' caveats
// (warmup, L2 flush, all-zero data, >1ms kernels) and also measures the
// *achieved* SM clock (B200 throttles under power limits).
//
// Build:
//   nvcc -arch=sm_100a -O3 -std=c++17 tma_load_bench.cu -o tma_load_bench
// Run:
//   ./tma_load_bench                 # full sweep, default 4 GiB working set
//   ./tma_load_bench 8               # 8 GiB working set
// =============================================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Error checking
// -----------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                         cudaGetErrorName(_e), __FILE__, __LINE__,             \
                         cudaGetErrorString(_e));                              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// -----------------------------------------------------------------------------
// Minimal, self-contained TMA / mbarrier PTX wrappers (sm_90+).
// Equivalent to the helpers in DeepGEMM's deep_gemm/ptx/tma.cuh, but with no
// CUTLASS dependency so this file builds standalone.
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t arrive_count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(smem_u32(bar)), "r"(arrive_count) : "memory");
}

// Fence so the initialised barriers are visible before any arrive/wait.
__device__ __forceinline__ void mbar_init_fence() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

// One "arrive" + declare how many transaction bytes the barrier should expect.
// The completion of the matching cp.async.bulk delivers those bytes.
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}

// Non-blocking parity test; returns true once the stage's copy has completed.
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t phase) {
    uint32_t ok;
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "  selp.u32 %0, 1, 0, p;\n"
                 "}\n"
                 : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase) : "memory");
    return ok != 0;
}

// 1-D TMA bulk copy: GMEM -> SMEM, completion signalled on the mbarrier.
__device__ __forceinline__ void tma_load_1d(void* dst_smem, const void* src_gmem,
                                            uint64_t* bar, uint32_t bytes) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];\n"
        :: "r"(smem_u32(dst_smem)), "l"(src_gmem), "r"(bytes), "r"(smem_u32(bar))
        : "memory");
}

// -----------------------------------------------------------------------------
// Achieved-clock measurement (slide 9). On pre-release B200 the preset clocks
// are often not hit, so we read %globaltimer (ns) and clock64() (cycles) on
// SM 0 around the benchmarked kernel.
// -----------------------------------------------------------------------------
__global__ void timestamp_kernel(uint64_t* g_ns, uint64_t* sm0_clk) {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (smid == 0 && threadIdx.x == 0) {
        uint64_t ns;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
        *g_ns    = ns;
        *sm0_clk = clock64();
    }
}

// -----------------------------------------------------------------------------
// The TMA load benchmark kernel.
//
//   STAGES  : number of pipeline stages (independent outstanding copies)
//   BYTES   : bytes per TMA load (must be a multiple of 16, 128B-aligned src)
//
// Each block owns STAGES shared-memory buffers and STAGES transaction barriers.
// A single warp (32 threads) drives the pipeline; the TMA copy is issued by one
// elected lane, while all lanes "consume" a little of the data so the loads
// stay live. Blocks grid-stride over all chunks of the global buffer, so the
// grid stays persistent and inter-CTA launch latency is irrelevant.
//
// Bytes in flight per block  = STAGES * BYTES.
// Bytes in flight per SM      = STAGES * BYTES * (blocks resident per SM).
// -----------------------------------------------------------------------------
template <int STAGES, int BYTES>
__global__ __launch_bounds__(32, 1) void tma_load_kernel(
        const uint8_t* __restrict__ src, uint64_t num_chunks, uint64_t* __restrict__ sink) {
    extern __shared__ __align__(128) uint8_t smem[];   // STAGES * BYTES
    __shared__ uint64_t bar[STAGES];                   // one barrier per stage

    const uint32_t lane   = threadIdx.x;               // blockDim.x == 32
    const uint64_t stride = gridDim.x;                 // chunks between iterations

    // --- init barriers (each expects 1 arrive + BYTES transaction bytes) ---
    if (lane < STAGES)
        mbar_init(&bar[lane], 1);
    mbar_init_fence();
    __syncwarp();

    auto stage_ptr = [&](int s) -> uint8_t* { return smem + s * BYTES; };

    // Issue one TMA load for chunk `c` into stage `s` (elected lane only).
    auto issue = [&](int s, uint64_t c) {
        if (lane == 0) {
            const uint8_t* gptr = src + c * uint64_t(BYTES);
            tma_load_1d(stage_ptr(s), gptr, &bar[s], BYTES);
            mbar_arrive_expect_tx(&bar[s], BYTES);
        }
    };

    uint32_t phase[STAGES];
    #pragma unroll
    for (int s = 0; s < STAGES; ++s) phase[s] = 0;

    // First chunk this block handles, and a running "next chunk to fetch".
    const uint64_t base = blockIdx.x;
    uint64_t next = base;

    // --- prologue: fill all stages ---
    #pragma unroll
    for (int s = 0; s < STAGES; ++s) {
        if (next < num_chunks) { issue(s, next); next += stride; }
    }
    __syncwarp();

    // --- steady state: wait a stage, consume it, refill it ---
    uint4 acc = make_uint4(0, 0, 0, 0);
    uint64_t consumed = base;
    int s = 0;
    while (consumed < num_chunks) {
        // Wait for this stage's copy to land.
        while (!mbar_try_wait(&bar[s], phase[s])) { /* spin */ }

        // Lightly touch the data so the load cannot be elided. Each lane reads
        // a couple of 16-byte words spread across the buffer (cheap on SMEM
        // bandwidth, but forces the full BYTES to have been delivered).
        const uint4* buf = reinterpret_cast<const uint4*>(stage_ptr(s));
        constexpr int kElems = BYTES / 16;
        uint4 v0 = buf[lane % kElems];
        uint4 v1 = buf[(kElems / 2 + lane) % kElems];
        acc.x ^= v0.x ^ v1.x;
        acc.y ^= v0.y ^ v1.y;
        acc.z ^= v0.z ^ v1.z;
        acc.w ^= v0.w ^ v1.w;

        // Refill this stage with the next outstanding chunk.
        if (next < num_chunks) { issue(s, next); next += stride; }
        phase[s] ^= 1;

        consumed += stride;
        s = (s + 1) % STAGES;
    }

    // Keep `acc` (and therefore every load/wait) live.
    if (acc.x ^ acc.y ^ acc.z ^ acc.w) sink[blockIdx.x] = acc.x;
}

// -----------------------------------------------------------------------------
// Host side
// -----------------------------------------------------------------------------
static int   g_num_sms        = 0;
static int   g_max_smem_optin = 0;   // max dynamic smem per block (opt-in)
static int   g_max_smem_perSM = 0;   // max smem per multiprocessor

// L2 flush: overwrite a buffer larger than L2 so cached lines are evicted.
static uint8_t* g_l2_flush = nullptr;
static size_t   g_l2_flush_bytes = 0;
static void flush_l2() {
    CUDA_CHECK(cudaMemset(g_l2_flush, 0, g_l2_flush_bytes));
}

// Measure achieved SM clock (GHz) by bracketing a callable with timestamp kernels.
template <typename F>
static double measure_clock_ghz(F&& f) {
    static uint64_t *d_ns = nullptr, *d_clk = nullptr;
    if (!d_ns) { CUDA_CHECK(cudaMalloc(&d_ns, 8)); CUDA_CHECK(cudaMalloc(&d_clk, 8)); }
    uint64_t ns0, clk0, ns1, clk1;
    timestamp_kernel<<<g_num_sms, 32>>>(d_ns, d_clk);
    CUDA_CHECK(cudaMemcpy(&ns0,  d_ns,  8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&clk0, d_clk, 8, cudaMemcpyDeviceToHost));
    f();
    timestamp_kernel<<<g_num_sms, 32>>>(d_ns, d_clk);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&ns1,  d_ns,  8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&clk1, d_clk, 8, cudaMemcpyDeviceToHost));
    double dns  = double(ns1  - ns0);
    double dclk = double(clk1 - clk0);
    return dns > 0 ? dclk / dns : 0.0;   // cycles per ns == GHz
}

struct Result {
    int    stages, bytes;
    int    blocks_per_sm;
    double gbps;       // best of timed runs
    double ghz;        // achieved SM clock
};

// Run one (STAGES, BYTES) configuration. Returns false if it does not fit.
template <int STAGES, int BYTES>
static bool run_config(const uint8_t* d_src, uint64_t total_bytes,
                       uint64_t* d_sink, Result& out) {
    const int smem_bytes = STAGES * BYTES;   // dynamic shared memory per block
    if (smem_bytes > g_max_smem_optin) return false;

    auto kernel = tma_load_kernel<STAGES, BYTES>;
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    // How many of these blocks fit on one SM (limited here by shared memory)?
    // More resident blocks -> more bytes in flight per SM.
    int blocks_per_sm = g_max_smem_perSM / (smem_bytes + 1024 /*barriers+slack*/);
    blocks_per_sm = std::max(1, std::min(blocks_per_sm, 8));

    const uint64_t num_chunks = total_bytes / BYTES;
    const int grid = g_num_sms * blocks_per_sm;

    auto launch = [&]() {
        kernel<<<grid, 32, smem_bytes>>>(d_src, num_chunks, d_sink);
    };

    // Warmup.
    for (int i = 0; i < 3; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Timed runs (best-of-N), flushing L2 before each.
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    const int iters = 7;
    double best_ms = 1e30;
    for (int i = 0; i < iters; ++i) {
        flush_l2();
        CUDA_CHECK(cudaEventRecord(beg));
        launch();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
        best_ms = std::min(best_ms, double(ms));
    }
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));

    // Achieved clock during a representative run.
    double ghz = measure_clock_ghz([&]() { launch(); });

    const double moved   = double(num_chunks) * BYTES;        // bytes read
    const double gbps    = moved / (best_ms * 1e-3) / 1e9;    // GB/s (10^9)
    out = Result{STAGES, BYTES, blocks_per_sm, gbps, ghz};
    return true;
}

int main(int argc, char** argv) {
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    g_num_sms        = prop.multiProcessorCount;
    g_max_smem_perSM = prop.sharedMemPerMultiprocessor;
    g_max_smem_optin = prop.sharedMemPerBlockOptin;

    // Working-set size in GiB (default 4). Larger -> deeper into DRAM regime.
    double gib = (argc > 1) ? std::atof(argv[1]) : 4.0;
    size_t total_bytes = size_t(gib * (1ull << 30));
    total_bytes &= ~size_t(64 * 1024 - 1);   // round to 64KB for clean tiling

    std::printf("Device         : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("SMs            : %d\n", g_num_sms);
    std::printf("Smem/SM        : %d KiB   Smem/block(opt-in): %d KiB\n",
                g_max_smem_perSM >> 10, g_max_smem_optin >> 10);
    std::printf("Working set    : %.2f GiB (%zu bytes, all-zero data)\n",
                total_bytes / double(1ull << 30), total_bytes);
    int mem_clk_khz = 0, bus_width = 0;
    cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_width,  cudaDevAttrGlobalMemoryBusWidth, dev);
    std::printf("Mem clock      : %.0f MHz   bus width: %d bit\n",
                mem_clk_khz / 1e3, bus_width);
    // SOL DRAM bandwidth = bus_width * mem_clock * 2 (DDR) / 8  (slide 9).
    double sol_tbps = double(bus_width) * (double(mem_clk_khz) * 1e3) * 2.0
                      / 8.0 / 1e12;
    std::printf("Reported SOL BW: %.2f TB/s (note: r5xx drivers misreport bus width)\n\n",
                sol_tbps);

    // Allocate working set (all-zero, per slide 9 to avoid clock throttle on
    // pre-production parts) and L2-flush scratch (> L2 capacity).
    uint8_t* d_src = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, total_bytes));
    CUDA_CHECK(cudaMemset(d_src, 0, total_bytes));

    g_l2_flush_bytes = size_t(prop.l2CacheSize) * 2 + (64u << 20);
    CUDA_CHECK(cudaMalloc(&g_l2_flush, g_l2_flush_bytes));

    uint64_t* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t) * g_num_sms * 8));

    std::printf("%-7s %-10s %-9s %-12s %-10s %-9s\n",
                "stages", "bytes/ld", "in-flight", "BW(TB/s)", "%SOL", "clk(GHz)");
    std::printf("------------------------------------------------------------------\n");

    std::vector<Result> results;
    auto record = [&](bool ok, const Result& r) {
        if (!ok) return;
        results.push_back(r);
        double inflight_kb = double(r.stages) * r.bytes / 1024.0;
        std::printf("%-7d %-10s %-9s %-12.3f %-10.1f %-9.3f\n",
                    r.stages,
                    (r.bytes >= 1024 ? std::to_string(r.bytes / 1024) + "KB"
                                     : std::to_string(r.bytes) + "B").c_str(),
                    (std::to_string((int)(inflight_kb * r.blocks_per_sm)) + "KB/SM").c_str(),
                    r.gbps / 1000.0,
                    sol_tbps > 0 ? r.gbps / 1000.0 / sol_tbps * 100.0 : 0.0,
                    r.ghz);
    };

    Result r;
    // Sweep: #stages x bytes/load. The slides' sweet spot is 4-8 stages of
    // 8-16 KB (~64 KB total). We also probe 2 stages and 32 KB loads, plus a
    // few larger budgets to find the true ceiling on B200.
    #define RUN(S, B) record(run_config<S, B>(d_src, total_bytes, d_sink, r), r)
    RUN(2,  4096);  RUN(4,  4096);  RUN(8,  4096);  RUN(16, 4096);
    RUN(2,  8192);  RUN(4,  8192);  RUN(8,  8192);  RUN(16, 8192);
    RUN(2, 16384);  RUN(4, 16384);  RUN(8, 16384);  RUN(12, 16384);
    RUN(2, 32768);  RUN(4, 32768);  RUN(6, 32768);
    RUN(2, 65536);  RUN(3, 65536);
    #undef RUN

    // Summary.
    std::printf("------------------------------------------------------------------\n");
    if (!results.empty()) {
        auto best = *std::max_element(results.begin(), results.end(),
            [](const Result& a, const Result& b){ return a.gbps < b.gbps; });
        std::printf("BEST: %d stages x %d KB  ->  %.3f TB/s  (%.1f%% of SOL, %.3f GHz)\n",
                    best.stages, best.bytes / 1024, best.gbps / 1000.0,
                    sol_tbps > 0 ? best.gbps / 1000.0 / sol_tbps * 100.0 : 0.0,
                    best.ghz);
    }

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(g_l2_flush));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}
