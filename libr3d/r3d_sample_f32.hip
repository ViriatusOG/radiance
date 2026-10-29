/* r3d_sample_f32.hip -- the sampler: the two full-vocabulary reductions, the candidate set, the
 * seeded draw and the four narrowing stages.
 *
 * The arithmetic is libref's, transcribed, including the order of its passes. `abi/rad_sample.h`
 * is the wire format: every scalar (`temp`, `top_k`, `top_p`, `min_p`, `typical_p`, `xtc_*`, `seed`,
 * `pos`, the flags, the history window) is data, one 96-byte row per sampled position, because a
 * parameter is geometry frozen at declare and two rows of one batch with different top_p must
 * resolve to one kernel. The params plane must be 32-bit words (see params_ok).
 *
 * ============================== reproducibility ==============================
 *
 * Draws must reproduce, so summation order is transcribed, not chosen for speed:
 *
 *   - Every max is a strict `>`, order-free, and keeps NaN out: libref's `if (v > mx) mx = v` from
 *     -infinity never adopts a NaN, and the butterfly merges the same way.
 *   - An f32 sum is not order-free. libref's sums (the pick, the narrowing stages) and the CDF scan
 *     run over the candidates in index order, so exps are computed in parallel and accumulated
 *     serially on one lane in index order: parallel values, serial sums.
 *   - The draw is specified by the ABI: splitmix64 over `seed * 0x9E3779B97F4A7C15 + pos`, top 24
 *     bits over 2^24, integer arithmetic, so device and host agree. Independent draws at one
 *     position XOR the seed with a key: RAD_SAMPLE_KEY_PICK (0) for the pick, RAD_SAMPLE_KEY_XTC for
 *     XTC's coin.
 *   - `ref_exp`/`ref_log` are `(float)std::exp((double)x)` and `(float)std::log((double)x)`, as in
 *     the reference. On gfx1102 the device's double exp/log match glibc's bit for bit over 2^20
 *     points in [-100, 100], while `expf` differs from the host's by 1 ulp on 5.7% of them, and a
 *     `u < cum` comparison on an ulp boundary changes the drawn token. Consumer RDNA3 runs fp64 at
 *     1:64, which the candidate set (1024 exponentials a row) can afford. The attention unit, at
 *     millions of exponentials a layer, uses a 0.5-ulp polynomial instead.
 *
 * ============================== one-pass top-k ==============================
 *
 * libref selects with `topk_next`: R scans of the row, each taking the largest value strictly below
 * the last pick, ties to the lower column. That is a sort by (-value, +column), so one pass keeping
 * the best R in that order yields the same R pairs in the same order. At N=130560, R=16, M=1792
 * this reads the row once instead of sixteen times. The pass is bound by accepted inserts into
 * each lane's list, not by memory; filtering against a block-wide threshold instead of each lane's
 * own tail would cut the inserts about tenfold.
 *
 * The per-lane list is at most 64 deep, and the merge keeps the best 64 of two lists. Almost every
 * element costs one comparison against the lane's last entry (about 12% clear it at 510 elements a
 * lane). 64 covers R=16, a 64-wide candidate set and practical top_k values; a larger k takes
 * ceil(k/64) passes, carrying the chain's state, so the answer is the same at any k.
 *
 * The chain's quirk is transcribed too, because kbench compares the plane: `topk_next` seeds its
 * scan with `best_i < 0 ||`, so the first element becomes the best even when it is a NaN, and
 * nothing displaces it. A row whose first logit is NaN yields R identical (NaN, 0) slots. The
 * kernels test the first element for that case explicitly.
 *
 * ============================== shape ==============================
 *
 * One block a row everywhere. At the band a row is 130560 f32 (510 KiB) and M is 2048, so argmax
 * and top-k each read 1 GiB and temp reads and writes 2 GiB: these are memory-bound. A candidate
 * row is at most 1024 f32 and the work is transcendentals, so the same shape costs nothing and
 * gives the serial order for free; the stages keep the row in LDS, which makes typical's O(n^2)
 * distance scan affordable.
 *
 * Single rank. `vocab_off` is applied where libref applies it (`row_topk`'s reported ids,
 * `sample_topk`'s `pairs` plane), and the pair planes are written as the shape hooks describe them.
 * There is no merge kernel: libref's own shape hook declines `sample_merge_topk`.
 */

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <rad_sample.h>

/* ================================================================== shared device helpers */
namespace {

constexpr uint32_t kThreads = 256;          /* 8 waves */
constexpr uint32_t kWave    = 32;
constexpr uint32_t kWaves   = kThreads / kWave;

/* The candidate pool's depth, and the widest candidate set the narrowing stages carry in LDS (the
 * engine's declared buffer width). */
constexpr int kPool = 64;
constexpr int kCandMax = 1024;

/* The reference's transcendentals: a double call rounded once to f32 (see the file header), so the
 * draws are the reference's. */
__device__ __forceinline__ float ref_exp(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float ref_log(float x) { return (float)log((double)x); }

/* abi/rad_sample.h's draw, verbatim: splitmix64 over (seed * golden + pos), top 24 bits as a float
 * in [0, 1). 24 bits fit f32's mantissa exactly. */
__device__ __forceinline__ uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__device__ __forceinline__ float uniform01(uint64_t seed, uint64_t pos) {
    const uint64_t z = splitmix64(seed * 0x9E3779B97F4A7C15ull + pos);
    return (float)(uint32_t)(z >> 40) * (1.0f / 16777216.0f);
}

/* A block-wide maximum: a wave butterfly of fmaxf, then the wave partials. fmaxf drops a NaN, so a
 * NaN is never the result. `lds` holds kWaves floats. */
__device__ __forceinline__ float block_max(float v, float* lds) {
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    v = wave_max(v);                                   /* a butterfly of fmaxf: drops a NaN */
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float t = lds[0];
#pragma unroll
    for (uint32_t w = 1; w < kWaves; ++w) t = fmaxf(t, lds[w]);
    return t;
}

/* ------------------------------------------------------------------ sample_argmax
 *
 * The largest logit, ties to the lower column, so greedy and top-1 agree. The reference's scan is
 * `bi < 0 || v > best`: strict, so the first of a run wins, and a NaN is adopted only as the row's
 * first element. The parallel form uses the same total order, which is associative, so the
 * butterfly and the serial scan agree.
 */
struct Best { float v; uint32_t i; };

/* `best_take(a, b)` keeps `a` unless `b` is strictly better (larger, or equal with a lower column);
 * `b.i == 0xffffffff` means nothing examined yet. */
__device__ __forceinline__ Best best_take(Best a, Best b) {
    const bool better = (b.v > a.v) || (b.v == a.v && b.i < a.i);
    return (b.i != 0xffffffffu && better) ? b : a;
}

__global__ __launch_bounds__(kThreads) R3D_WAVES(8) void sample_argmax_kernel(
    const float* __restrict__ lg, int32_t* __restrict__ tok, uint32_t n, uint32_t lg_ld) {
    __shared__ float red[kWaves];
    __shared__ uint32_t redi[kWaves];

    const uint32_t row = blockIdx.x;
    const float* __restrict__ lr = lg + (size_t)row * lg_ld;

    /* Non-NaN values only, lowest column winning a tie. */
    Best b{ -INFINITY, 0xffffffffu };
    for (uint32_t c = threadIdx.x; c < n; c += kThreads) {
        const float v = ldg_f32(lr + c);
        if (v != v) continue;
        if (b.i == 0xffffffffu || v > b.v) { b.v = v; b.i = c; }
    }
#define R3D_BEST_XOR(M)                                                                             \
    {                                                                                              \
        const Best o{ r3d_xor_lane<M>(b.v), r3d_xor_lane_u32<M>(b.i) };                             \
        b = best_take(b, o);                                                                        \
    }
    R3D_BEST_XOR(16)
    R3D_BEST_XOR(8)
    R3D_BEST_XOR(4)
    R3D_BEST_XOR(2)
    R3D_BEST_XOR(1)
#undef R3D_BEST_XOR
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) { red[wave] = b.v; redi[wave] = b.i; }
    __syncthreads();
    if (threadIdx.x == 0) {
        Best t{ red[0], redi[0] };
#pragma unroll
        for (uint32_t w = 1; w < kWaves; ++w) t = best_take(t, Best{ red[w], redi[w] });
        /* topk_next's seed: a row whose first element is NaN reports column 0 (see the file
         * header); every other NaN is skipped. */
        const float x0 = ldg_f32(lr);
        tok[row] = (int32_t)(x0 != x0 ? 0u : t.i);
    }
}

/* ------------------------------------------------------------------ sample_temp
 *
 * `inv = 1/t` once per row, then every logit multiplied by it (not divided), in place. A row with
 * `!(t > 0) || t == 1` is left bit-exact, so temp 1.0 and temp 0.0 (greedy) are no-ops.
 */
__global__ __launch_bounds__(kThreads) R3D_WAVES(8) void sample_temp_kernel(
    float* __restrict__ lg, const RadSampleParams* __restrict__ pr, uint32_t n, uint32_t lg_ld) {
    const uint32_t row = blockIdx.x;
    const float t = pr[row].temp;
    if (!(t > 0.0f) || t == 1.0f) return;
    const float inv = 1.0f / t;
    float* __restrict__ lr = lg + (size_t)row * lg_ld;
    for (uint32_t c = threadIdx.x; c < n; c += kThreads) {
        const float v = ldg_f32(lr + c);
        lr[c] = v * inv;
    }
}

}  /* namespace */

/*   lg     [M][n_vocab] f32, INOUT for temp, read-only for argmax; `lg_ld` elements a row (0 = n)
 *   tok    [M]          i32
 *   params [M]          24 words a row; argmax does not read it (libref describes it anyway)
 *
 * 0, -1 null, -2 bad extent, -3 pitch that does not describe the operand, -4 the HIP launch failed.
 * No allocation, no synchronisation. */
extern "C" int r3d_sample_argmax(long lg, long tok, int M, int n_vocab, long long lg_ld,
                                long stream) {
    if (!lg || !tok) return -1;
    if (M <= 0 || n_vocab <= 0) return -2;
    if (!lg_ld) lg_ld = n_vocab;
    if (lg_ld < n_vocab) return -3;
    if (M > 0x7fffffff || n_vocab > 0x7fffffff) return -2;

    const dim3 grid((unsigned)M), block(kThreads);
    R3D_KLAUNCH(R3D_KNAME(sample_argmax_kernel), grid, block, (hipStream_t)(uintptr_t)stream,
                (const float*)lg, (int32_t*)tok, (uint32_t)n_vocab, (uint32_t)lg_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_sample_temp(long lg, long params, int M, int n_vocab, long long lg_ld,
                              long stream) {
    if (!lg || !params) return -1;
    if (M <= 0 || n_vocab <= 0) return -2;
    if (!lg_ld) lg_ld = n_vocab;
    if (lg_ld < n_vocab) return -3;
    if (M > 0x7fffffff || n_vocab > 0x7fffffff) return -2;

    const dim3 grid((unsigned)M), block(kThreads);
    R3D_KLAUNCH(R3D_KNAME(sample_temp_kernel), grid, block, (hipStream_t)(uintptr_t)stream,
                (float*)lg, (const RadSampleParams*)params, (uint32_t)n_vocab, (uint32_t)lg_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ================================================================== the top-k pool
 *
 * One lane's share of a row, kept as the POOL best by (value descending, column ascending), the
 * order `topk_next` walks.
 *
 *   - The order is strict and total, so merging two lists keeps the first POOL of the union.
 *   - The list lives in registers, so insertion is a POOL-step sift with static indices; a dynamic
 *     index would spill the array to private memory. Only elements that beat the lane's last entry
 *     (one compare) are sifted.
 *
 * Unfilled slots hold (-INFINITY, 0xffffffff): below any real value and above any real column, so a
 * real -infinity logit (a masked one) beats a pad and extraction stops at the first pad.
 */
namespace {

constexpr uint32_t kPadCol = 0xffffffffu;

/* The pool's order, strict: larger value wins, equal value goes to the lower column. */
__device__ __forceinline__ bool pool_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

/* Insert (v, i) into a lane's register list. POOL is the list depth and the width of a pass, so a
 * router's top_k of 16 uses a 16-deep pool rather than paying for 64.
 *
 * `pv`/`pi` carry the pick chain across passes: an element above the last pick, or equal to it at
 * or before its column, was already picked (`topk_next`'s two skips, in its order). */
template <int POOL>
__device__ __forceinline__ void pool_insert(float v, uint32_t i, float pv, uint32_t pi,
                                            float* lv, uint32_t* li) {
    if (v != v) return;                          /* skip NaN; the caller handles a NaN first element */
    if (v > pv) return;                          /* picked on an earlier pass */
    if (v == pv && i <= pi) return;              /* the last pick, or one before it */
    if (!pool_better(v, i, lv[POOL - 1], li[POOL - 1])) return;   /* fast path: one compare */

    float cv = v;
    uint32_t ci = i;
#pragma unroll
    for (int j = 0; j < POOL; ++j) {            /* the sift, static indices */
        if (pool_better(cv, ci, lv[j], li[j])) {
            const float tv = lv[j]; const uint32_t ti = li[j];
            lv[j] = cv; li[j] = ci; cv = tv; ci = ti;
        }
    }
}

/* The best POOL of the union of two lists, in the pool's order.
 *
 * A and B are descending, so C[j] = the better of A[j] and B[POOL-1-j] is a bitonic sequence
 * holding the union's top POOL; a bitonic sort in registers with static strides finishes it without
 * spilling. Both lanes of a shuffle pair compute the same result, so a butterfly of merges is a
 * wave reduction. */
template <int POOL>
__device__ __forceinline__ void pool_merge(const float* av, const uint32_t* ai,
                                           const float* bv, const uint32_t* bi,
                                           float* ov, uint32_t* oi) {
#pragma unroll
    for (int j = 0; j < POOL; ++j) {
        const int r = POOL - 1 - j;
        if (pool_better(av[j], ai[j], bv[r], bi[r])) { ov[j] = av[j]; oi[j] = ai[j]; }
        else                                        { ov[j] = bv[r]; oi[j] = bi[r]; }
    }
#pragma unroll
    for (int k = POOL; k > 1; k >>= 1) {
#pragma unroll
        for (int h = k / 2; h > 0; h >>= 1) {
#pragma unroll
            for (int j = 0; j < POOL; ++j) {
                const int o = j ^ h;
                if (o <= j) continue;
                /* One descending run: every compare-exchange moves the better entry to the lower
                 * index. */
                if (pool_better(ov[o], oi[o], ov[j], oi[j])) {
                    const float tv = ov[j]; const uint32_t ti = oi[j];
                    ov[j] = ov[o]; oi[j] = oi[o]; ov[o] = tv; oi[o] = ti;
                }
            }
        }
    }
}

/* One lane's pass over its columns of a row. STEP is 1 (scalar) or 4 (one vector load a step). */
template <int XDT, int STEP, int POOL>
__device__ __forceinline__ void pool_scan(const void* __restrict__ x, uint32_t n, uint32_t x_ld,
                                          float pv, uint32_t pi, float* lv, uint32_t* li) {
    const uint32_t tid = threadIdx.x;
    if (XDT == 0) {                                              /* bf16: four columns a 64-bit load */
        const uint16_t* __restrict__ p = (const uint16_t*)x + (size_t)blockIdx.x * x_ld;
        if (STEP == 4) {
            const uint32_t nw = n / 4;
            for (uint32_t w = tid; w < nw; w += kThreads) {
                const uint2 u = *(const uint2*)(p + (size_t)w * 4);
                const uint32_t lo = u.x, hi = u.y;
                pool_insert<POOL>(bf16_to_f32((uint16_t)(lo & 0xffffu)), 4 * w + 0, pv, pi, lv, li);
                pool_insert<POOL>(bf16_to_f32((uint16_t)(lo >> 16)),     4 * w + 1, pv, pi, lv, li);
                pool_insert<POOL>(bf16_to_f32((uint16_t)(hi & 0xffffu)), 4 * w + 2, pv, pi, lv, li);
                pool_insert<POOL>(bf16_to_f32((uint16_t)(hi >> 16)),     4 * w + 3, pv, pi, lv, li);
            }
            for (uint32_t c = nw * 4 + tid; c < n; c += kThreads)   /* the row's tail */
                pool_insert<POOL>(bf16_to_f32((uint16_t)ldg_u16(p + c)), c, pv, pi, lv, li);
        } else {
            for (uint32_t c = tid; c < n; c += kThreads)
                pool_insert<POOL>(bf16_to_f32((uint16_t)ldg_u16(p + c)), c, pv, pi, lv, li);
        }
    } else {                                                     /* f32: four columns a 128-bit load */
        const float* __restrict__ p = (const float*)x + (size_t)blockIdx.x * x_ld;
        if (STEP == 4) {
            const uint32_t nw = n / 4;
            for (uint32_t w = tid; w < nw; w += kThreads) {
                const float4 v = *(const float4*)(p + (size_t)w * 4);
                pool_insert<POOL>(v.x, 4 * w + 0, pv, pi, lv, li);
                pool_insert<POOL>(v.y, 4 * w + 1, pv, pi, lv, li);
                pool_insert<POOL>(v.z, 4 * w + 2, pv, pi, lv, li);
                pool_insert<POOL>(v.w, 4 * w + 3, pv, pi, lv, li);
            }
            for (uint32_t c = nw * 4 + tid; c < n; c += kThreads)
                pool_insert<POOL>(ldg_f32(p + c), c, pv, pi, lv, li);
        } else {
            for (uint32_t c = tid; c < n; c += kThreads)
                pool_insert<POOL>(ldg_f32(p + c), c, pv, pi, lv, li);
        }
    }
}

/* One pass over the row: every lane's list, a wave butterfly of merges, the eight wave lists in
 * LDS, and the block's POOL best left in LDS at [kWaves*POOL, ...) (in LDS because the extraction
 * walks them with a runtime index). Returns how many are real. */
template <int POOL>
__device__ __forceinline__ uint32_t pool_pass(const void* __restrict__ x, uint32_t n,
                                              uint32_t x_ld, int xdt, int vec,
                                              float pv, uint32_t pi,
                                              float* lds_v, uint32_t* lds_i) {
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    float lv[POOL], bv[POOL], cv[POOL];
    uint32_t li[POOL], bi[POOL], ci[POOL];
#pragma unroll
    for (int j = 0; j < POOL; ++j) { lv[j] = -INFINITY; li[j] = kPadCol; }

    if (vec) {
        if (xdt == 0) pool_scan<0, 4, POOL>(x, n, x_ld, pv, pi, lv, li);
        else          pool_scan<1, 4, POOL>(x, n, x_ld, pv, pi, lv, li);
    } else {
        if (xdt == 0) pool_scan<0, 1, POOL>(x, n, x_ld, pv, pi, lv, li);
        else          pool_scan<1, 1, POOL>(x, n, x_ld, pv, pi, lv, li);
    }

#define R3D_POOL_XOR(J)                                                                             \
    {                                                                                              \
        _Pragma("unroll") for (int q = 0; q < POOL; ++q) {                                        \
            bv[q] = r3d_xor_lane<J>(lv[q]); bi[q] = r3d_xor_lane_u32<J>(li[q]);                    \
        }                                                                                          \
        pool_merge<POOL>(lv, li, bv, bi, cv, ci);                                                        \
        _Pragma("unroll") for (int q = 0; q < POOL; ++q) { lv[q] = cv[q]; li[q] = ci[q]; }        \
    }
    R3D_POOL_XOR(16)
    R3D_POOL_XOR(8)
    R3D_POOL_XOR(4)
    R3D_POOL_XOR(2)
    R3D_POOL_XOR(1)
#undef R3D_POOL_XOR

    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < POOL; ++j) { lds_v[wave * POOL + j] = lv[j]; lds_i[wave * POOL + j] = li[j]; }
    }
    __syncthreads();

    uint32_t nout = 0;
    if (threadIdx.x == 0) {                        /* merge the eight wave lists with cursors */
        uint32_t cur[kWaves];
#pragma unroll
        for (int w = 0; w < kWaves; ++w) cur[w] = 0;
        for (int j = 0; j < POOL; ++j) {
            float bv = -INFINITY; uint32_t bi = kPadCol; int bw = -1;
            for (int w = 0; w < kWaves; ++w) {
                const uint32_t idx = (uint32_t)w * POOL + cur[w];
                const float v = lds_v[idx]; const uint32_t i = lds_i[idx];
                if (i == kPadCol) continue;
                if (bw < 0 || pool_better(v, i, bv, bi)) { bv = v; bi = i; bw = w; }
            }
            if (bw < 0) break;
            lds_v[kWaves * POOL + j] = bv; lds_i[kWaves * POOL + j] = bi;
            cur[bw]++; nout = (uint32_t)j + 1;
        }
    }
    __syncthreads();
    return nout;
}

}  /* namespace */

/* ================================================================== row_topk
 *
 * Exact per-row top-R, descending, ties to the lower column, with `vocab_off` added to every
 * reported id, and the optional [M, R, 2] pair plane an all-gather moves (an int32 id's bits in the
 * first f32 slot, the value in the second, id -1 for an unfilled slot). One scan per POOL entries
 * (see the file header).
 */
namespace {

template <int XDT, int POOL>
__global__ __launch_bounds__(kThreads) R3D_WAVES(4) void row_topk_kernel(
    const void* __restrict__ x, int32_t* __restrict__ idx, float* __restrict__ val,
    float* __restrict__ pairs, uint32_t n, uint32_t k, int vec, int32_t vocab_off,
    uint32_t x_ld, uint32_t idx_ld, uint32_t val_ld, uint32_t pairs_ld) {
    __shared__ float lds_v[kWaves * kPool + kPool];
    __shared__ uint32_t lds_i[kWaves * kPool + kPool];
    __shared__ float st[4];
    __shared__ uint32_t s2[2];

    const uint32_t row = blockIdx.x;
    float* __restrict__ vr = val + (size_t)row * val_ld;
    int32_t* __restrict__ ir = idx + (size_t)row * idx_ld;
    float* __restrict__ pr = pairs ? pairs + (size_t)row * pairs_ld : nullptr;

    /* topk_next's seed: a row whose first element is NaN reports (0, NaN) in every slot. */
    if (threadIdx.x == 0)
        st[0] = XDT == 0 ? bf16_to_f32((uint16_t)ldg_u16((const uint16_t*)x + (size_t)row * x_ld))
                         : ldg_f32((const float*)x + (size_t)row * x_ld);
    __syncthreads();
    /* The test reads shared memory after the barrier, so the whole block takes the same branch
     * and none of it reaches the pool passes' barriers. */
    if (st[0] != st[0]) {
        if (threadIdx.x == 0) {
            for (uint32_t r = 0; r < k; ++r) {
                ir[r] = (int32_t)vocab_off;
                vr[r] = st[0];
                if (pr) {
                    int32_t bits = (int32_t)vocab_off;
                    float v = st[0];
                    __builtin_memcpy(&pr[2 * r], &bits, 4);
                    pr[2 * r + 1] = v;
                }
            }
        }
        return;
    }

    float pv = INFINITY;
    uint32_t pi = 0xffffffffu;
    uint32_t done = 0;
    constexpr uint32_t kBlk = kWaves * POOL;      /* where pool_pass leaves the block's list */
    while (done < k) {
        __syncthreads();
        const uint32_t nout = pool_pass<POOL>(x, n, x_ld, XDT, vec, pv, pi, lds_v, lds_i);
        if (threadIdx.x == 0) {
            /* Only the first `nout` block entries were written this pass (the rest are stale), so
             * the count, not the pad sentinel, bounds the copy. */
            const uint32_t room = (k - done < POOL) ? k - done : (uint32_t)POOL;
            const uint32_t take = nout < room ? nout : room;
            uint32_t j = 0;
            for (; j < take; ++j) {
                const uint32_t col = lds_i[kBlk + j];
                if (col == kPadCol) break;
                const float v = lds_v[kBlk + j];
                const int32_t id = (int32_t)(col + (uint32_t)vocab_off);
                ir[done + j] = id;
                vr[done + j] = v;
                if (pr) { __builtin_memcpy(&pr[2 * (done + j)], &id, 4); pr[2 * (done + j) + 1] = v; }
                pv = v; pi = col;
            }
            for (; j < room; ++j) {                 /* holes to the end of the window */
                ir[done + j] = -1;
                vr[done + j] = -INFINITY;
                if (pr) {
                    const int32_t h = -1;
                    const float hinf = -INFINITY;
                    __builtin_memcpy(&pr[2 * (done + j)], &h, 4);
                    pr[2 * (done + j) + 1] = hinf;
                }
            }
            st[1] = pv;
            s2[0] = pi;
            s2[1] = nout == 0 ? k : done + take;
        }
        __syncthreads();
        pv = st[1]; pi = s2[0]; done = s2[1];
    }
}

/* One row's top_k: the request's, clamped to the buffer; the whole width when absent; none for a
 * greedy row, whose token is sample_argmax's. */
__device__ __forceinline__ uint32_t row_k(const RadSampleParams& sp, uint32_t n_cand) {
    if (sp.flags & RAD_SP_GREEDY) return 0;
    const int32_t k = sp.top_k;
    if (k <= 0 || (uint32_t)k > n_cand) return n_cand;
    return (uint32_t)k;
}

}  /* namespace */

/*   x      [M][N]      bf16 (x_dt 0) or f32 (x_dt 1), `x_ld` elements a row (0 = N)
 *   idx    [M][R]      i32, the column plus `vocab_off`, -1 for a slot the row could not fill
 *   val    [M][R]      f32, -infinity with the hole
 *   pairs  [M][2R]     f32, optional: (int32 id bits, value) as the all-gather moves it
 *
 * 0, -1, -2 a bad extent, -3 a pitch that does not describe the operand, -4 the launch failed. */
extern "C" int r3d_row_topk(long x, long idx, long val, long pairs, int M, int N, int R, int x_dt,
                            long long x_ld, long long idx_ld, long long val_ld, long long pairs_ld,
                            int vocab_off, long stream) {
    if (!x || !idx || !val) return -1;
    if (M <= 0 || N <= 0 || R <= 0) return -2;
    if (x_dt != 0 && x_dt != 1) return -2;
    if (!x_ld) x_ld = N;
    if (!idx_ld) idx_ld = R;
    if (!val_ld) val_ld = R;
    if (!pairs_ld) pairs_ld = 2 * R;
    if (x_ld < N || idx_ld < R || val_ld < R || (pairs && pairs_ld < 2 * R)) return -3;
    if (M > 0x7fffffff || N > 0x7fffffff || R > 0x7fffffff) return -2;

    /* The vector load needs a row start that is 16 bytes (f32) or 8 bytes (bf16) aligned. Any pitch
     * that is a multiple of 4 elements does it for every row; the base is the caller's to align and
     * is checked here, so a slice handed in at an odd column takes the scalar path and is right. */
    const int es = x_dt == 0 ? 2 : 4;
    const int vec = (x_ld % 4 == 0) && ((x % (4 * (long long)es)) == 0);

    const dim3 grid((unsigned)M), block(kThreads);

    /* The list depth is the pass width: R 16 (a router's top_k, the band's row) sifts 16 deep, a
     * wide candidate set 64. Beyond 64 the row is scanned again, so the depth buys passes, not
     * per-element work. */
    const int pool = R <= 16 ? 16 : (R <= 32 ? 32 : 64);
#define R3D_RTK(XDT, POOL)                                                                        \
    R3D_KLAUNCH(R3D_KNAME(row_topk_kernel<XDT, POOL>), grid, block,                              \
                (hipStream_t)(uintptr_t)stream, (const void*)x, (int32_t*)idx, (float*)val,       \
                (float*)pairs, (uint32_t)N, (uint32_t)R, vec, (int32_t)vocab_off,                 \
                (uint32_t)x_ld, (uint32_t)idx_ld, (uint32_t)val_ld, (uint32_t)pairs_ld)
    if (x_dt == 0) {
        if (pool == 16) { R3D_RTK(0, 16); }
        else if (pool == 32) { R3D_RTK(0, 32); }
        else { R3D_RTK(0, 64); }
    } else {
        if (pool == 16) { R3D_RTK(1, 16); }
        else if (pool == 32) { R3D_RTK(1, 32); }
        else { R3D_RTK(1, 64); }
    }
#undef R3D_RTK
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ================================================================== sample_topk
 *
 * The row's top_k largest logits into the fixed-width candidate set, descending, ties to the lower
 * token id, with `cand_idx` -1 and `cand_val` -infinity for a hole. Unlike row_topk, the width is
 * the buffer's (a declare-time bound) and the row's top_k narrows within it (top_k 40 in a 1024-wide
 * buffer is 40 candidates and 984 holes), and a greedy row takes none.
 *
 * `cand_val` carries logits, not probabilities: each narrowing stage takes its own softmax over the
 * set, so the stages compose in any order.
 */
template <int POOL>
__global__ __launch_bounds__(kThreads) R3D_WAVES(4) void sample_topk_kernel(
    const float* __restrict__ lg, const RadSampleParams* __restrict__ pr, int32_t* __restrict__ ci,
    float* __restrict__ cv, int32_t* __restrict__ pairs, uint32_t n_vocab, uint32_t n_cand, int vec,
    int32_t vocab_off, uint32_t lg_ld, uint32_t ci_ld, uint32_t cv_ld, uint32_t pairs_ld) {
    __shared__ float lds_v[kWaves * kPool + kPool];
    __shared__ uint32_t lds_i[kWaves * kPool + kPool];
    __shared__ float st[4];
    __shared__ uint32_t s2[2];

    const uint32_t row = blockIdx.x;
    const uint32_t k = row_k(pr[row], n_cand);
    int32_t* __restrict__ ir = ci + (size_t)row * ci_ld;
    float* __restrict__ vr = cv + (size_t)row * cv_ld;
    int32_t* __restrict__ pp = pairs ? pairs + (size_t)row * pairs_ld : nullptr;

    /* The whole buffer is written every step, since a stale candidate from the previous step would
     * be drawn. */
    auto hole = [&](uint32_t r) {
        ir[r] = -1;
        vr[r] = -INFINITY;
        if (pp) { pp[2 * r] = (int32_t)0xffffffffu; pp[2 * r + 1] = (int32_t)0xff800000; }
    };

    const float x0 = ldg_f32(lg + (size_t)row * lg_ld);
    if (x0 != x0) {                                /* topk_next's seed, as in row_topk */
        for (uint32_t r = 0; r < n_cand; ++r) {
            if (r >= k) { hole(r); continue; }
            ir[r] = vocab_off;
            vr[r] = x0;
            if (pp) { pp[2 * r] = vocab_off; pp[2 * r + 1] = *(const int32_t*)&x0; }
        }
        return;
    }

    float pv = INFINITY;
    uint32_t pi = 0xffffffffu;
    uint32_t done = 0;
    constexpr uint32_t kBlk = kWaves * POOL;
    while (done < k) {
        __syncthreads();
        const uint32_t nout = pool_pass<POOL>(lg, n_vocab, lg_ld, 1, vec, pv, pi, lds_v, lds_i);
        if (threadIdx.x == 0) {
            /* As in row_topk: `nout` bounds the copy. */
            const uint32_t room = (k - done < POOL) ? k - done : (uint32_t)POOL;
            const uint32_t take = nout < room ? nout : room;
            uint32_t j = 0;
            for (; j < take; ++j) {
                const uint32_t col = lds_i[kBlk + j];
                if (col == kPadCol) break;
                const float v = lds_v[kBlk + j];
                const int32_t id = (int32_t)(col + (uint32_t)vocab_off);
                ir[done + j] = (int32_t)col;      /* cand_idx is local to this row of logits */
                vr[done + j] = v;
                if (pp) { pp[2 * (done + j)] = id; pp[2 * (done + j) + 1] = *(const int32_t*)&v; }
                pv = v; pi = col;
            }
            for (; j < room; ++j) hole(done + j);   /* holes to the end of the window */
            s2[0] = pi;
            s2[1] = nout == 0 ? k : done + take;
            st[1] = pv;
        }
        __syncthreads();
        pv = st[1]; pi = s2[0]; done = s2[1];
    }
    for (uint32_t r = done + threadIdx.x; r < n_cand; r += kThreads) hole(r);
}

/*   lg      [M][n_vocab] f32; `lg_ld` a row (0 = n_vocab)
 *   params  [M]          24 words a row
 *   ci      [M][n_cand]  i32, local columns, -1 for a hole
 *   cv      [M][n_cand]  f32, the logit, -infinity with the hole
 *   pairs   [M][2*n_cand] i32, optional: (global id, logit bits), 0xffffffff for a hole
 *
 * The candidate width is ci's last dimension, as libref takes it; the row's top_k only narrows
 * inside it. 0, -1, -2, -3, -4. */
extern "C" int r3d_sample_topk(long lg, long params, long ci, long cv, long pairs, int M,
                               int n_vocab, int n_cand, long long lg_ld, long long ci_ld,
                               long long cv_ld, long long pairs_ld, int vocab_off, long stream) {
    if (!lg || !params || !ci || !cv) return -1;
    if (M <= 0 || n_vocab <= 0 || n_cand <= 0 || n_cand > kCandMax) return -2;
    if (!lg_ld) lg_ld = n_vocab;
    if (!ci_ld) ci_ld = n_cand;
    if (!cv_ld) cv_ld = n_cand;
    if (!pairs_ld) pairs_ld = 2 * n_cand;
    if (lg_ld < n_vocab || ci_ld < n_cand || cv_ld < n_cand || (pairs && pairs_ld < 2 * n_cand))
        return -3;
    if (M > 0x7fffffff || n_vocab > 0x7fffffff) return -2;
    const int vec = (lg_ld % 4 == 0) && ((lg % 16) == 0);

    const dim3 grid((unsigned)M), block(kThreads);
    const int pool = n_cand <= 16 ? 16 : (n_cand <= 32 ? 32 : 64);
#define R3D_STK(POOL)                                                                             \
    R3D_KLAUNCH(R3D_KNAME(sample_topk_kernel<POOL>), grid, block,                                \
                (hipStream_t)(uintptr_t)stream, (const float*)lg,                                \
                (const RadSampleParams*)params, (int32_t*)ci, (float*)cv, (int32_t*)pairs,        \
                (uint32_t)n_vocab, (uint32_t)n_cand, vec, (int32_t)vocab_off,                     \
                (uint32_t)lg_ld, (uint32_t)ci_ld, (uint32_t)cv_ld, (uint32_t)pairs_ld)
    if (pool == 16) { R3D_STK(16); }
    else if (pool == 32) { R3D_STK(32); }
    else { R3D_STK(64); }
#undef R3D_STK
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ================================================================== the narrowing stages and the draw
 *
 * One block a row of the candidate set, the row in LDS (five arrays of up to 1024 f32, 20 KiB):
 *
 *   - the max is libref's strict `if (v > mx) mx = v` from -infinity, so a hole, a masked
 *     candidate at -infinity and a NaN are never the max;
 *   - exp(v - mx) is computed for all candidates in parallel into LDS;
 *   - every sum (the normaliser, the top-p prefix, the entropy, the inverse-CDF scan) is libref's
 *     loop in index order on lane 0. A parallel prefix sum would round differently and could draw a
 *     different token; the serial loop is 1024 adds, small next to the transcendentals.
 *
 * Every stage leaves a greedy row alone, as the reference does.
 */
namespace {

enum { ST_TOPP = 0, ST_MINP = 1, ST_TYPICAL = 2, ST_XTC = 3, ST_PICK = 4 };

/* A minimum to match, for typical's cut: the smallest qualifying distance. */
__device__ __forceinline__ float block_min(float v, float* lds) {
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
#define R3D_MIN_XOR(M) v = fminf(v, r3d_xor_lane<M>(v))
    R3D_MIN_XOR(16); R3D_MIN_XOR(8); R3D_MIN_XOR(4); R3D_MIN_XOR(2); R3D_MIN_XOR(1);
#undef R3D_MIN_XOR
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float t = lds[0];
#pragma unroll
    for (uint32_t w = 1; w < kWaves; ++w) t = fminf(t, lds[w]);
    return t;
}

template <int STAGE>
__global__ __launch_bounds__(kThreads) R3D_WAVES(4) void narrow_kernel(
    int32_t* __restrict__ ci_p, float* __restrict__ cv_p, const RadSampleParams* __restrict__ pr,
    int32_t* __restrict__ tok, uint32_t n_cand, uint32_t ci_ld, uint32_t cv_ld) {
    __shared__ int32_t s_ci[kCandMax];
    __shared__ float s_cv[kCandMax];
    __shared__ float s_ev[kCandMax];
    __shared__ float s_q[kCandMax];
    __shared__ float s_d[kCandMax];
    __shared__ float red[kWaves];
    __shared__ float st[4];
    __shared__ uint32_t su[4];

    const uint32_t row = blockIdx.x;
    const RadSampleParams& sp = pr[row];
    if (sp.flags & RAD_SP_GREEDY) return;         /* the row's token is the argmax's, untouched */

    int32_t* __restrict__ ir = ci_p + (size_t)row * ci_ld;
    float* __restrict__ vr = cv_p + (size_t)row * cv_ld;
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) { s_ci[i] = ir[i]; s_cv[i] = vr[i]; }
    __syncthreads();

    /* mx over live candidates with a strict `>` from -infinity, excluding holes, masked candidates
     * and NaN, as the reference's loop does. */
    float mine = -INFINITY;
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads)
        if (s_ci[i] >= 0 && s_cv[i] > mine) mine = s_cv[i];
    const float mx = block_max(mine, red);
    if (mx == -INFINITY) {
        if (STAGE == ST_PICK && threadIdx.x == 0) tok[row] = -1;
        return;                                   /* an empty row: nothing to narrow, nothing to draw */
    }

    if (STAGE == ST_MINP) {
        const float p = sp.min_p;
        if (!(p > 0.0f)) return;
        for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
            if (s_ci[i] < 0) continue;
            const float v = s_cv[i];
            if (v == mx) continue;                /* the max always survives */
            if (ref_exp(v - mx) < p) { s_ci[i] = -1; s_cv[i] = -INFINITY; }
        }
    } else {
        float inv = 0.0f;
        /* exp(v - mx) for every live candidate, in parallel, in place in LDS. */
        for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads)
            s_ev[i] = (s_ci[i] >= 0) ? ref_exp(s_cv[i] - mx) : 0.0f;
        __syncthreads();
        if (threadIdx.x == 0) {                   /* the normaliser: libref's loop, index order */
            float sum = 0.0f;
            for (uint32_t i = 0; i < n_cand; ++i) sum += s_ev[i];
            st[0] = sum > 0.0f ? 1.0f / sum : 0.0f;
        }
        __syncthreads();
        inv = st[0];

        if (STAGE == ST_TOPP) {
            const float p = sp.top_p;
            if (!(p < 1.0f) || !(p > 0.0f)) return;
            if (threadIdx.x == 0) {               /* the shortest prefix reaching p, at least one */
                float cum = 0.0f;
                bool done = false;
                for (uint32_t i = 0; i < n_cand; ++i) {
                    if (s_ci[i] < 0) continue;
                    if (done) { s_ci[i] = -1; s_cv[i] = -INFINITY; continue; }
                    cum += s_ev[i] * inv;
                    if (cum >= p) done = true;
                }
            }
            __syncthreads();
        } else if (STAGE == ST_XTC) {
            const float pq = sp.xtc_probability, th = sp.xtc_threshold;
            if (!(pq > 0.0f) || !(th > 0.0f)) return;
            if (threadIdx.x == 0)
                su[1] = uniform01(sp.seed ^ RAD_SAMPLE_KEY_XTC, sp.pos) >= pq ? 1u : 0u;
            __syncthreads();
            if (su[1]) return;                    /* the coin says leave the set alone this step */
            float cnt = 0.0f, last = -1.0f;
            for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
                if (s_ci[i] < 0) continue;
                if (s_ev[i] * inv >= th) { cnt += 1.0f; last = (float)i; }
            }
            /* Count and last index are order-free (a sum of small integers and a max), so no serial
             * loop is needed; candidates are re-tested below from the same `s_ev`. */
            const float n_above = block_sum<kThreads>(cnt, red);
            const float hi = block_max(last, red);
            if (n_above < 2.0f) return;           /* needs two to exclude all but the least likely */
            const int32_t keep = (int32_t)hi;
            for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
                if (s_ci[i] < 0 || (int32_t)i == keep) continue;
                if (s_ev[i] * inv >= th) { s_ci[i] = -1; s_cv[i] = -INFINITY; }
            }
            __syncthreads();
        } else if (STAGE == ST_TYPICAL) {         /* overwrites s_ev with log(q); only the pick
                                                   * reads s_ev afterwards, and it never runs this */
            const float p = sp.typical_p;
            if (!(p < 1.0f) || !(p > 0.0f)) return;
            for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads)
                s_q[i] = (s_ci[i] >= 0) ? s_ev[i] * inv : 0.0f;
            __syncthreads();
            for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads)
                s_ev[i] = (s_q[i] > 0.0f) ? ref_log(s_q[i]) : 0.0f;   /* log(q) once, used twice */
            __syncthreads();
            if (threadIdx.x == 0) {               /* the entropy, in index order */
                float H = 0.0f;
                for (uint32_t i = 0; i < n_cand; ++i)
                    if (s_q[i] > 0.0f) H -= s_q[i] * s_ev[i];
                st[1] = H;
            }
            __syncthreads();
            const float H = st[1];
            for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads)
                s_d[i] = (s_q[i] > 0.0f) ? fabsf(-s_ev[i] - H) : 0.0f;
            __syncthreads();
            /* The smallest distance whose mass reaches p. A lane owns one candidate's distance and
             * sums the mass at or below it in index order; ties are kept, so the set may overshoot
             * p but does not depend on candidate order. */
            float cut = INFINITY;
            for (uint32_t c = threadIdx.x; c < n_cand; c += kThreads) {
                if (!(s_q[c] > 0.0f)) continue;
                float mass = 0.0f;
                for (uint32_t i = 0; i < n_cand; ++i)
                    if (s_q[i] > 0.0f && s_d[i] <= s_d[c]) mass += s_q[i];
                if (mass >= p && s_d[c] < cut) cut = s_d[c];
            }
            const float cuth = block_min(cut, red);
            if (cuth == INFINITY) return;         /* no qualifying band: the reference leaves it be */
            for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
                if (s_ci[i] < 0) continue;
                if (!(s_q[i] > 0.0f) || s_d[i] > cuth) { s_ci[i] = -1; s_cv[i] = -INFINITY; }
            }
            __syncthreads();
        }

        if (STAGE == ST_PICK) {
            if (threadIdx.x == 0) {               /* the inverse CDF, in candidate order */
                const float u = uniform01(sp.seed ^ RAD_SAMPLE_KEY_PICK, sp.pos);
                float cum = 0.0f;
                int32_t chosen = -1, last = -1;
                for (uint32_t i = 0; i < n_cand; ++i) {
                    const int32_t id = s_ci[i];
                    if (id < 0) continue;
                    const float v = s_cv[i];
                    if (v > -INFINITY) last = id; /* a masked tail is never the fallback */
                    cum += s_ev[i] * inv;
                    if (u < cum) { chosen = id; break; }
                }
                tok[row] = chosen >= 0 ? chosen : last;
            }
            return;                               /* the draw writes no candidate plane */
        }
    }
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) { ir[i] = s_ci[i]; vr[i] = s_cv[i]; }
}

}  /* namespace */

/*   ci      [M][n_cand] i32, INOUT: -1 marks a dropped candidate
 *   cv      [M][n_cand] f32, INOUT: the logit, -infinity with a drop
 *   params  [M]         24 words a row: the stage's own scalar, the seed and the pos
 *
 * The set is not re-sorted: it arrives descending from sample_topk and the stages skip holes. A
 * greedy row is left exactly as it was. 0, -1, -2, -3, -4. */
namespace {
template <int STAGE>
int narrow_launch(long ci, long cv, long params, int M, int n_cand, long long ci_ld,
                  long long cv_ld, long stream, long tok) {
    if (!ci || !cv || !params) return -1;
    if (M <= 0 || n_cand <= 0 || n_cand > kCandMax) return -2;
    if (!ci_ld) ci_ld = n_cand;
    if (!cv_ld) cv_ld = n_cand;
    if (ci_ld < n_cand || cv_ld < n_cand) return -3;
    if (M > 0x7fffffff || n_cand > 0x7fffffff) return -2;

    const dim3 grid((unsigned)M), block(kThreads);
    R3D_KLAUNCH(R3D_KNAME(narrow_kernel<STAGE>), grid, block, (hipStream_t)(uintptr_t)stream,
                (int32_t*)ci, (float*)cv, (const RadSampleParams*)params, (int32_t*)tok,
                (uint32_t)n_cand, (uint32_t)ci_ld, (uint32_t)cv_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}
}  /* namespace */

extern "C" int r3d_sample_topp(long ci, long cv, long params, int M, int n_cand, long long ci_ld,
                               long long cv_ld, long stream) {
    return narrow_launch<ST_TOPP>(ci, cv, params, M, n_cand, ci_ld, cv_ld, stream, 0);
}
extern "C" int r3d_sample_minp(long ci, long cv, long params, int M, int n_cand, long long ci_ld,
                               long long cv_ld, long stream) {
    return narrow_launch<ST_MINP>(ci, cv, params, M, n_cand, ci_ld, cv_ld, stream, 0);
}
extern "C" int r3d_sample_typical(long ci, long cv, long params, int M, int n_cand,
                                  long long ci_ld, long long cv_ld, long stream) {
    return narrow_launch<ST_TYPICAL>(ci, cv, params, M, n_cand, ci_ld, cv_ld, stream, 0);
}
extern "C" int r3d_sample_xtc(long ci, long cv, long params, int M, int n_cand, long long ci_ld,
                              long long cv_ld, long stream) {
    return narrow_launch<ST_XTC>(ci, cv, params, M, n_cand, ci_ld, cv_ld, stream, 0);
}

/* The inverse-CDF draw over the candidate set: the splitmix64 uniform at (seed ^ PICK, pos), the
 * normaliser and the accumulation in candidate order, and the fallback to the last candidate above
 * -infinity when the f32 probabilities sum to less than u. Writes `token` only. */
extern "C" int r3d_sample_pick(long ci, long cv, long params, long tok, int M, int n_cand,
                               long long ci_ld, long long cv_ld, long stream) {
    if (!tok) return -1;
    return narrow_launch<ST_PICK>(ci, cv, params, M, n_cand, ci_ld, cv_ld, stream, tok);
}

/* ================================================================== the plugin ABI
 *
 * Nine shims, one per op. Dtypes come from the tensors and geometry from the operands; `M` is the
 * band's ceiling, never the row count. The params plane is checked as libref checks it (a 32-bit
 * dtype and enough bytes for M rows) and must be contiguous, since the kernels index it flat.
 */

#define R3D_NEED(expr) do { const int r3d_s = (expr); if (r3d_s) return R3D_NO(r3d_s); } while (0)

/* A launcher checked the geometry and answered with a code; translate it to a RAD status. */
static inline int r3d_launch_status(int e) {
    switch (e) {
        case  0: return RAD_OK;
        case -1: return RAD_E_INVAL;
        case -2: return RAD_E_SHAPE;
        case -3: return RAD_E_STRIDE;
        default: return RAD_E_DEVICE;
    }
}

static inline int sp_is_word(const RadTensor* t) {
    return t && (t->dtype == RAD_U32 || t->dtype == RAD_I32 || t->dtype == RAD_F32);
}

/* The params plane: [M, 24] words. The dtype test matters: a 16-bit view passes a byte-count check
 * and then misreads every row. */
static inline int params_ok(const RadTensor* t, long long M) {
    if (!t || !t->data) return RAD_E_INVAL;
    if (!sp_is_word(t)) return RAD_E_DTYPE;
    if (r3d_numel(t) * 4 < M * (long long)sizeof(RadSampleParams)) return RAD_E_SHAPE;
    if (!r3d_contig(t)) return RAD_E_STRIDE;
    if (!r3d_aligned(r3d_p(t), 8)) return RAD_E_ALIGN;   /* the seed and pos are 8-byte fields */
    return RAD_OK;
}

static inline int logits_ok(const RadTensor* t) {
    if (!t || !t->data) return RAD_E_INVAL;
    if (!r3d_is_f32(t)) return RAD_E_DTYPE;
    if (!r3d_aligned(r3d_p(t), 4)) return RAD_E_ALIGN;
    return RAD_OK;
}

/* ------------------------------------------------------------------ row_topk */

enum { RTK_X = 0, RTK_IDX, RTK_VAL, RTK_PAIRS };

extern "C" int r3d_rad_row_topk(const RadArgs* a, RadStream stream) {
    const RadTensor* x   = r3d_opt(a, RTK_X);
    const RadTensor* idx = r3d_opt(a, RTK_IDX);
    const RadTensor* val = r3d_opt(a, RTK_VAL);
    const RadTensor* pr  = r3d_opt(a, RTK_PAIRS);
    if (!x || !x->data || !idx || !idx->data || !val || !val->data) return R3D_NO(RAD_E_INVAL);
    const int xdt = r3d_is_bf16(x) ? 0 : (r3d_is_f32(x) ? 1 : -1);
    if (xdt < 0) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_i32(idx)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(val)) return R3D_NO(RAD_E_DTYPE);
    if (pr && pr->data && !r3d_is_f32(pr)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_aligned(r3d_p(x), 4) || !r3d_aligned(r3d_p(idx), 4) || !r3d_aligned(r3d_p(val), 4))
        return R3D_NO(RAD_E_ALIGN);

    const long long N = r3d_geti_or(a, "N", r3d_dim(x, (int)x->rank - 1));
    const long long R = r3d_geti_or(a, "R", 1);
    if (N <= 0 || R <= 0) return R3D_NO(RAD_E_SHAPE);
    long long rows = 0, irows = 0, vrows = 0, prows = 0;
    const long long x_ld = r3d_rows_pitch(x, N, &rows);
    const long long idx_ld = r3d_rows_pitch(idx, R, &irows);
    const long long val_ld = r3d_rows_pitch(val, R, &vrows);
    if (x_ld < 0 || idx_ld < 0 || val_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (rows <= 0 || rows > 0x7fffffffLL || N > 0x7fffffffLL || R > 0x7fffffffLL)
        return R3D_NO(RAD_E_SHAPE);
    if (irows < rows || vrows < rows) return R3D_NO(RAD_E_SHAPE);
    long long pairs = 0, pairs_ld = 0;
    if (pr && pr->data) {
        pairs_ld = r3d_rows_pitch(pr, 2 * R, &prows);
        if (pairs_ld < 0) return R3D_NO(RAD_E_STRIDE);
        if (prows < rows || r3d_numel(pr) < rows * 2 * R) return R3D_NO(RAD_E_SHAPE);
        pairs = r3d_addr(pr);
    }
    return r3d_launch_status(r3d_row_topk(r3d_addr(x), r3d_addr(idx), r3d_addr(val), pairs,
                                         (int)rows, (int)N, (int)R, xdt, x_ld, idx_ld, val_ld,
                                         pairs_ld, (int)r3d_geti_or(a, "vocab_off", 0),
                                         (long)(uintptr_t)stream));
}

/* ------------------------------------------------------------------ argmax and temp */

enum { ARG_LG = 0, ARG_PARAMS, ARG_TOK };

extern "C" int r3d_rad_sample_argmax(const RadArgs* a, RadStream stream) {
    const RadTensor* lg = r3d_opt(a, ARG_LG);
    const RadTensor* pr = r3d_opt(a, ARG_PARAMS);
    const RadTensor* tk = r3d_opt(a, ARG_TOK);
    R3D_NEED(logits_ok(lg));
    if (!tk || !tk->data) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_i32(tk)) return R3D_NO(RAD_E_DTYPE);
    const long long n_vocab = r3d_geti_or(a, "n_vocab", r3d_dim(lg, (int)lg->rank - 1));
    if (n_vocab <= 0) return R3D_NO(RAD_E_SHAPE);
    long long rows = 0;
    const long long lg_ld = r3d_rows_pitch(lg, n_vocab, &rows);
    if (lg_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (rows <= 0 || rows > 0x7fffffffLL || n_vocab > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(tk) < rows) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(tk) || !r3d_aligned(r3d_p(tk), 4)) return R3D_NO(RAD_E_SHAPE);
    /* argmax reads nothing from params, so a missing plane is not a refusal. */
    if (pr && pr->data) R3D_NEED(params_ok(pr, rows));
    return r3d_launch_status(r3d_sample_argmax(r3d_addr(lg), r3d_addr(tk), (int)rows, (int)n_vocab,
                                              lg_ld, (long)(uintptr_t)stream));
}

extern "C" int r3d_rad_sample_temp(const RadArgs* a, RadStream stream) {
    const RadTensor* lg = r3d_opt(a, ARG_LG);
    const RadTensor* pr = r3d_opt(a, ARG_PARAMS);
    R3D_NEED(logits_ok(lg));
    const long long n_vocab = r3d_geti_or(a, "n_vocab", r3d_dim(lg, (int)lg->rank - 1));
    if (n_vocab <= 0) return R3D_NO(RAD_E_SHAPE);
    long long rows = 0;
    const long long lg_ld = r3d_rows_pitch(lg, n_vocab, &rows);
    if (lg_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (rows <= 0 || rows > 0x7fffffffLL || n_vocab > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);
    R3D_NEED(params_ok(pr, rows));
    return r3d_launch_status(r3d_sample_temp(r3d_addr(lg), r3d_addr(pr), (int)rows, (int)n_vocab,
                                            lg_ld, (long)(uintptr_t)stream));
}

/* ------------------------------------------------------------------ sample_topk */

enum { TK_LG = 0, TK_PARAMS, TK_CI, TK_CV, TK_PAIRS };

extern "C" int r3d_rad_sample_topk(const RadArgs* a, RadStream stream) {
    const RadTensor* lg = r3d_opt(a, TK_LG);
    const RadTensor* pr = r3d_opt(a, TK_PARAMS);
    const RadTensor* ci = r3d_opt(a, TK_CI);
    const RadTensor* cv = r3d_opt(a, TK_CV);
    const RadTensor* pp = r3d_opt(a, TK_PAIRS);
    R3D_NEED(logits_ok(lg));
    if (!ci || !ci->data || !cv || !cv->data) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_i32(ci)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(cv)) return R3D_NO(RAD_E_DTYPE);
    if (pp && pp->data && !r3d_is_i32(pp)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_aligned(r3d_p(ci), 4) || !r3d_aligned(r3d_p(cv), 4)) return R3D_NO(RAD_E_ALIGN);

    const long long n_vocab = r3d_geti_or(a, "n_vocab", r3d_dim(lg, (int)lg->rank - 1));
    /* The width is the buffer's declare-time bound; the row's top_k narrows inside it. */
    const long long n_cand = r3d_dim(ci, (int)ci->rank - 1);
    if (n_vocab <= 0 || n_cand <= 0 || n_cand > kCandMax) return R3D_NO(RAD_E_SHAPE);
    long long rows = 0, cirows = 0, cvrows = 0, prows = 0;
    const long long lg_ld = r3d_rows_pitch(lg, n_vocab, &rows);
    const long long ci_ld = r3d_rows_pitch(ci, n_cand, &cirows);
    const long long cv_ld = r3d_rows_pitch(cv, n_cand, &cvrows);
    if (lg_ld < 0 || ci_ld < 0 || cv_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (rows <= 0 || rows > 0x7fffffffLL || n_vocab > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);
    if (cirows < rows || cvrows < rows || r3d_numel(cv) < rows * n_cand) return R3D_NO(RAD_E_SHAPE);
    R3D_NEED(params_ok(pr, rows));
    long long pairs = 0, pairs_ld = 0;
    if (pp && pp->data) {
        pairs_ld = r3d_rows_pitch(pp, 2 * n_cand, &prows);
        if (pairs_ld < 0) return R3D_NO(RAD_E_STRIDE);
        if (prows < rows || r3d_numel(pp) < rows * 2 * n_cand) return R3D_NO(RAD_E_SHAPE);
        pairs = r3d_addr(pp);
    }
    return r3d_launch_status(r3d_sample_topk(r3d_addr(lg), r3d_addr(pr), r3d_addr(ci), r3d_addr(cv),
                                            pairs, (int)rows, (int)n_vocab, (int)n_cand, lg_ld,
                                            ci_ld, cv_ld, pairs_ld,
                                            (int)r3d_geti_or(a, "vocab_off", 0),
                                            (long)(uintptr_t)stream));
}

/* ------------------------------------------------------------------ the narrowing stages and the draw */

enum { NW_CI = 0, NW_CV, NW_PARAMS, NW_TOK };

struct CandRow {
    long long n_cand, rows, ci_ld, cv_ld;
};

static inline int cand_row_ok(const RadArgs* a, const RadTensor* ci, const RadTensor* cv,
                              const RadTensor* pr, long long* M_hint, CandRow* g) {
    if (!ci || !ci->data || !cv || !cv->data) return RAD_E_INVAL;
    if (!r3d_is_i32(ci)) return RAD_E_DTYPE;
    if (!r3d_is_f32(cv)) return RAD_E_DTYPE;
    if (!r3d_aligned(r3d_p(ci), 4) || !r3d_aligned(r3d_p(cv), 4)) return RAD_E_ALIGN;
    const long long n_cand = r3d_geti_or(a, "n_cand", r3d_dim(ci, (int)ci->rank - 1));
    if (n_cand <= 0 || n_cand > kCandMax) return RAD_E_SHAPE;
    long long cirows = 0, cvrows = 0;
    const long long ci_ld = r3d_rows_pitch(ci, n_cand, &cirows);
    const long long cv_ld = r3d_rows_pitch(cv, n_cand, &cvrows);
    if (ci_ld < 0 || cv_ld < 0) return RAD_E_STRIDE;
    /* The step's rows come from the operand; the band key `M` is only checked as an upper bound. */
    const long long rows = cirows;
    if (rows <= 0 || rows > 0x7fffffffLL) { r3d_trace_desc("ci", ci);
        if (r3d_trace_on())
            std::fprintf(stderr, "[r3d] sample: %lld candidate rows, n_cand %lld\n", rows, n_cand);
        return RAD_E_SHAPE; }
    if (M_hint && *M_hint > 0 && rows > *M_hint) { r3d_trace_desc("ci", ci);
        if (r3d_trace_on())
            std::fprintf(stderr, "[r3d] sample: %lld rows in a band that declared M <= %lld\n",
                         rows, *M_hint);
        return RAD_E_SHAPE; }
    if (cvrows < rows) { r3d_trace_desc("cv", cv);
        if (r3d_trace_on())
            std::fprintf(stderr, "[r3d] sample: cv has %lld rows, ci has %lld\n", cvrows, rows);
        return RAD_E_SHAPE; }
    if (r3d_numel(cv) < rows * n_cand) return RAD_E_SHAPE;
    const int s = params_ok(pr, rows);
    if (s) return s;
    g->n_cand = n_cand; g->rows = rows; g->ci_ld = ci_ld; g->cv_ld = cv_ld;
    return RAD_OK;
}

static int narrow_shim(const RadArgs* a, RadStream stream,
                       int (*launch)(long, long, long, int, int, long long, long long, long)) {
    const RadTensor* ci = r3d_opt(a, NW_CI);
    const RadTensor* cv = r3d_opt(a, NW_CV);
    const RadTensor* pr = r3d_opt(a, NW_PARAMS);
    CandRow g{};
    long long M_hint = r3d_geti_or(a, "M", 0);
    const int s = cand_row_ok(a, ci, cv, pr, &M_hint, &g);
    if (s) return R3D_NO(s);
    return r3d_launch_status(launch(r3d_addr(ci), r3d_addr(cv), r3d_addr(pr), (int)g.rows,
                                   (int)g.n_cand, g.ci_ld, g.cv_ld, (long)(uintptr_t)stream));
}

extern "C" int r3d_rad_sample_topp(const RadArgs* a, RadStream s) {
    return narrow_shim(a, s, r3d_sample_topp);
}
extern "C" int r3d_rad_sample_minp(const RadArgs* a, RadStream s) {
    return narrow_shim(a, s, r3d_sample_minp);
}
extern "C" int r3d_rad_sample_typical(const RadArgs* a, RadStream s) {
    return narrow_shim(a, s, r3d_sample_typical);
}
extern "C" int r3d_rad_sample_xtc(const RadArgs* a, RadStream s) {
    return narrow_shim(a, s, r3d_sample_xtc);
}

extern "C" int r3d_rad_sample_pick(const RadArgs* a, RadStream stream) {
    const RadTensor* ci = r3d_opt(a, NW_CI);
    const RadTensor* cv = r3d_opt(a, NW_CV);
    const RadTensor* pr = r3d_opt(a, NW_PARAMS);
    const RadTensor* tk = r3d_opt(a, NW_TOK);
    if (!tk || !tk->data) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_i32(tk)) return R3D_NO(RAD_E_DTYPE);
    CandRow g{};
    long long M_hint = r3d_geti_or(a, "M", 0);
    const int s = cand_row_ok(a, ci, cv, pr, &M_hint, &g);
    if (s) return R3D_NO(s);
    if (r3d_numel(tk) < g.rows) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(tk) || !r3d_aligned(r3d_p(tk), 4)) return R3D_NO(RAD_E_SHAPE);
    return r3d_launch_status(r3d_sample_pick(r3d_addr(ci), r3d_addr(cv), r3d_addr(pr), r3d_addr(tk),
                                            (int)g.rows, (int)g.n_cand, g.ci_ld, g.cv_ld,
                                            (long)(uintptr_t)stream));
}
