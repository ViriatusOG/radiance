/* r3d_sample_f32.hip -- the sampler: the two full-vocabulary reductions, the candidate set, the
 * seeded draw and the four narrowing stages.
 *
 * The arithmetic is libref's, transcribed -- the reference's sampler is the law and the ORDER of
 * its passes is the point. `abi/rad_sample.h` is the wire format: every scalar (`temp`, `top_k`,
 * `top_p`, `min_p`, `typical_p`, `xtc_*`, `seed`, `pos`, the flags, the history window) is DATA, one
 * 96-byte row per sampled position, because a parameter is geometry and frozen at declare, and two
 * rows of one batch with different top_p must resolve to one kernel. `param_rows` below is that
 * check, and it is the dtype check libref states: the plane is 32-bit words, and a 16-bit view
 * passes a byte-count test and then misreads every row.
 *
 * ============================== what has to agree, and with what ==============================
 *
 * The draws must REPRODUCE, so the summation order is transcription and not performance:
 *
 *   - A MAX is order-free and every max here is a strict `>`, which is also how NaN is kept out:
 *     libref's `if (v > mx) mx = v` over -infinity never adopts a NaN, and the butterfly merges with
 *     the same strict comparison, so a NaN in a row of logits cannot become the max anywhere.
 *   - A SUM of f32 is not order-free. libref's `sum` (the pick, the narrowing stages) runs
 *     over the candidates in INDEX order, and the CDF scan below it accumulates in index order too.
 *     So the exps are computed in parallel -- each is a separate value, order-free -- and the
 *     accumulation is serial on one lane, in index order, exactly as written. That is the whole
 *     design of the candidate kernels: parallel values, serial sums.
 *   - The draw is specified in the ABI, not left to a library: splitmix64 over
 *     `seed * 0x9E3779B97F4A7C15 + pos`, top 24 bits over 2^24. Integer arithmetic, so it is the
 *     same number here as on the host. Independent draws at one position are independent streams:
 *     the pick XORs the seed with RAD_SAMPLE_KEY_PICK (0), XTC's firing coin with RAD_SAMPLE_KEY_XTC.
 *   - `ref_exp`/`ref_log` are `(float)std::exp((double)x)` and `(float)std::log((double)x)`, and
 *     they are transcribed as double calls rounded once to f32. There is no fp8-style hardware
 *     shortcut to take: an f32 exp2 (`v_exp_f32`) is a couple of ulps away, and a `u < cum`
 *     comparison sitting on an ulp boundary changes the drawn token, which is a fixture failure even
 *     when the distribution is right. Consumer RDNA3 fp64 is 1:64, so this is the slow path and the
 *     candidate set is where it shows: it costs time, not agreement. The device's
 *     `(float)exp((double)x)` and `(float)log((double)x)` are bit-identical to glibc's over
 *     a million points in [-100, 100] (0 of 2^20 differ), while `expf` differs from the host's
 *     `expf` in 5.7% of them by 1 ulp -- the double call is not only the law, it is the only one
 *     of the two that reproduces the recording.
 *
 * This is where the law is affordable, and it is why this file and the attention unit differ. Here
 * the widest softmax is the candidate set: 1024 exponentials a row, 2 per lane, and the whole op is
 * half a millisecond a step at the band. Attention at the prefill band is 17 M exponentials a layer
 * on the same 1:64 fp64, and there the law cannot be paid, so that unit ships a 0.5-ulp polynomial
 * and says so. A sampler that draws a different token than the recording is wrong at any speed, so
 * this unit keeps the double call and the switch stays off.
 *
 * ============================== the top-k, and why it is one pass ==============================
 *
 * libref selects with `topk_next`: R scans of the row, each taking the largest
 * value strictly below the last pick, ties to the LOWER column. Read as a whole that is a sort by
 * (-value, +column), and the rule `v == *pv && i <= *pi` is exactly "the next one within a run of
 * equal values" -- so one pass that keeps the best R by that order produces the same R pairs, in the
 * same order, with the same tie-breaks. At the band's row (N=130560, R=16, bf16, M=1792) the
 * reference scans the row sixteen times: 7.0 GB of traffic. This file scans it once: 468 MB.
 *
 * At the band (M=1792, N=130560, R=16) the one pass is 22.0 ms bf16, 22.4 ms f32, i.e. 21 GB/s and
 * 42 GB/s: the same time for half the bytes, so the pass is not bound by the plane, it is bound by
 * the accepted inserts (~55 sifts a lane at 510 elements a lane, POOL steps each). The traffic is 15x
 * lower than the reference's and the headroom is in the insert, not in the load; a lane that filtered
 * against a block-wide threshold instead of its own tail would take ~5 sifts a lane and land near the
 * 2 ms the bandwidth floor allows. The number is stated so it is not read as the memory system's
 * answer.
 *
 * One pass needs a per-lane list, and a list costs comparisons, so it is 64 deep and the merge keeps
 * the best 64 of two lists: a lane's test against its 64th best is ONE comparison for almost every
 * element (only ~12% of a row's elements clear it at 510 elements a lane), and the deep comparisons
 * happen only for the elements that enter. Sixty-four covers R=16, the selftest's 64-wide candidate
 * set, and every top_k a request asks for in practice. Deeper is not free and is not silent:
 * `r3d_sample_topk` and `r3d_row_topk` take ceil(k/64) passes when the row asks for more, and each
 * pass is one more read of the row. That is the reference's own chain, so the answer is the same one
 * at any k; only the time is the reference's.
 *
 * The chain's QUIRK is transcribed too, because kbench compares the plane. `topk_next` seeds its
 * scan with `best_i < 0 ||`, so the first element it examines becomes the best even when it is a
 * NaN, and after that `v > best_v` is false for everything: a row whose FIRST logit is NaN yields R
 * identical (NaN, 0) slots, not a hole. To reproduce that the 64-deep pool is re-scanned by ONE lane
 * in ascending column order with the literal chain -- the pool provably contains every element the
 * chain can ever pick (a pick is in the global top-64, so it is in its own lane's top-64), and the
 * chain's comparisons are only exact when the scan order is the row's.
 *
 * ============================== shape ==============================
 *
 * One block a row everywhere: at the band a row of 130560 f32 is 510 KiB and M is 2048, so the whole
 * plane is 1 GiB and these are the memory-bound kernels of the library -- argmax and top-k each read
 * 1 GiB, temp reads and writes 2 GiB. A row of the candidate set is 1024 f32 and the work is the
 * transcendentals, so the same shape costs nothing and buys the serial order for free. The candidate
 * stages keep their probabilities in LDS (2 * 1024 floats), which is what makes typical's O(n^2)
 * distance scan affordable.
 *
 * Single rank. `vocab_off` is read where libref reads it (`row_topk`'s reported ids, `sample_topk`'s
 * `pairs` plane) and the pair planes are written as the shape hooks describe them, because kbench
 * fills and compares them. There is no all-gather here and no merge kernel: `sample_merge_topk` is
 * declined by libref's own shape hook and rank 0 has nothing to merge.
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

/* The candidate pool's depth, and the widest candidate set the narrowing stages carry in LDS.
 * 1024 candidates is the engine's declared buffer width; two f32 arrays of it is 8 KiB of 64 KiB. */
constexpr int kPool = 64;
constexpr int kCandMax = 1024;

/* The double call and the single rounding to f32 ARE the law: libref's oracle
 * is glibc's double exp and log, and the f32 result of a double-accurate implementation agrees with
 * it for every input but a double-rounding tie. */
/* The two transcendentals, and the A/B switch. The law is libref's `(float)std::exp((double)x)`,
 * and on this card the device's double `exp`/`log` are BIT-IDENTICAL to glibc's (0 of 2^20 points
 * in [-100, 100] differ, both functions), so the law is reproducible here and the draw is the
 * recording's. `expf` is not: it moves 5.7% of inputs by 1 ulp relative to the host's, which is
 * under a bit-exact fixture. -DR3D_EXP_F32 swaps both for the f32 hardware path, so the cost of the
 * law can be measured in kbench; nothing in the shipped build uses it, and the flips it causes are
 * reported with it. */
#ifdef R3D_EXP_F32
__device__ __forceinline__ float ref_exp(float x) { return expf(x); }
__device__ __forceinline__ float ref_log(float x) { return logf(x); }
#else
__device__ __forceinline__ float ref_exp(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float ref_log(float x) { return (float)log((double)x); }
#endif

/* abi/rad_sample.h's draw, verbatim: splitmix64 over (seed * golden + pos), top 24 bits as a float
 * in [0, 1) -- exactly f32's mantissa, so no double anywhere. */
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

/* The params row, as the kernels see it. 96 bytes, a multiple of 8, so row m is 8-byte aligned in a
 * plane of words and the u64 seed and pos load as one instruction. */
struct Params {
    const RadSampleParams* p;
    __device__ __forceinline__ const RadSampleParams& operator[](uint32_t m) const {
        return p[m];
    }
};

/* A block-wide maximum with a strict `>`, which is the comparison libref writes and the reason a NaN
 * is never the max: a wave butterfly, then the wave partials. `lds` holds kWaves floats. */
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
 * The law: the largest logit, ties to the LOWER column, because greedy and top-1 have to
 * agree. The scan's rule is `bi < 0 || v > best`: strict, so the first of a run wins, and a NaN is
 * never adopted -- except as the row's very first element, which the seed of the scan adopts. The
 * parallel form is the same total order: take the right candidate only when it is strictly better
 * (larger, or equal with a lower column), which is associative, so the butterfly and the serial scan
 * agree, and a NaN in `best` is never displaced by anything.
 */
struct Best { float v; uint32_t i; };

/* The scan's law, made associative. `best_take(a, b)` keeps `a` unless `b` is strictly better --
 * larger, or equal with a lower column -- and `b.i == kNone` is "nothing examined yet". The
 * reference's serial scan seeds with `bi < 0 ||` and then moves only on a strict `>`, which is the
 * same total order, so a butterfly of this and a scan of that agree on every input. */
__device__ __forceinline__ Best best_take(Best a, Best b) {
    const bool better = (b.v > a.v) || (b.v == a.v && b.i < a.i);
    return (b.i != 0xffffffffu && better) ? b : a;
}

/* A max with a strict `>` over the block, NaN dropped: `fmaxf` is the drop, and libref's
 * `if (v > mx) mx = v` from -infinity never adopts a NaN either. */
__device__ __forceinline__ float block_max_strict(float v, float* lds) {
    return block_max(v != v ? -INFINITY : v, lds);
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
        /* topk_next's seed, and nothing else: the first element the reference examines becomes its
         * best even when it is a NaN, and no finite value can displace a NaN afterwards, so a row
         * that opens with a NaN reports column 0. Every other NaN the scan walks past. */
        const float x0 = ldg_f32(lr);
        tok[row] = (int32_t)(x0 != x0 ? 0u : t.i);
    }
}

/* ------------------------------------------------------------------ sample_temp
 *
 * The law: `inv = 1/t` computed once per row, then every logit multiplied -- by the
 * reciprocal, not divided, and only when the row is sampled at all: `!(t > 0) || t == 1` leaves the
 * row exactly as it was, bit for bit, which is what makes temp 1.0 and temp 0.0 (greedy) no-ops the
 * kernel has to keep. The write is in place: logits are an INOUT operand.
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
 *   params [M]          24 words a row; argmax does not read it, and libref's shape hook describes
 *                       it anyway, so the operand is there and unchecked, as libref leaves it.
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
 * One lane's share of a row, kept as the 64 best by (value descending, column ascending) -- the
 * order `topk_next` walks. Two rules make it one pass and make it exact:
 *
 *   - the order is a STRICT total order, so a merge of two such lists is a merge of two sorted
 *     lists and keeps the first 64 of the union, which is what the reference's R scans converge on;
 *   - the list lives in registers, so insertion is a 64-step sift with STATIC indices (compare, and
 *     swap when the newcomer is better) and never a dynamic index, which would spill the array to
 *     private memory. The sift costs 64 compares per INSERTED element, and elements are inserted
 *     only after a single compare against the lane's 64th best -- about one element in eight at the
 *     band's 510 columns a lane -- so the amortised cost is a few comparisons a column.
 *
 * Slots the row never filled hold (-INFINITY, 0xffffffff): lower than any real value, and higher
 * than any real column, so a real -infinity logit (a masked one) beats a pad and the extraction
 * stops at the first pad.
 */
namespace {

constexpr uint32_t kPadCol = 0xffffffffu;

/* The pool's order, strict: larger value wins, equal value goes to the lower column. */
__device__ __forceinline__ bool pool_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

/* Insert (v, i) into a lane's register list. POOL is the list depth, which is the width of a pass:
 * the sift is POOL steps and the merges are POOL-wide. The band asks for 16 (a router's top_k) and
 * a candidate set for 64; carrying 64 through a router row costs four times the sift and returns
 * the same 16 pairs.
 *
 * `pv`/`pi` carry the pick chain: an element at or above the last pick has been picked already, and
 * within a run of equal values only columns after the last pick are still candidates --
 * `topk_next`'s two skips, in its order. */
template <int POOL>
__device__ __forceinline__ void pool_insert(float v, uint32_t i, float pv, uint32_t pi,
                                            float* lv, uint32_t* li) {
    if (v != v) return;                          /* the scan walks past a NaN (its first element is
                                                    handled by the caller, as the reference's is) */
    if (v > pv) return;                          /* picked on an earlier pass */
    if (v == pv && i <= pi) return;              /* the last pick, or one before it */
    if (!pool_better(v, i, lv[POOL - 1], li[POOL - 1])) return;   /* the fast path: one compare */

    float cv = v;
    uint32_t ci = i;
#pragma unroll
    for (int j = 0; j < POOL; ++j) {            /* the sift: every index a compile-time constant */
        if (pool_better(cv, ci, lv[j], li[j])) {
            const float tv = lv[j]; const uint32_t ti = li[j];
            lv[j] = cv; li[j] = ci; cv = tv; ci = ti;
        }
    }
}

/* The best 64 of the union of two lists, in the pool's order.
 *
 * A[j] and B[j] are descending, so C[j] = the better of A[j] and B[kPool-1-j] is a bitonic sequence
 * that already contains the top 64 of the union, and a bitonic sort of 64 in registers is six
 * passes of 32 compare-exchanges with static strides -- no dynamic index, no spill, 448 compares.
 * Both lanes of a shuffle pair compute the same result, which is what makes a butterfly of this
 * operation a wave reduction. */
template <int POOL>
__device__ __forceinline__ void pool_merge(const float* av, const uint32_t* ai,
                                           const float* bv, const uint32_t* bi,
                                           float* ov, uint32_t* oi) {
#define R3D_POOL_GET(X, J)                                                                          \
    const float x##v = r3d_xor_lane<J>(X[0]); (void)x##v
#pragma unroll
    for (int j = 0; j < POOL; ++j) {
        const int r = POOL - 1 - j;
        if (pool_better(av[j], ai[j], bv[r], bi[r])) { ov[j] = av[j]; oi[j] = ai[j]; }
        else                                        { ov[j] = bv[r]; oi[j] = bi[r]; }
    }
#undef R3D_POOL_GET
#pragma unroll
    for (int k = POOL; k > 1; k >>= 1) {
#pragma unroll
        for (int h = k / 2; h > 0; h >>= 1) {
#pragma unroll
            for (int j = 0; j < POOL; ++j) {
                const int o = j ^ h;
                if (o <= j) continue;
                /* Descending everywhere: the whole array is one descending run, so the direction of
                 * every compare-exchange is "the larger value to the lower index". */
                if (pool_better(ov[o], oi[o], ov[j], oi[j])) {
                    const float tv = ov[j]; const uint32_t ti = oi[j];
                    ov[j] = ov[o]; oi[j] = oi[o]; ov[o] = tv; oi[o] = ti;
                }
            }
        }
    }
}

/* One lane's pass over its columns of a row. STEP is 1 (scalar) or 4 (a vector load a step); LD
 * reads one column. */
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
            for (uint32_t c = nw * 4 + tid; c < n; c += kThreads)   /* the row's odd tail, once */
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
 * LDS, and the block's 64 best in LDS at [kWaves*kPool, ...). Returns how many are real (pads
 * stop it). */
/* The block's 64 best are left in LDS at [kWaves*kPool, ...): the extraction that follows walks
 * them with a runtime index, and a runtime index into a register array is a spill. */
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
    if (threadIdx.x == 0) {                        /* the eight lists, merged with cursors */
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

/* The pool, in LDS: the eight wave lists, then the block's. */
}  /* namespace */

/* ================================================================== row_topk
 *
 * The law: exact per-row top-R, descending, ties to the lower column, with `vocab_off`
 * added to every reported id and the optional [M, R, 2] pair plane an all-gather moves (an int32 id
 * in the first f32 slot, the value in the second, id -1 for a slot the row could not fill).
 *
 * The reference calls `topk_next` R times, a full scan each time; the pool above produces the same
 * R pairs from one scan, because the pool's order IS the chain's order. A row that wants more than
 * 64 costs ceil(R/64) scans and carries the chain's state across them, so the answer is the same at
 * any R and only the time is the reference's.
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

    /* topk_next's seed of the scan: the first element it looks at becomes the best even as a NaN,
     * and nothing displaces a NaN, so a row that opens with one reports (0, NaN) in every slot.
     * Stated once, here, rather than buried in the pool's comparisons. */
    if (threadIdx.x == 0)
        st[0] = XDT == 0 ? bf16_to_f32((uint16_t)ldg_u16((const uint16_t*)x + (size_t)row * x_ld))
                         : ldg_f32((const float*)x + (size_t)row * x_ld);
    __syncthreads();
    if (st[0] != st[0] && threadIdx.x == 0) {
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
            /* Only the first `nout` entries of the block list were written this pass; what sits past
             * them is last pass's list, so the count, not the pad sentinel, bounds the copy. */
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
#ifdef R3D_POOL_TRACE
        if (threadIdx.x == 0)
            printf("    [pool] n=%u done=%u nout=%u take=%u pv=%g pi=%u -> pv=%g pi=%u done=%u\n",
                   n, done, nout, (uint32_t)0, pv, pi, st[1], s2[0], s2[1]);
#endif
        pv = st[1]; pi = s2[0]; done = s2[1];
    }
}

/* One row's top_k: the request's, clamped to the buffer, whole width when absent, none when the row
 * is greedy (its token is sample_argmax's, and a stage that wrote it candidates would be a bug the
 * pick cannot see). */
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
 * The law: the row's top_k largest logits into the fixed-width candidate set, descending,
 * ties to the lower token id, with `cand_idx` -1 and `cand_val` -infinity for a hole. Two things
 * here are not in row_topk: the width is the BUFFER's (a declare-time bound), the row's top_k only
 * narrows within it -- top_k 40 in a 1024-wide buffer is 40 candidates and 984 holes -- and a greedy
 * row takes none, because its token is sample_argmax's and the pick must not overwrite it.
 *
 * `cand_val` carries LOGITS, not probabilities: every narrowing stage takes its own softmax over the
 * set, which is what lets the stages compose in any order.
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

    /* The whole buffer is written every step: a hole is a value the later stages test for, and a
     * stale candidate from the previous step would be drawn. */
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
            /* Only the first `nout` entries of the block list were written this pass; what sits past
             * them is last pass's list, so the count, not the pad sentinel, bounds the copy. */
            const uint32_t room = (k - done < POOL) ? k - done : (uint32_t)POOL;
            const uint32_t take = nout < room ? nout : room;
            uint32_t j = 0;
            for (; j < take; ++j) {
                const uint32_t col = lds_i[kBlk + j];
                if (col == kPadCol) break;
                const float v = lds_v[kBlk + j];
                const int32_t id = (int32_t)(col + (uint32_t)vocab_off);
                ir[done + j] = (int32_t)col;      /* cand_idx is LOCAL: it indexes this row of logits */
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
 *   ci      [M][n_cand]  i32, LOCAL columns, -1 for a hole
 *   cv      [M][n_cand]  f32, the LOGIT, -infinity with the hole
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
 * One block a row of the candidate set, the row in LDS (five arrays of n_cand f32 at the band's
 * 1024 is 20 KiB of the 64 KiB), and the division of labour the arithmetic asks for:
 *
 *   - the max is a butterfly of strict `>`: libref's `if (v > mx) mx = v` from -infinity, so a
 *     candidate at -infinity (a hole, or a grammar-masked token at the set's tail) is never the max
 *     and a NaN never is;
 *   - exp(v - mx) is one value per candidate, so the wave computes them all, in parallel, into LDS;
 *   - every SUM that follows is libref's loop, in index order, on lane 0. The normaliser, the top-p
 *     prefix, the entropy and the inverse-CDF scan are all order-sensitive f32 accumulations, and
 *     the draw is decided by the last of them. A parallel prefix sum would round differently, and a
 *     draw that rounds differently is a different token: this is the law, so the loop stays serial.
 *     It is 1024 adds, half a microsecond; the transcendentals it waits on are the cost, and they
 *     are already out of the loop.
 *
 * A greedy row is left alone by every stage, as the reference leaves it, which is the same rule
 * that keeps its token the argmax's.
 */
namespace {

enum { ST_TOPP = 0, ST_MINP = 1, ST_TYPICAL = 2, ST_XTC = 3, ST_PICK = 4 };

__device__ __forceinline__ void cand_load(uint32_t n_cand, const int32_t* __restrict__ ir,
                                          const float* __restrict__ vr, int32_t* __restrict__ ci,
                                          float* __restrict__ cv) {
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
        ci[i] = ir[i];
        cv[i] = vr[i];
    }
}

__device__ __forceinline__ void cand_write(uint32_t n_cand, int32_t* __restrict__ ir,
                                           float* __restrict__ vr, const int32_t* __restrict__ ci,
                                           const float* __restrict__ cv) {
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
        ir[i] = ci[i];
        vr[i] = cv[i];
    }
}

/* exp(v[i] - mx) for every live candidate, in parallel. A dropped candidate gets 0, which is what
 * the reference's loops contribute for it (they skip it, and a 0 is inert in every sum below). */
__device__ __forceinline__ void cand_exp(uint32_t n_cand, const int32_t* __restrict__ ci,
                                         const float* __restrict__ cv, float mx,
                                         float* __restrict__ ev) {
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads)
        ev[i] = (ci[i] >= 0) ? ref_exp(cv[i] - mx) : 0.0f;
}

__device__ __forceinline__ float lane0_sum(uint32_t n_cand, const float* __restrict__ ev,
                                           uint32_t* out) {
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < n_cand; ++i) sum += ev[i];
        out[0] = (uint32_t)(uintptr_t)0;
        ((float*)out)[0] = sum;
    }
    __syncthreads();
    return ((float*)out)[0];
}

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

    /* mx over live candidates with a strict `>` from -infinity: a hole, a masked tail at -infinity
     * and a NaN are all excluded, exactly as the reference's loop excludes them. */
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
            /* Count and last are a sum and a maximum: both order-free, which is why this stage needs
             * no serial loop. Candidates above the threshold are re-tested below, from the same
             * values in `s_ev`. */
            const float n_above = block_sum<kThreads>(cnt, red);
            const float hi = block_max(last, red);
            if (n_above < 2.0f) return;           /* needs two to exclude all but the least likely */
            const int32_t keep = (int32_t)hi;
            for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
                if (s_ci[i] < 0 || (int32_t)i == keep) continue;
                if (s_ev[i] * inv >= th) { s_ci[i] = -1; s_cv[i] = -INFINITY; }
            }
            __syncthreads();
        } else if (STAGE == ST_TYPICAL) {         /* typical: the pick must not fall through here,
                                                   * it needs s_ev to still hold the exponentials */
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
             * p and does not depend on the candidate order. */
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
 *   cv      [M][n_cand] f32, INOUT: the LOGIT, -infinity with a drop
 *   params  [M]         24 words a row -- the stage's own scalar, the seed and the pos
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
 * Nine shims, one per op. What a row's dtype is, the tensor says; what a row's geometry is, the
 * operand says; `M` is the band's ceiling and never the row count, which is numel(x)/n_vocab. The
 * params plane is checked the way libref checks it -- a 32-bit dtype and enough bytes for M rows --
 * and contiguous, because the kernels index it flat, as libref does.
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

/* param_rows: the params operand is a byte blob of [M, 24] words. The dtype test
 * is not decoration -- a 16-bit view of the plane passes a byte-count check and then misreads every
 * row, so a stage would sample with the top half of one row's temp and the bottom of the next. */
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
    /* The kernel reads no scalar from the params plane, so a missing one is not a refusal: argmax is
     * the row every greedy request needs, and libref describes the operand without reading it. */
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
    /* The width is the buffer's, the declare-time bound: the row's top_k narrows inside it. */
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
    /* The step's rows are the operand's row count. The band key `M` is the plan's ceiling, never a
     * source of row count: the plane decides the rows, and `M` is checked only as the upper bound
     * the plan promised not to exceed. Preferring the key over the plane would refuse a step that
     * fits the band. */
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
