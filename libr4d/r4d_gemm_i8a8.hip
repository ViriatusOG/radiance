// The int8 GEMMs over a fragment-order weight with a bf16 scale per row and 128 (or 256) of K, or
// one per 8 / 32 rows. Both operands are int8 codes; v_wmma_i32_16x16x16_iu8 accumulates them in
// int32. Three tiles share one operand mapping:
//
//   narrow   M <= 64 (decode and speculative verify). A wave's 16 weight rows against up to four
//            16-token tiles, split-K over the caller's scratch (gemm_i8a8_nt_m16).
//   K-par    M <= 16, a weight of few rows (gemm_i8_kpar_kernel, gemm_nt_q_multi).
//   prefill  any M. 128x128 tile, eight waves, each 2 weight tiles x 4 activation tiles; the
//            weight from registers, the activation from LDS (gemm_i8a8_tiled and its row forms).
//
// Properties of the K loop:
//   1. The bf16 scale decodes as `__uint_as_float(s << 16)` (r4d_fp8_wmma.h).
//   2. The weight is the WMMA A operand on both tiles, so D is C^T of `C[M,N] = A[M,K] @ W[N,K]^T`
//      and a lane's eight accumulator slots are eight weight rows of one activation row.
//   3. Split-K uses the caller's scratch (nothing on a launch path allocates), sized by
//      m16_scratch(). The summation order is fixed.
//   4. C is bf16, like every GEMM in this library.
//
// Load-bearing: the padded LDS row pitch (r4d_fp8_wmma.h `lds_pitch`), the one-K-tile-ahead
// register prefetch, the two accumulator sets, the staging assignment (`kTPR`) and, on the prefill
// tile, keeping the weight out of LDS. Reasons are at each.
//
// Entry points return a negative code rather than throwing: they are reachable from a kernel
// plugin, and an exception must not cross the .so boundary.

#include "r4d.h"
#include "r4d_keyed.h"
#include "r4d_quant_args.h"
#include "r4d_emit.h"
#include "r4d_chain.h"
#include "r4d_fp8_wmma.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <type_traits>

namespace {

using namespace r4d_fp8;

constexpr uint32_t kWave = 32;

// The scale group along K. The weight's scale grid, the activation quantiser's group
// (r4d_quant_act_fp8) and this fold must agree, so it is one constant, not a build flag.
constexpr uint32_t kBlock = 128;

// Narrow-tile K width: one gfx1201 cacheline, 256 B. `kTPR` is FK/32, so this puts eight threads
// on a row, which makes the weight stream coalesce (64 B would put two and cost a third of the
// bandwidth; 128 B is also full speed; 512 B needs 64 VGPRs of prefetch and does not build).
constexpr uint32_t kFKNarrow = 256;

// ---- the prefill tile: its geometry and body (shared with the IQR tile) ------------------------
#include "r4d_gemm_8a8_prefill.h"

// Blocks the split aims for. A fixed tuning default, not derived from the CU count: `ksplit` sets
// the order K partials are summed in, so deriving it would make output depend on the card.
//
// `gx * mw` is N/16, so the rule depends on N alone. A target of 512 leaves the attention
// projections (4096-5120 rows) at ~60% of their split speed; 256 leaves the 2048-3072-row
// projections 3-12% short; above 8192 rows splitting costs. 128 holds across the whole M band: the
// split's cost (an M*N f32 partial plane) grows with M while its benefit does not.
//
// The scratch query is asked once at declare with the band's upper bound and is safe only because
// `ksplit * M * N` falls with M, so any rule that splits more at small M could exceed the bound's
// answer and the launch would silently fall back to ksplit 1. A lower target would also
// repartition K and change the emitted text, so the target is kept.
constexpr uint32_t kSplitTarget = 128u;
// Split cap when the output is thin (`M*N < 8192`), where the reduce could be thinner than the GEMM
// the split widened. The reduce keeps eight loads in flight, which lets the cap sit this high.
constexpr uint32_t kSplitThinCap = 64u;

// ---- fragment-order weight ---------------------------------------------------------------------
//
// A 16x16 fragment from a row-major [N][K] plane is 8 bytes every `w_ld`, so the weight would have
// to go global -> LDS -> fragment (a staging pass, a barrier per K tile, 36 KiB of LDS). In
// fragment order (r4d_fp8_frag.h, applied by the layout hook at convert) the bytes are contiguous
// and none of that is needed. The order is a container property, so the prefill tile serves it
// too. 8 bytes per lane (a 256 B request) loses to staging; 32 bytes per lane (1 KiB per wave)
// wins everywhere, by ~13% on gate_up's decode shape.
//
// ---- the body ----------------------------------------------------------------------------------
//
// int8 x int8 -> int32 with no conversion in the loop: both operands are staged as raw bytes.
// Both carry a scale per `group` of K, so the scales are applied once per group: `raw` accumulates
// the group unscaled and then folds into `out` as f32 (no per-element arithmetic in the inner
// loop).
//
// ---- the weight is the A operand ---------------------------------------------------------------
//
// On both tiles a lane's A fragment bytes are its 32 bytes of the fragment-order plane, and the
// activation is B. The accumulator is column-distributed (a lane owns one B row and eight A rows),
// so with the weight on A a lane's eight accumulator slots share one activation scale and hold
// eight weight rows, whose scales are one 16-byte load from the tile-major plane. D is [N][M]: the
// narrow tile's epilogue transposes on the way out (sixteen lanes write addresses `N` apart),
// which at M <= 64 is negligible against the weight read; the prefill tile stores C row by row,
// eight consecutive weight rows of one activation row as one 16-byte store.
//
// The narrow path's weight loads carry no nontemporal hint: the activation every block re-reads
// was just written by the previous kernel and is resident, so protecting it does not move a step.

/* Waves per SIMD for each narrow tile shape (R4D_WAVES): pins the register budget each shape was
 * tuned at (the NT accumulators and K-step unroll are sized against it), so a compiler release
 * cannot change occupancy. */
constexpr uint32_t narrow_waves(bool full, uint32_t nt, uint32_t mw, uint32_t fk) {
    if (fk == 128u)
        return nt == 1u ? 16u : nt == 2u ? (full ? 10u : 12u) : nt == 3u ? (full ? 9u : 10u)
             : full ? 7u : 9u;
    /* 10, not 12: the fold's LDS-staged scales need a few more registers than the 120 that 12
     * waves leave, and spill there. gfx12 allocates registers 24 at a time, so 11 waves leave 120
     * as well and 10 is the next step (144). No decode shape puts more than 8 of these waves on a
     * SIMD, so the lower pin costs no occupancy. */
    if (nt == 1u) return 10u;
    if (nt == 2u) return full ? 9u : mw == 4u ? 9u : 10u;
    if (nt == 3u) return full ? (mw == 16u ? 8u : 7u) : (mw == 4u ? 8u : 9u);
    return full ? (mw == 4u ? 5u : 6u) : (mw == 4u ? 7u : 8u);
}

/* The narrow tile: MW waves a block, a wave one 16-row weight tile (A, fragments straight from the
 * plane into registers) against NT 16-token tiles (B, staged through LDS). The weight scale is one
 * bf16 per row per SB of K, tile-major [N/16][K/SB][16] (r4d_fp8_layout.cpp).
 *
 * The body is a function of its own with two kernels around it (gemm_i8a8_narrow_kernel and its
 * chained form r4dchain_gemm_i8a8_narrow, r4d_chain.h): blocks return early here (an empty K slice,
 * the finisher's losers), and every block of a chained launch must still count itself in. */
template <bool Full, uint32_t NT, uint32_t MW, uint32_t FK, uint32_t SB = kBlock>
__device__ __forceinline__ void gemm_i8a8_narrow_body(
    const uint8_t* __restrict__ Ap, const void* __restrict__ Sa,
    const uint8_t* __restrict__ Bp, const void* __restrict__ Sb,
    void* __restrict__ Y, void* __restrict__ C, unsigned int* __restrict__ Ctr,
    uint32_t rows, uint32_t cols, uint32_t nb,
    uint32_t a_ld, uint32_t b_ld, uint32_t sa_ld, uint32_t sb_ld, uint32_t ck) {
    constexpr uint32_t FN = NT * 16;              // B rows a block
    constexpr uint32_t FM = MW * 16;              // A rows a block

    // One buffer: two (74 KiB) would be over the 64 KiB limit, and decode waits on HBM anyway, so
    // the two-barrier structure is kept by the barrier at the loop foot. No LDS for A: a lane loads
    // its fragment bytes straight from global, so the tile needs 4 KiB of LDS instead of 38.
    __shared__ uint8_t s_b[FN * lds_pitch<FK>()];

    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    // Split-K. `ck` is a multiple of both 128 (the scale block) and FK, and the launcher derives
    // ksplit from it, so every slice has work and starts on a scale boundary.
    const uint32_t kbeg = ck ? blockIdx.z * ck : 0u;
    const uint32_t kend = ck ? min(cols, kbeg + ck) : cols;
    if (kbeg >= kend) return;

    const uint32_t m0 = blockIdx.x * FM;
    const uint32_t n0 = blockIdx.y * FN;
    // Wave grid MW(M) x 1(N): a wave owns 16 A rows and all FN B rows.
    const uint32_t wm = wave * 16u, wn = 0;
    const uint32_t wrow0 = m0 + wm;
    /* SB: the K extent of one scale on both sides. 128, or 256 at IQ8R's group
     * (R4D_I8_GROUP_WIDE), whose int32 sums then run over two rotation blocks before one fold
     * (|sum| <= 256 * 127 * 127 < 2^24, exact in the f32 conversion). The staged tile must not
     * be wider than the group's two halves meet in: FK 256 with SB 256 folds once a tile. */
    static_assert(SB == kBlock || (SB == 256u && FK == 256u),
                  "a 256-wide scale group is the tile's at the 256-byte K step");

    f8v raw[NT], out[NT];
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) { raw[j] = acc_zero(); out[j] = acc_zero(); }

    // ---- staging assignment (a bandwidth decision) ------------------------------------------------
    //
    // A thread stages one 32-byte granule pair, and the threads sharing a row are consecutive, so
    // `kTPR` threads cover a row and one wave load asks for `32 * 32 / kTPR` rows of `kTPR * 32`
    // contiguous bytes. Streaming 5120-byte rows on gfx1201:
    //
    //     threads per row    2         4         8        16        32
    //     GB/s             422       626       633       635       636      (plain stream: 638)
    //
    // Two threads per row leaves a third of the bandwidth unused; four recover it. kTPR = FK/32 is 8
    // on the 256-byte K tile (fetch-bound). A pass covers blockDim/kTPR rows.
    constexpr uint32_t kSeg = 32;                       // bytes a thread stages, contiguous
    constexpr uint32_t kTPR = FK / kSeg;                // threads covering one row
    constexpr uint32_t kRPP = (MW * kWave) / kTPR;      // rows one pass covers
    // Fragment-order supertiles per staged K tile: one 32-byte load per lane, four fragments each
    // (see r4d_fp8_frag.h).
    constexpr uint32_t kAF  = FK / r4d_fp8::kSuperK;
    constexpr uint32_t kBP  = (FN + kRPP - 1) / kRPP;   // passes over the B tile
    static_assert(kTPR >= 1 && kRPP >= 1, "FK must be at least 32 B and no wider than the block");

    const uint32_t sn = tid / kTPR, sk = (tid % kTPR) * kSeg;

    const uint8_t* bsrc = Bp + (size_t)(n0 + sn) * b_ld + sk;
    // A as fragments: a wave owns one 16-row weight tile, so its A operand for 64 k is this lane's
    // 32 bytes. `ksuper` is the plane's tile stride; `arow` is out of range only when N is not a
    // multiple of FM.
    const uint32_t ksuper = cols / r4d_fp8::kSuperK;
    const uint32_t arow   = wrow0 / 16u;
    const bool     alive  = wrow0 < rows;
    const uint8_t* afrag  = Ap + (size_t)lane * r4d_fp8::kLaneBytes;

    // One K tile ahead in registers: the load for tile c0+FK is issued before tile c0's WMMA and
    // consumed after it, hiding memory latency under arithmetic. `sk` is the same column for every
    // pass, so only row existence varies per pass.
    const bool k_live = Full || (kbeg + sk + kSeg <= cols);
    /* `pf` is read straight from registers by the mma loop, so its prefetch goes to `pfn` and the
     * loop foot rotates them; prefetching into `pf` would overwrite the current tile (a bug hidden
     * whenever split-K gives a block only one K iteration). */
    Stage8 pb[kBP];
    Stage8 pf[kAF], pfn[kAF];
#pragma unroll
    for (uint32_t p = 0; p < kBP; ++p) {
        const uint32_t r = sn + p * kRPP;
        pb[p] = stage8_load(bsrc + (size_t)p * kRPP * b_ld + kbeg,
                            k_live && r < FN && (Full || (n0 + r) < nb));
    }
#pragma unroll
    for (uint32_t p = 0; p < kAF; ++p) {
        const uint32_t st = kbeg / r4d_fp8::kSuperK + p;
        pf[p] = stage8_load(
            afrag + (size_t)(arow * ksuper + st) * r4d_fp8::kTileBytes, alive && st < ksuper);
    }

    /* The fold's scales are staged through LDS one K tile ahead, beside the B tile. A fold that
     * loaded its own scales after the next tile's prefetch was issued would wait for that whole
     * prefetch (loads return in order), twice a tile at FK = 256. Instead each thread loads one
     * weight-scale dword and one activation scale of tile c0+FK with its B prefetch (a few each
     * only in the selftest's one- and two-wave pins) and stores them with the B tile at the top of
     * the next pass, so a fold reads LDS. That costs two registers a thread; holding a tile's
     * scales per lane (8 + 2*NT registers) spills at the decode shapes' register pins. The staged
     * values are the ones a direct load reads, so results are bit-identical. A dead wave's
     * scales read as 0; its rows are never stored.
     *
     * A tile holds kFolds = FK / SB scale blocks, and the fold at `ks` uses block ks / SB of its
     * tile. s_ws: each wave's 16 bf16 row scales a block, [MW][kFolds][8] dwords (the tile-major
     * plane). s_as: each B row's f32 scale a block, [kFolds][FN]. */
    constexpr uint32_t kFolds = (FK + SB - 1) / SB;
    constexpr uint32_t kWS = MW * kFolds * 8u;
    constexpr uint32_t kAS = kFolds * FN;
    constexpr uint32_t kWSPer = (kWS + MW * kWave - 1) / (MW * kWave);
    constexpr uint32_t kASPer = (kAS + MW * kWave - 1) / (MW * kWave);
    __shared__ __attribute__((aligned(16))) uint32_t s_ws[kWS];
    __shared__ float s_as[kAS];
    const uint32_t nkb8 = (cols + SB - 1) / SB;
    auto load_ws = [&](uint32_t c, uint32_t i) -> uint32_t {
        if (i >= kWS) return 0u;
        const uint32_t tr = (m0 + (i / (kFolds * 8u)) * 16u) / 16u;
        const uint32_t kb = c / SB + (i / 8u) % kFolds;
        return ((Full || tr * 16u < rows) && kb < nkb8)
            ? ((const uint32_t*)Sa)[((size_t)tr * nkb8 + kb) * 8u + i % 8u] : 0u;
    };
    auto load_as = [&](uint32_t c, uint32_t i) -> float {
        if (i >= kAS) return 0.0f;
        const uint32_t tok = n0 + i % FN, kb = c / SB + i / FN;
        return ((Full || tok < nb) && kb < nkb8) ? ((const float*)Sb)[(size_t)tok * sb_ld + kb]
                                                 : 0.0f;
    };
    uint32_t qws[kWSPer];
    float qas[kASPer];
    auto load_q = [&](uint32_t c) {
#pragma unroll
        for (uint32_t r = 0; r < kWSPer; ++r) qws[r] = load_ws(c, tid + r * MW * kWave);
#pragma unroll
        for (uint32_t r = 0; r < kASPer; ++r) qas[r] = load_as(c, tid + r * MW * kWave);
    };
    load_q(kbeg);
    // The B rows this lane owns within the tile, one per N tile (column `frag_c` of each of its NT
    // accumulators, 16 apart). A per-B-row scale must be looked up for each, not just the first.
    const uint32_t bl0 = wn + frag_c(lane);

    for (uint32_t c0 = kbeg; c0 < kend; c0 += FK) {
#pragma unroll
        for (uint32_t p = 0; p < kBP; ++p) {
            const uint32_t r = sn + p * kRPP;
            if (r < FN) stage8_store<FK>(s_b, r, sk, pb[p]);
        }
#pragma unroll
        for (uint32_t r = 0; r < kWSPer; ++r)
            if (tid + r * MW * kWave < kWS) s_ws[tid + r * MW * kWave] = qws[r];
#pragma unroll
        for (uint32_t r = 0; r < kASPer; ++r)
            if (tid + r * MW * kWave < kAS) s_as[tid + r * MW * kWave] = qas[r];
        __syncthreads();
        const uint32_t c1 = c0 + FK;
        if (c1 < kend) {
            const bool live = Full || (c1 + sk + kSeg <= cols);
#pragma unroll
            for (uint32_t p = 0; p < kBP; ++p) {
                const uint32_t r = sn + p * kRPP;
                if (r < FN)
                    pb[p] = stage8_load(bsrc + (size_t)p * kRPP * b_ld + c1,
                                        live && (Full || (n0 + r) < nb));
            }
#pragma unroll
            for (uint32_t p = 0; p < kAF; ++p) {
                const uint32_t st = c1 / r4d_fp8::kSuperK + p;
                pfn[p] = stage8_load(
                    afrag + (size_t)(arow * ksuper + st) * r4d_fp8::kTileBytes,
                    alive && st < ksuper);
            }
            load_q(c1);
        }

#pragma unroll
        for (uint32_t ks = 0; ks < FK; ks += kK) {
            // A comes from registers: the 32 bytes this lane loaded for the 64-k supertile are its
            // four fragments (+0, +8, +16, +24). `ks` is constant after unrolling, so the selection
            // folds away.
            const i2v af = frag_of(pf[ks / r4d_fp8::kSuperK], (ks % r4d_fp8::kSuperK) / 16u);
            i2v bf[NT];
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) bf[j] = frag<FK>(s_b, wn + j * 16, ks, lane);
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) raw[j] = mma_i8(af, bf[j], raw[j]);

            // End of a scale group: fold both scales in and reset. Inside the ks loop because at
            // FK = 256 one staged tile spans two 128-wide groups.
            //
            // `c0` is a runtime value, so the compiler if-converts this test at every unrolled ks.
            // Gating it at the compile-time-known positions is legal and byte-identical but raises
            // the register live range of the fold; see libr4d/isa/README.md.
            if ((c0 + ks + kK) % SB == 0 || c0 + ks + kK >= kend) {
                /* This tile's scale block for `ks`, staged with the tile. */
                const uint32_t f = ks / SB < kFolds ? ks / SB : kFolds - 1u;
                // A is the int8 weight with a bf16 scale per row. A lane's eight accumulator slots
                // are eight consecutive rows of one 16-row tile, adjacent in the tile-major
                // [N/16][K/SB][16] plane: one 16-byte load. B is the activation.
                float wsr[8];
                {
                    const uint4 w8 =
                        *(const uint4*)&s_ws[(wave * kFolds + f) * 8u + (lane >> 4) * 4u];
                    const uint32_t d[4] = { w8.x, w8.y, w8.z, w8.w };
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t)
                        wsr[t] = bscale((uint16_t)(d[t / 2] >> (16 * (t % 2))));
                }
#pragma unroll
                for (uint32_t j = 0; j < NT; ++j) {
                    const float s = s_as[f * FN + bl0 + j * 16u];
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) {
                        out[j][t] += acc_i32(raw[j][t]) * (wsr[t] * s);
                        raw[j][t] = 0.0f;
                    }
                }
            }
        }
        // Wait until every wave has finished reading before the tile and its scales are
        // overwritten. The prefetched fragments become current.
#pragma unroll
        for (uint32_t p = 0; p < kAF; ++p) pf[p] = pfn[p];
        __syncthreads();
    }

    // Split-K writes an f32 partial plane; without it the store is the final bf16 C.
    float* const part = ck ? (float*)Y + (size_t)blockIdx.z * rows * nb : nullptr;
    uint16_t* const c16 = (uint16_t*)Y;

    const uint32_t c = frag_c(lane);
    /* ---- eight consecutive elements in one store ---------------------------------------------
     *
     * `acc_row(lane, t)` is `(lane>>4)*8 + t`, so with the output index `col * rows + r` a lane's
     * eight slots are eight consecutive addresses in one 16-byte piece; one store instead of eight
     * removes the epilogue's M-proportional cost. Bit-identical. Requires `rows` % 8 == 0 for
     * 16-byte alignment (`r` is already a multiple of 8). */
    const bool vec8 = (rows & 7u) == 0u;
    /* C's row pitch: the output and the split-K partial plane are dense at N. */
    const uint32_t cld = rows;
    /* The split-K and vec8 tests are hoisted out of the store loop as template arguments;
     * otherwise the compiler keeps a compare and branch at every store and assembles the vec8
     * words from register halves. */
    auto store = [&](auto split, auto v8) {
        constexpr bool CK = decltype(split)::value;
        constexpr bool V8 = decltype(v8)::value;
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
            const uint32_t col = n0 + wn + j * 16 + c;
            if (!Full && col >= nb) continue;
            const uint32_t rb = m0 + wm;
            const uint32_t r0 = rb + acc_row(lane, 0);
            f8v ov = out[j];
            if (V8 && (Full || r0 + 8u <= rows)) {
                const size_t o = (size_t)col * cld + r0;
                if constexpr (CK) {
                    __attribute__((aligned(16))) float pv[8];
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) pv[t] = ov[t];
                    *(uint4*)(part + o)     = *(const uint4*)pv;
                    *(uint4*)(part + o + 4) = *(const uint4*)(pv + 4);
                } else {
                    /* Packed in whole registers (bf16x2_rne, f32_to_bf16's bits); a
                     * uint16_t[8] would be assembled from register halves. */
                    *(uint4*)(c16 + o) = make_uint4(bf16x2_rne(ov[0], ov[1]), bf16x2_rne(ov[2], ov[3]),
                                                    bf16x2_rne(ov[4], ov[5]), bf16x2_rne(ov[6], ov[7]));
                }
                continue;
            }
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                const uint32_t r = rb + acc_row(lane, t);
                if (!Full && r >= rows) continue;
                // A held the weight, so [r][col] is [N][M] and C is [M][N].
                const size_t o = (size_t)col * cld + r;
                if constexpr (CK) part[o] = ov[t];
                else              c16[o]  = f32_to_bf16(ov[t]);
            }
        }
    };
    if (vec8) {
        if (ck) store(std::true_type{}, std::true_type{});
        else    store(std::false_type{}, std::true_type{});
    } else {
        if (ck) store(std::true_type{}, std::false_type{});
        else    store(std::false_type{}, std::false_type{});
    }

    // ---- split finisher ---------------------------------------------------------------------------
    //
    // Dependent dispatches cost ~2.8 us each on this part (2.3 in a hipGraph), so a separate reduce
    // launch per split GEMM is expensive. Instead the last z block of a column reduces it:
    // `ctr[blockIdx.x]` counts arrivals, and the block that reaches gridDim.z resets it and sums.
    // The z loop runs in fixed ascending order with no atomic on an accumulator, so the result is
    // bit-identical to the separate reduce and deterministic.
    //
    // The counters are a small allocation keyed on (stream, output) (`split_ctr`), not the caller's
    // scratch, which other ops overwrite. The self-reset keeps them zero, which is sound only while
    // launches sharing a counter block never overlap; see split_ctr.
    if (ck && Ctr) {
        // Every thread's stores must be visible device-wide before the arrival is published: a
        // per-thread fence, then the barrier.
        __threadfence();
        __syncthreads();
        __shared__ unsigned int s_arrived;
        if (tid == 0)
            s_arrived = __hip_atomic_fetch_add(&Ctr[blockIdx.x], 1u, __ATOMIC_ACQ_REL,
                                               __HIP_MEMORY_SCOPE_AGENT);
        __syncthreads();
        if (s_arrived + 1u == gridDim.z) {
            if (tid == 0)
                __hip_atomic_store(&Ctr[blockIdx.x], 0u, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
            __threadfence();                       // acquire the other slices' partials
            // This block owns A rows [m0, m0+FM) and all nb output columns, exactly what the stores
            // above wrote.
            const uint32_t rlo = m0, rhi = min(m0 + FM, rows);
            const float* const w0 = (const float*)Y;
            uint16_t* const cy = (uint16_t*)C;
            /* Consecutive threads take consecutive weight rows, which are adjacent in a
             * partial plane ([nb][rows]): four rows a thread as one 16-byte load, kFinU quads in
             * flight at each z. Walking the token index fastest would put consecutive threads
             * `rows` floats apart, a cache line and a round trip per element. Both branches sum
             * each element from 0.0f in ascending z, so they are bit-identical. */
            const uint32_t nr = rhi - rlo;
            const size_t zs = (size_t)rows * nb;
            if ((rows & 3u) == 0u && (nr & 3u) == 0u && ((uintptr_t)C & 7u) == 0u
                && ((uintptr_t)Y & 15u) == 0u) {
                constexpr uint32_t kFinU = 4;
                const uint32_t nq = nr / 4u, total = nq * nb;
                for (uint32_t b = tid; b < total; b += kFinU * MW * kWave) {
                    size_t o[kFinU];
                    float4 acc[kFinU];
#pragma unroll
                    for (uint32_t u = 0; u < kFinU; ++u) {
                        const uint32_t idx = b + u * MW * kWave;
                        o[u] = idx < total ? (size_t)(idx / nq) * rows + rlo + (idx % nq) * 4u : 0u;
                        acc[u] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                    for (uint32_t z = 0; z < gridDim.z; ++z) {
                        float4 v[kFinU];
#pragma unroll
                        for (uint32_t u = 0; u < kFinU; ++u)
                            v[u] = b + u * MW * kWave < total ? *(const float4*)(w0 + z * zs + o[u])
                                                              : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
                        for (uint32_t u = 0; u < kFinU; ++u) {
                            acc[u].x += v[u].x; acc[u].y += v[u].y;
                            acc[u].z += v[u].z; acc[u].w += v[u].w;
                        }
                    }
#pragma unroll
                    for (uint32_t u = 0; u < kFinU; ++u) {
                        if (b + u * MW * kWave >= total) continue;
                        *(uint2*)(cy + o[u]) = make_uint2(bf16x2_rne(acc[u].x, acc[u].y),
                                                          bf16x2_rne(acc[u].z, acc[u].w));
                    }
                }
            } else {
                for (uint32_t idx = tid; idx < nr * nb; idx += MW * kWave) {
                    const size_t o = (size_t)(idx / nr) * rows + rlo + idx % nr;
                    float acc = 0.0f;
                    for (uint32_t z = 0; z < gridDim.z; ++z) acc += w0[z * zs + o];
                    cy[o] = f32_to_bf16(acc);
                }
            }
        }
    }
}

#define R4D_NARROW_ARGS Ap, Sa, Bp, Sb, Y, C, Ctr, rows, cols, nb, a_ld, b_ld, sa_ld, sb_ld, ck
#define R4D_NARROW_PARAMS                                                                      \
    const uint8_t* __restrict__ Ap, const void* __restrict__ Sa,                               \
    const uint8_t* __restrict__ Bp, const void* __restrict__ Sb,                               \
    void* __restrict__ Y, void* __restrict__ C, unsigned int* __restrict__ Ctr,                \
    uint32_t rows, uint32_t cols, uint32_t nb,                                                 \
    uint32_t a_ld, uint32_t b_ld, uint32_t sa_ld, uint32_t sb_ld, uint32_t ck
template <bool Full, uint32_t NT, uint32_t MW, uint32_t FK, uint32_t SB = kBlock>
__global__ __launch_bounds__(MW * kWave) R4D_WAVES(narrow_waves(Full, NT, MW, FK))
void gemm_i8a8_narrow_kernel(R4D_NARROW_PARAMS) {
    gemm_i8a8_narrow_body<Full, NT, MW, FK, SB>(R4D_NARROW_ARGS);
}
template <bool Full, uint32_t NT, uint32_t MW, uint32_t FK, uint32_t SB = kBlock>
__global__ __launch_bounds__(MW * kWave) R4D_WAVES(narrow_waves(Full, NT, MW, FK))
void r4dchain_gemm_i8a8_narrow(R4D_NARROW_PARAMS) {
    r4d_chain_wait();
    gemm_i8a8_narrow_body<Full, NT, MW, FK, SB>(R4D_NARROW_ARGS);
    r4d_chain_done();
}
#undef R4D_NARROW_PARAMS
#undef R4D_NARROW_ARGS

// ---- the K-parallel int8 tile: M <= 16 ---------------------------------------------------------
//
// The narrow tile walks a wave down all of K, and a weight of few rows fills the card only through
// split-K: partial planes in global scratch and a finisher behind arrival counters. That holds an
// int8 GEMM near 18 us at K = 2560 whatever its rows (256 rows: 18.5 us, where the IQR 16-token
// tile takes 5). Here a block is one 16-row weight tile and its KW waves stride K by scale group
// (wave w takes groups w, w + KW, ..), as the IQR tile's do: the weight's fragments and the
// activation's are read straight from global memory, a group's int32 sums fold once with its
// (row, group) and (token, group) scales, and the waves' partials meet in LDS in ascending wave
// order. No scratch, no counters and no block barrier in the K loop. A block multiplies its weight
// tile against NT token tiles of sixteen; token columns past M read token M - 1 and are not stored. The sum order is this tile's own (per wave, then across waves,
// whatever N and M are), so its output is the product within f32 rounding, not the narrow tile's
// bytes.
#define R4D_I8KPAR_PARAMS                                                                      \
    const uint8_t* __restrict__ Ap, const uint16_t* __restrict__ Sa,                           \
    const int8_t* __restrict__ Bp, const float* __restrict__ Sb, uint16_t* __restrict__ C,     \
    uint32_t rows, uint32_t cols, uint32_t M, uint32_t b_ld, uint32_t sb_ld
/* `rt` is the block's 16-row weight tile: its blockIdx.x, or its place in a slice of the
 * several-weight form (gemm_i8_kpar_multi_kernel). */
template <uint32_t KW, uint32_t SB, uint32_t NT>
__device__ __forceinline__ void gemm_i8_kpar_body(R4D_I8KPAR_PARAMS, uint32_t rt) {
    constexpr uint32_t kUG = SB / kBlock;   /* 128-K units a scale group */
    __shared__ float s_red[KW][NT][8 * kWave];
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1);
    const uint32_t wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const uint32_t ngrp = cols / SB, nunit = cols / kBlock;
    const uint32_t ksuper = cols / r4d_fp8::kSuperK;
    /* Token tile j is tokens 16 j .. 16 j + 15; a column past M reads token M - 1. */
    const uint32_t col = frag_c(lane);
    const int8_t* xrow[NT];
    const float* xsr[NT];
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
        const uint32_t tok = min(j * 16u + col, M - 1u);
        xrow[j] = Bp + (size_t)tok * b_ld + frag_k0(lane);
        xsr[j] = Sb + (size_t)tok * sb_ld;
    }
    const uint8_t* const arow = Ap + (size_t)rt * ksuper * r4d_fp8::kTileBytes +
                                (size_t)lane * r4d_fp8::kLaneBytes;
    /* Unit u of the tile: the weight's two 64-K supertiles, a token tile's eight fragments. */
    auto aload = [&](uint32_t u, Stage8 (&a)[2]) {
#pragma unroll
        for (uint32_t p = 0; p < 2; ++p)
            a[p] = stage8_load(arow + (size_t)(2u * u + p) * r4d_fp8::kTileBytes, true);
    };
    auto bload = [&](uint32_t j, uint32_t u, i2v (&b)[8]) {
#pragma unroll
        for (uint32_t ks = 0; ks < 8; ++ks) {
            const uint2 v = *(const uint2*)(xrow[j] + u * kBlock + ks * kK);
            b[ks][0] = (int)v.x;
            b[ks][1] = (int)v.y;
        }
    };

    f8v out[NT];
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) out[j] = acc_zero();
    /* The weight and the first token tile are asked for a unit ahead of their WMMA steps, each
     * further tile's fragments a tile ahead. */
    Stage8 ac[2], an[2];
    i2v bc[8], bn[8], bq[2][8];
    if (wave < ngrp) {
        aload(wave * kUG, ac);
        bload(0u, wave * kUG, bc);
    }
    for (uint32_t g = wave; g < ngrp; g += KW) {
        f8v raw[NT];   /* int32 sums */
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) raw[j] = acc_zero();
#pragma unroll
        for (uint32_t h = 0; h < kUG; ++h) {
            const uint32_t u = g * kUG + h;
            const uint32_t un = h + 1u < kUG ? u + 1u : (g + KW) * kUG;
            if (un < nunit) {
                aload(un, an);
                bload(0u, un, bn);
            }
#pragma unroll
            for (uint32_t ks = 0; ks < 8; ++ks) bq[0][ks] = bc[ks];
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                if (j + 1u < NT) bload(j + 1u, u, bq[(j + 1u) & 1u]);
#pragma unroll
                for (uint32_t ks = 0; ks < kBlock / kK; ++ks)
                    raw[j] = mma_i8(frag_of(ac[ks / 4u], ks % 4u), bq[j & 1u][ks], raw[j]);
            }
#pragma unroll
            for (uint32_t p = 0; p < 2; ++p) ac[p] = an[p];
#pragma unroll
            for (uint32_t ks = 0; ks < 8; ++ks) bc[ks] = bn[ks];
        }
        /* The lane's eight rows' bf16 scales of this group are consecutive in the tile-major
         * plane: one 16-byte load. */
        const uint4 q4 = *(const uint4*)(Sa + ((size_t)rt * ngrp + g) * 16u + acc_row(lane, 0));
        const uint32_t qd[4] = { q4.x, q4.y, q4.z, q4.w };
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
            const float as = xsr[j][g];
#pragma unroll
            for (uint32_t t = 0; t < 4; ++t) {
                out[j][2 * t] = __builtin_fmaf(acc_i32(raw[j][2 * t]),
                                               __uint_as_float(qd[t] << 16) * as, out[j][2 * t]);
                out[j][2 * t + 1] = __builtin_fmaf(acc_i32(raw[j][2 * t + 1]),
                                                   __uint_as_float(qd[t] & 0xFFFF0000u) * as,
                                                   out[j][2 * t + 1]);
            }
        }
    }
    if constexpr (KW > 1u) {
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) s_red[wave][j][t * kWave + lane] = out[j][t];
        __syncthreads();
        if (wave != 0u) return;
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                float a = 0.0f;
#pragma unroll
                for (uint32_t w = 0; w < KW; ++w) a += s_red[w][j][t * kWave + lane];
                out[j][t] = a;
            }
    }
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
        const uint32_t tok = j * 16u + col;
        if (tok < M)
            *(uint4*)(C + (size_t)tok * rows + rt * 16u + acc_row(lane, 0)) =
                make_uint4(bf16x2_rne(out[j][0], out[j][1]), bf16x2_rne(out[j][2], out[j][3]),
                           bf16x2_rne(out[j][4], out[j][5]), bf16x2_rne(out[j][6], out[j][7]));
    }
}

// The kernel around the body, whose waves past the first return after the fold: CH `r4dchain` is
// the chained form (r4d_chain.h).
template <uint32_t KW, uint32_t SB, uint32_t NT, class CH>
__global__ __launch_bounds__(KW * kWave) R4D_WAVES(6)
void gemm_i8_kpar_kernel(R4D_I8KPAR_PARAMS) {
    if (CH::on) r4d_chain_wait_vec();
    gemm_i8_kpar_body<KW, SB, NT>(Ap, Sa, Bp, Sb, C, rows, cols, M, b_ld, sb_ld, blockIdx.x);
    if (CH::on) r4d_chain_done();
}
#undef R4D_I8KPAR_PARAMS

/* Several int8 linears at one M and K in one launch (gemm_nt_q_multi): slice s owns blocks
 * [end[s-1], end[s]) of the grid, a block its 16-row weight tile, so each slice's bytes are the
 * K-parallel tile's for that linear alone. */
constexpr uint32_t kI8MultiMax = 5;
struct I8KparSlices {
    const uint8_t*  w[kI8MultiMax];
    const uint16_t* ws[kI8MultiMax];
    const int8_t*   a[kI8MultiMax];
    const float*    as[kI8MultiMax];
    uint16_t*       c[kI8MultiMax];
    uint32_t        rows[kI8MultiMax], a_ld[kI8MultiMax], sa_ld[kI8MultiMax], end[kI8MultiMax];
    uint32_t        n;
};

template <uint32_t KW, uint32_t SB, uint32_t NT, class CH>
__global__ __launch_bounds__(KW * kWave) R4D_WAVES(6)
void gemm_i8_kpar_multi_kernel(I8KparSlices S, uint32_t cols, uint32_t M) {
    if (CH::on) r4d_chain_wait_vec();
    const uint32_t b = blockIdx.x;
    uint32_t s = 0, beg = 0;
    while (s + 1u < S.n && b >= S.end[s]) beg = S.end[s++];
    gemm_i8_kpar_body<KW, SB, NT>(S.w[s], S.ws[s], S.a[s], S.as[s], S.c[s], S.rows[s], cols, M,
                                  S.a_ld[s], S.sa_ld[s], b - beg);
    if (CH::on) r4d_chain_done();
}

// Sum the split-K partials into bf16 C, in fixed z order with no atomics. Latency bound, so eight
// loads are issued before their adds; the order of the adds is unchanged.
__global__ void splitk_reduce_kernel(const float* __restrict__ ws, uint16_t* __restrict__ y,
                                     uint32_t n_out, uint32_t ksplit) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t)n_out;
    if (i >= n) return;
    const float* __restrict__ p = ws + i;
    constexpr uint32_t U = 8;
    float acc = 0.0f;
    uint32_t z = 0;
    for (; z + U <= ksplit; z += U) {
        float v[U];
#pragma unroll
        for (uint32_t u = 0; u < U; ++u) v[u] = p[(size_t)(z + u) * n];
#pragma unroll
        for (uint32_t u = 0; u < U; ++u) acc += v[u];
    }
    for (; z < ksplit; ++z) acc += p[(size_t)z * n];
    y[i] = f32_to_bf16(acc);
}

template <int RB = 0, uint32_t SB = kBlock>
__global__ __launch_bounds__(kPWaves * kWave) R4D_WAVES(6)
void gemm_8a8_prefill_kernel(const uint8_t* __restrict__ A, const float* __restrict__ Sa,
                             const uint8_t* __restrict__ W, const uint16_t* __restrict__ Sw,
                             uint16_t* __restrict__ C, uint32_t M, uint32_t K, uint32_t N,
                             uint32_t a_ld, uint32_t sa_ld, uint32_t sw_ld) {
    gemm_8a8_prefill_body<RB, PrefillPlain, false, SB>(A, Sa, W, Sw, C, M, K, N, a_ld, sa_ld,
                                                       sw_ld, PrefillPlain{});
}

/* The same int8 prefill tile over a row-major weight and row-major per-row scales: the container's
 * bytes, so a vision tower on the mapped tier needs no relayout. `w_ld` is the weight's row pitch
 * in bytes (one byte a code). */
__global__ __launch_bounds__(kPWaves * kWave) R4D_WAVES(6)
void gemm_8a8_prefill_row_kernel(const uint8_t* __restrict__ A, const float* __restrict__ Sa,
                                 const uint8_t* __restrict__ W, const uint16_t* __restrict__ Sw,
                                 uint16_t* __restrict__ C, uint32_t M, uint32_t K, uint32_t N,
                                 uint32_t a_ld, uint32_t sa_ld, uint32_t w_ld, uint32_t sw_ld) {
    gemm_8a8_prefill_body<0, PrefillPlain, false, kBlock, true>(
        A, Sa, W, Sw, C, M, K, N, a_ld, sa_ld, sw_ld, PrefillPlain{}, nullptr, nullptr, w_ld);
}

// The narrow tile's K width is kFKNarrow (256 B); 128 B loses at every wave count (see `kTPR`), and
// is used only via narrow_fk for a K that 256 does not divide.

// Double-buffering the narrow tile is ~10% slower even where it fits (MW = 2): the tile is short
// of LDS, not waiting on the barrier.

// The finisher's arrival counters, one per grid.x column (see the finisher). Zeroed once and kept
// zero by the finisher. If allocation fails, the caller falls back to the separate reduce kernel,
// which computes the same bytes.
constexpr uint32_t kMaxCtr = 8192;

/* One counter block per (stream, output), so concurrent launches never share one (sharing would
 * mean a hang or a wrong answer). Launches overlap across streams (`rad_lane`) or on one stream
 * (rad_kernel_concurrent); overlapping launches on one stream write disjoint outputs, and launches
 * sharing an output on one stream are ordered by their dependency. A step has a few dozen distinct
 * outputs at fixed arena addresses, so this is a few dozen 32 KiB blocks, allocated once. A block
 * lives as long as its output's memory (r4d_keyed.h): a recorded pass replays the pointer.
 *
 * The lock is not held while a block is made. Zeroing it waits for the card, and under tensor
 * parallelism the card can be inside an all-reduce whose other half the peer rank has yet to issue;
 * if the peer's first launch of a split GEMM then waited on this lock, neither rank would advance. */
unsigned int* split_ctr(RadStream s, const void* out) {
  return r4d::keyed_block(r4d::Keyed::SplitI8, s, out,
                          (size_t)kMaxCtr * sizeof(unsigned int));
}
// The narrow tile's wave count MW, which is its M tile: the wave grid is MW x 1, so a block covers
// MW*16 weight rows and the grid over N is N/(MW*16). The rule is fitted together with
// kSplitTarget (which shapes split decides whether narrow or wide blocks win) on the shapes the
// engine actually issues (R4D_F8_MWLOG; n_ff is already per rank).
//
//   - M <= 16: MW = 4, best or within a microsecond on every decode shape.
//   - M > 16: MW = 8. The weight is streamed once whatever MW is, but every block stages the whole
//     [M, K] activation, so a call moves N*K + M*K*N/(MW*16) bytes; the second term matches the
//     weight at M = 64 (served from cache, but it is the drop from 586 to 460 GB/s on gate_up).
//     MW = 8 halves it at the same N/16 waves. The crossover is where NT leaves 1.
//   - M > 32 and N >= 2048: MW = 16 (FM = 256, legal because the fold reads the weight scale per
//     wave). Halves the re-read again; narrower N would leave too few blocks for 32 WGPs.
//
// Token tiles NT = ceil(M/16), including 3: rounding 33..48 up to a 64-token tile would spend
// three-eighths of the work at M = 40 on padding.
//
// Neither MW nor NT changes the output bits: `narrow_ksplit` depends on `gx * mw` = N/16 (the rule
// requires `n % 128` so that holds), so every element is one wave's accumulation over the same K
// tiles in the same order.
//
// This part shows occasional correlated +5 us outliers across repetitions; a cell that breaks its
// row's monotonicity needs re-measuring.
uint32_t narrow_mw(uint32_t n, uint32_t m) {
    uint32_t mw = 4u;
    if (m > 16u && (n % 128u) == 0u) mw = 8u;
    if (m > 32u && n >= 2048u && (n % 256u) == 0u) mw = 16u;
    /* Regrouping the same waves into more or fewer blocks (e.g. N = 2560: 40 blocks at MW 4) is
     * slower both ways: the cost is per-block work (activation re-read, scale plane, prologue), not
     * scheduling balance. */
    /* `R4D_F8_MWLOG` prints each distinct (N, M) once, up to 64 (n_ff in a container is already
     * per rank, so the shapes must be read, not derived). */
    static const bool mwlog = [] { const char* v = std::getenv("R4D_F8_MWLOG");
                                   return v && std::strtoul(v, nullptr, 10) != 0; }();
    if (mwlog) {
        static std::mutex lm;
        static uint64_t seen[64];
        static int nseen = 0;
        const uint64_t key = ((uint64_t)n << 32) | m;
        std::lock_guard<std::mutex> lk(lm);
        int i = 0;
        while (i < nseen && seen[i] != key) ++i;
        if (i == nseen && nseen < 64) {
            seen[nseen++] = key;
            std::fprintf(stderr, "[mwlog] N=%u M=%u -> mw=%u\n", n, m, mw);
        }
    }
    return mw;
}

// Raising the split target so N = 2560 (160 waves, 40 blocks) splits does not change its time, so
// the narrow grid is not parallelism-starved at 128. A split shows in grid.z, not grid.x.

/* The narrow tile's K width: 256, or 128 when K is a multiple of 128 but not 256 (e.g. K = 640, a
 * shared expert's down projection), which would otherwise fall to the prefill kernel. At 128 the
 * fold takes the `FK <= kBlock` branch. One function, because the scratch query and the launch
 * must agree. */
inline uint32_t narrow_fk(uint32_t K) { return (K % kFKNarrow == 0u) ? kFKNarrow : 128u; }

/* Test hook behind r4d_gemm_i8a8_pin_split_waves: zero is each caller's own target, anything else
 * overrides every call's. The selftest pins the IQR narrow tile's target around its int8
 * reference, so the two sum their K partials in one order and compare to the byte. */
std::atomic<uint32_t> g_split_waves_force{0u};

// The split, given the narrow grid. Shared by the scratch query and the launch so they agree on the
// number of partial planes.
// `target` is the waves the split aims for: kSplitTarget for the i8a8 tiles; the IQR
// narrow tile (r4d_gemm_iqr.hip), whose waves decode before they multiply, asks for more.
uint32_t narrow_ksplit(uint32_t N, uint32_t K, uint32_t M, uint32_t* ck_out,
                       uint32_t target = kSplitTarget) {
    *ck_out = 0;
    if (const uint32_t f = g_split_waves_force.load(std::memory_order_relaxed)) target = f;
    const uint32_t mw = narrow_mw(N, M);
    const uint32_t fm = mw * 16u;   // the narrow tile is one M tile a wave
    const uint32_t gx = (N + fm - 1) / fm;              // the M grid runs over the weight rows
    if (K < 512u || gx * mw >= target) return 1;
    const uint32_t want = (target + gx * mw - 1) / (gx * mw);
    uint32_t ckw = ((K / want) + kBlock - 1u) / kBlock * kBlock;
    if (ckw < kBlock) ckw = kBlock;
    uint32_t k2 = (K + ckw - 1) / ckw;
    if (k2 > kSplitThinCap && (uint64_t)N * M < 8192) k2 = kSplitThinCap;
    if (k2 < 2) return 1;
    // Round the slice to the staged tile, not the 128-column scale block: a slice that is only a
    // multiple of 128 would run one 256-wide tile past its end into the next slice.

    const uint32_t fk = narrow_fk(K);
    uint32_t ck = ((K + k2 - 1) / k2 + fk - 1u) / fk * fk;
    k2 = (K + ck - 1) / ck;
    if (k2 < 2) return 1;
    *ck_out = ck;
    return k2;
}

}  // namespace

/* The narrow tile's grid and split rules and its finisher's counters, for the IQR narrow tile
 * (r4d_gemm_iqr.hip): its split sums each output as this tile's does. */
uint32_t r4d_8a8_narrow_mw(uint32_t n, uint32_t m) { return narrow_mw(n, m); }
uint32_t r4d_8a8_narrow_ksplit(uint32_t N, uint32_t K, uint32_t M, uint32_t* ck, uint32_t target) {
    return narrow_ksplit(N, K, M, ck, target);
}
unsigned int* r4d_8a8_split_ctr(RadStream s, const void* out) { return split_ctr(s, out); }

// ================================================================== the entry points

extern "C" int r4d_gemm_i8a8_group(void)          { return (int)kBlock; }

extern "C" int r4d_gemm_i8a8_nt_m16_max_m(void)   { return 64; }

/* See r4d.h. */
extern "C" void r4d_gemm_i8a8_pin_split_waves(int waves) {
    g_split_waves_force.store(waves > 0 ? (uint32_t)waves : 0u, std::memory_order_relaxed);
}

// Bytes of f32 scratch one r4d_gemm_i8a8_nt_m16 call needs, or 0 when it will not split. The
// caller may pass more, or none (the launch then runs unsplit: correct but slow, and unreported).
// The bound is max_m() so the query and the band cannot drift apart.
//
// `split_n` (0: N) is the width the split is chosen for. The split depends on N alone, and it
// fixes the order K partials are summed in, so a row slice of a wider linear (N rows of a
// split_n-row weight) passes the whole width to sum each output as the whole linear does: the
// slice's outputs are then the linear's to the bit.
static long m16_scratch(int M, int K, int N, int split_n) {
    if (M <= 0 || K <= 0 || N <= 0 || M > r4d_gemm_i8a8_nt_m16_max_m()) return 0;
    uint32_t ck = 0;
    const uint32_t k2 = narrow_ksplit((uint32_t)(split_n > 0 ? split_n : N), (uint32_t)K,
                                      (uint32_t)M, &ck);
    return k2 > 1 ? (long)k2 * M * N * (long)sizeof(float) : 0;
}

// C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T for M <= 64, int8 codes on both sides.
//
//   a       [M][K]                     int8, row-major, `a_ld` bytes a row (0 = K)
//   ascale  [M][K/group]               f32,  `sa_ld` elements a row (0 = K/group)
//   wq      [N][K]                     int8, fragment order (r4d_fp8_frag.h)
//   ws      [N/16][K/group][16]        bf16, tile-major
//   c       [M][N]                     bf16
//
// Returns 0, or a negative code for a shape it does not serve. `scratch` may be 0. `split_n` is
// m16_scratch's. `wide` selects the scale group: 256 (IQ8R, K a multiple of 256) over 128.
static int gemm_i8a8_nt_m16_emit(long a, long ascale, long wq, long ws, long c,
                                 int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                 long scratch, long scratch_bytes, RadStream stream, bool wide,
                                 int split_n, bool chain) {
    int emit_rc = RAD_OK;
    if (!a || !ascale || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    // K must be a multiple of 128; the staged tile is 256 bytes where K allows and 128 otherwise
    // (narrow_fk). The declared constraint must match this guard (cGemmNtI8A8M16 in r4d_rows.cpp):
    // the resolver offers a call by what a row claims, not by what the entry point accepts.
    if (K % 128) return -4;
    // A wave owns 16 weight rows and the transposing store writes `N`-strided columns.
    if (N % 16) return -5;
    if (!a_ld)  a_ld  = K;
    if (!w_ld)  w_ld  = K;
    const int grp = wide ? (int)R4D_I8_GROUP_WIDE : (int)kBlock;
    if (K % grp) return -4;
    const int nkb = (K + grp - 1) / grp;
    if (!sa_ld) sa_ld = nkb;
    if (!sw_ld) sw_ld = nkb;

    /* An int8 weight to 16 tokens: the K-parallel tile (its 16-byte stores want an aligned C). */
    /* Measured against the narrow tile at Flash-Next's per-rank shapes, weights cold (us, narrow
     * -> this): to 16 tokens 768 x 2560 8.5 -> 6.1, 512 x 2560 6.3 -> 4.4, 2560 x 3072 16.7 ->
     * 15.9, 8192 x 2560 38.0 -> 36.8; at 64 tokens 768 x 2560 10.9 -> 7.5, 2560 x 3072 22.8 ->
     * 18.5, 6144 x 2560 level, and 8192 x 2560 41.8 -> 42.8, so from 49 tokens a weight of 8192
     * rows and up keeps the narrow tile. Never slower elsewhere. A call under
     * r4d_gemm_i8a8_pin_split_waves keeps the narrow tile too: the pin asks for the narrow
     * tile's split, whose bytes the IQR narrow tile is compared with. */
    const bool split_pinned = g_split_waves_force.load(std::memory_order_relaxed) != 0u;
    if (!split_pinned && M <= 64 && (M <= 48 || N < 8192) &&
        ((uintptr_t)c & 15u) == 0u) {
        const uint32_t ngrp = (uint32_t)(K / grp);
        const uint32_t knt = ((uint32_t)M + 15u) / 16u;
        const dim3 kgrid((uint32_t)N / 16u);
#define R4D_I8K_E(kw_, sb_, nt_, ch_)                                                          \
        emit_rc = R4D_EMIT(stream, (gemm_i8_kpar_kernel<kw_, sb_, nt_, ch_>), kgrid,          \
                           dim3(kw_ * kWave), 0, (const uint8_t*)wq, (const uint16_t*)ws,      \
                           (const int8_t*)a, (const float*)ascale, (uint16_t*)c, (uint32_t)N,  \
                           (uint32_t)K, (uint32_t)M, (uint32_t)a_ld, (uint32_t)sa_ld)
#define R4D_I8K(kw_, sb_, nt_)                                                                 \
        do { if (chain) R4D_I8K_E(kw_, sb_, nt_, r4dchain);                                    \
             else       R4D_I8K_E(kw_, sb_, nt_, R4dUnchained); } while (0)
#define R4D_I8K_NT(kw_, sb_)                                                                   \
        do { if (knt == 1u)      R4D_I8K(kw_, sb_, 1u);                                        \
             else if (knt == 2u) R4D_I8K(kw_, sb_, 2u);                                        \
             else if (knt == 3u) R4D_I8K(kw_, sb_, 3u);                                        \
             else                R4D_I8K(kw_, sb_, 4u); } while (0)
#define R4D_I8K_KW(sb_)                                                                        \
        do { if (ngrp >= 4u)      R4D_I8K_NT(4u, sb_);                                         \
             else if (ngrp >= 2u) R4D_I8K_NT(2u, sb_);                                         \
             else                 R4D_I8K_NT(1u, sb_); } while (0)
        if (wide) R4D_I8K_KW(256u); else R4D_I8K_KW(128u);
#undef R4D_I8K_KW
#undef R4D_I8K_NT
#undef R4D_I8K
#undef R4D_I8K_E
        if (emit_rc != RAD_OK || hipGetLastError() != hipSuccess) return -6;
        return 0;
    }

    const uint32_t mw = narrow_mw((uint32_t)N, (uint32_t)M);
    const uint32_t fm = mw * 16u;   // the narrow tile is one M tile a wave
    const uint32_t gx = ((uint32_t)N + fm - 1) / fm;

    uint32_t ck = 0;
    uint32_t ksplit = narrow_ksplit((uint32_t)(split_n > 0 ? split_n : N), (uint32_t)K,
                                    (uint32_t)M, &ck);
    const long need = ksplit > 1 ? (long)ksplit * M * N * (long)sizeof(float) : 0;
    if (need && (!scratch || scratch_bytes < need)) { ksplit = 1; ck = 0; }

    const dim3 grid(gx, 1, ksplit);
    void* const dst = ksplit > 1 ? (void*)scratch : (void*)c;
    /* Token tiles per wave: the activation is the B operand, so NT tiles cover NT*16 tokens with
     * one weight read. NT = ceil(M/16), including 3 (see the note above narrow_mw). */
    uint32_t nt = (M > 48) ? 4u : ((M > 32) ? 3u : ((M > 16) ? 2u : 1u));
    /* `Full` drops the column guard, so it must mean all NT tiles are live, not just M % 16 == 0.
     * Otherwise (e.g. M = 48, NT = 4) the epilogue writes past a split-K scratch sized for M, over
     * the arrival counter, and the finisher hangs. */
    const bool full = ((uint32_t)N % fm == 0) && ((uint32_t)M % (nt * 16u) == 0) &&
                      (K % (int)kBlock == 0);
    // The finisher needs one counter per column; beyond kMaxCtr or on allocation failure, the
    // separate reduce kernel computes the same bytes.
    unsigned int* const ctr =
        (ksplit > 1 && gx <= kMaxCtr) ? split_ctr(stream, (const void*)c) : nullptr;

#define R4D_NARROW(f, mw_, nt_, fk_, sb_)                                                      \
    emit_rc = R4D_EMIT(stream,                                                                 \
                       (chain ? r4dchain_gemm_i8a8_narrow<f, nt_, mw_, fk_, sb_>               \
                              : gemm_i8a8_narrow_kernel<f, nt_, mw_, fk_, sb_>),               \
                       grid, dim3(mw_ * kWave),                                                \
                       0, (const uint8_t*)wq, (const void*)ws,                                 \
                       (const uint8_t*)a, (const void*)ascale, dst, (void*)c, ctr,              \
                       (uint32_t)N, (uint32_t)K,                                                \
                       (uint32_t)M, (uint32_t)w_ld, (uint32_t)a_ld, (uint32_t)sw_ld,            \
                       (uint32_t)sa_ld, ck)
#define R4D_NARROW_NT(f, mw_, fk_, sb_)                                                        \
    do { if (nt == 4u)      R4D_NARROW(f, mw_, 4u, fk_, sb_);                                  \
         else if (nt == 3u) R4D_NARROW(f, mw_, 3u, fk_, sb_);                                  \
         else if (nt == 2u) R4D_NARROW(f, mw_, 2u, fk_, sb_);                                  \
         else               R4D_NARROW(f, mw_, 1u, fk_, sb_); } while (0)
#define R4D_NARROW_MW(fk_, sb_)                                                                \
    do { if (full) {                                                                           \
             if (mw == 4u)      R4D_NARROW_NT(true, 4u, fk_, sb_);                             \
             else if (mw == 8u) R4D_NARROW_NT(true, 8u, fk_, sb_);                             \
             else               R4D_NARROW_NT(true, 16u, fk_, sb_);                            \
         } else {                                                                              \
             if (mw == 4u)      R4D_NARROW_NT(false, 4u, fk_, sb_);                            \
             else if (mw == 8u) R4D_NARROW_NT(false, 8u, fk_, sb_);                            \
             else               R4D_NARROW_NT(false, 16u, fk_, sb_);                           \
         } } while (0)
    if (narrow_fk((uint32_t)K) == kFKNarrow) {
        if (wide) R4D_NARROW_MW(kFKNarrow, 256u);
        else      R4D_NARROW_MW(kFKNarrow, kBlock);
    } else {
        if (wide) return -4;   /* K % 256 was checked above; unreachable */
        R4D_NARROW_MW(128u, kBlock);
    }
#undef R4D_NARROW_MW
#undef R4D_NARROW_NT
#undef R4D_NARROW
    if (emit_rc != RAD_OK) return -6;
    if (hipGetLastError() != hipSuccess) return -6;

    if (ksplit > 1 && !ctr) {
        const uint32_t n = (uint32_t)M * (uint32_t)N;
        if (R4D_EMIT(stream, splitk_reduce_kernel, dim3((n + 255) / 256), dim3(256), 0,
                     (const float*)scratch, (uint16_t*)c, n, ksplit) != RAD_OK) return -7;
    }
    return 0;
}

// The same GEMM for prefill row counts: same operands and layouts, the 128x128 prefill tile
// (gemm_8a8_prefill_kernel). No split-K: the grid is ceil(M/128) x N/128, hundreds of blocks at any
// prefill shape.
template <int RB = 0, uint32_t SB = kBlock>
static int gemm_8a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int a_ld, int sa_ld, int w_ld, int sw_ld, long stream);

// The int8 form: i8 codes for both operands and a bf16 weight scale per row per 128 K (ws is
// [N][ceil(K/128)]).
extern "C" int r4d_gemm_i8a8_tiled(long a, long ascale, long wq, long ws, long c,
                                   int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                   long stream) {
    return gemm_8a8_tiled<>(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, stream);
}

// The same with the weight's scale one per 8 rows and 128 K (each row of the plane holding its
// block's value), folded as a block scale (gemm_8a8_prefill_kernel's RB).
extern "C" int r4d_gemm_i8a8_rb8_tiled(long a, long ascale, long wq, long ws, long c,
                                       int M, int K, int N, int a_ld, int sa_ld, int w_ld,
                                       int sw_ld, long stream) {
    return gemm_8a8_tiled<8>(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, stream);
}

// The same at a weight scale per 32 rows, a wave's: one scale per wave.
extern "C" int r4d_gemm_i8a8_rb32_tiled(long a, long ascale, long wq, long ws, long c,
                                        int M, int K, int N, int a_ld, int sa_ld, int w_ld,
                                        int sw_ld, long stream) {
    return gemm_8a8_tiled<32>(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, stream);
}

template <int RB, uint32_t SB>
static int gemm_8a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int a_ld, int sa_ld, int w_ld, int sw_ld, long stream) {
    if (!a || !ascale || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    // Required: the fold assumes whole SB-column scale blocks, and a block's 128 weight rows are
    // one scale row.
    if (K % (int)SB) return -4;
    if (N % (int)kPBN) return -5;
    // C is written 16 bytes at a time.
    if ((uintptr_t)c & 15u) return -3;
    if (!a_ld)  a_ld  = K;
    (void)w_ld;   // the fragment-order plane has no row pitch
    const int nkb = (K + (int)SB - 1) / (int)SB;
    if (!sa_ld) sa_ld = nkb;
    if (!sw_ld) sw_ld = nkb;

    const dim3 grid(((uint32_t)M + kPBM - 1) / kPBM, (uint32_t)N / kPBN, 1);
    const int rc = R4D_EMIT((RadStream)(uintptr_t)stream, (gemm_8a8_prefill_kernel<RB, SB>), grid,
                            dim3(kPWaves * kWave), 0, (const uint8_t*)a, (const float*)ascale,
                            (const uint8_t*)wq, (const uint16_t*)ws, (uint16_t*)c, (uint32_t)M,
                            (uint32_t)K, (uint32_t)N, (uint32_t)a_ld, (uint32_t)sa_ld,
                            (uint32_t)sw_ld);
    return rc == RAD_OK && hipGetLastError() == hipSuccess ? 0 : -6;
}

/* The operands of an int8 linear, with the activation's dtype checked. */
static int i8_gemm_ops_checked(const RadArgs* a, I8GemmOps* g) {
    const int rc = i8_gemm_ops(a, g);
    if (rc != RAD_OK) return rc;
    if (g->a->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    return RAD_OK;
}

extern "C" int64_t r4d_rad_gemm_i8a8_m16_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), N = r4d_geti_or(a, "N", 0),
                    K = r4d_geti_or(a, "K", 0), sn = r4d_geti_or(a, "split_n", 0);
    if (M < 1 || N < 1 || K < 1) return 0;
    if (M > r4d_gemm_i8a8_nt_m16_max_m()) return 0;
    return (int64_t)m16_scratch((int)M, (int)K, (int)N, (int)sn);
}

extern "C" int r4d_rad_gemm_i8a8_m16(const RadArgs* a, RadStream stream) {
    I8GemmOps g;
    const int rc = i8_gemm_ops(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.M > r4d_gemm_i8a8_nt_m16_max_m()) return R4D_NO(RAD_E_SHAPE);
    if (g.a->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    const int e = gemm_i8a8_nt_m16_emit(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)a->scratch, (long)a->scratch_bytes, stream,
        g.group == R4D_I8_GROUP_WIDE,
        (int)r4d_geti_or(a, "split_n", 0), r4d_geti_or(a, "chain", 0) != 0);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

/* gemm_nt_q_multi (oGemmNtQMulti in r4d_rows.cpp): up to five int8 linears at one M and K,
 * slice i reading input src<i> (operands 2 src, 2 src + 1) with weight 6 + 2 i, its scale 7 + 2 i,
 * into y 16 + i, in one K-parallel launch. Each slice is checked as gemm_nt_q's own operands. */
enum { QM_A = 0, QM_B = 6, QM_Y = 16, QM_NIN = 3 };

extern "C" int r4d_rad_gemm_i8a8_multi(const RadArgs* a, RadStream stream) {
    static const char* const kN[kI8MultiMax]   = { "N0", "N1", "N2", "N3", "N4" };
    static const char* const kSrc[kI8MultiMax] = { "src0", "src1", "src2", "src3", "src4" };
    if (!a || !a->t) return R4D_NO(RAD_E_INVAL);
    if (r4d_geti_or(a, "group", 0) != (long long)kBlock) return R4D_NO(RAD_E_SHAPE);
    I8KparSlices S{};
    long long M = -1, K = -1;
    uint32_t blocks = 0;
    for (uint32_t i = 0; i < kI8MultiMax; ++i) {
        const long long Ni = r4d_geti_or(a, kN[i], 0);
        if (Ni <= 0) break;
        const long long src = r4d_geti_or(a, kSrc[i], 0);
        if (src < 0 || src >= QM_NIN) return R4D_NO(RAD_E_SHAPE);
        const int at[5] = { QM_A + 2 * (int)src, QM_A + 2 * (int)src + 1, QM_B + 2 * (int)i,
                            QM_B + 2 * (int)i + 1, QM_Y + (int)i };
        RadTensor t[5];
        for (int k = 0; k < 5; ++k) {
            const RadTensor* x = r4d_opt(a, at[k]);
            if (!x) return R4D_NO(RAD_E_INVAL);
            t[k] = *x;
        }
        RadArgs sub = *a;
        sub.t = t;
        sub.n_t = 5;
        I8GemmOps g;
        const int rc = i8_gemm_ops_checked(&sub, &g);
        if (rc != RAD_OK) return rc;
        if (g.N != Ni || g.N % 16 || g.K % (long long)kBlock) return R4D_NO(RAD_E_SHAPE);
        if ((M >= 0 && g.M != M) || (K >= 0 && g.K != K)) return R4D_NO(RAD_E_SHAPE);
        if (((uintptr_t)g.y->data) & 15u) return R4D_NO(RAD_E_ALIGN);
        M = g.M;
        K = g.K;
        S.w[i]  = (const uint8_t*)g.b->data;
        S.ws[i] = (const uint16_t*)g.bscale->data;
        S.a[i]  = (const int8_t*)g.a->data;
        S.as[i] = (const float*)g.ascale->data;
        S.c[i]  = (uint16_t*)g.y->data;
        S.rows[i]  = (uint32_t)g.N;
        S.a_ld[i]  = (uint32_t)g.a_ld;
        S.sa_ld[i] = (uint32_t)g.sa_ld;
        blocks += (uint32_t)(g.N / 16);
        S.end[i] = blocks;
        S.n = i + 1;
    }
    if (S.n == 0) return R4D_NO(RAD_E_SHAPE);
    if (M < 1 || M > 16) return R4D_NO(RAD_E_SHAPE);
    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    const uint32_t ngrp = (uint32_t)(K / (long long)kBlock);
    int rc = RAD_OK;
#define R4D_I8M_E(kw_, ch_)                                                                    \
    rc = R4D_EMIT(stream, (gemm_i8_kpar_multi_kernel<kw_, kBlock, 1u, ch_>), dim3(blocks),    \
                  dim3(kw_ * kWave), 0, S, (uint32_t)K, (uint32_t)M)
#define R4D_I8M(kw_)                                                                           \
    do { if (chain) R4D_I8M_E(kw_, r4dchain); else R4D_I8M_E(kw_, R4dUnchained); } while (0)
    if (ngrp >= 4u)      R4D_I8M(4u);
    else if (ngrp >= 2u) R4D_I8M(2u);
    else                 R4D_I8M(1u);
#undef R4D_I8M
#undef R4D_I8M_E
    return rc == RAD_OK && hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_gemm_i8a8_tiled(const RadArgs* a, RadStream s) {
    I8GemmOps g;
    const int rc = i8_gemm_ops_checked(a, &g);
    if (rc != RAD_OK) return rc;
    const int e = g.group == R4D_I8_GROUP_WIDE
        ? gemm_8a8_tiled<0, 256u>(
              r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
              (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
              (long)(uintptr_t)s)
        : r4d_gemm_i8a8_tiled(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

/* Vision's int8 linear: the prefill tile over container-form weights. `w_rb` 1 is how the
 * architecture names it; the fragment rows do not read the parameter, so they stay the text path. */
extern "C" int r4d_rad_gemm_i8a8_row(const RadArgs* a, RadStream s) {
    I8GemmOps g;
    const int rc = i8_gemm_ops(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.a->dtype != RAD_I8 || g.b->dtype != RAD_I8 || g.bscale->dtype != RAD_BF16 ||
        g.ascale->dtype != RAD_F32 || g.y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (r4d_geti_or(a, "w_rb", 0) != 1 || g.group != R4D_I8_GROUP) return R4D_NO(RAD_E_SHAPE);
    if (g.K % (int)kBlock || g.N % (int)kPBN) return R4D_NO(RAD_E_SHAPE);
    if (((uintptr_t)g.y->data) & 15u) return R4D_NO(RAD_E_ALIGN);
    const int e = R4D_EMIT(s, gemm_8a8_prefill_row_kernel,
                           dim3(((uint32_t)g.M + kPBM - 1) / kPBM, (uint32_t)g.N / kPBN, 1),
                           dim3(kPWaves * kWave), 0, (const uint8_t*)r4d_addr(g.a),
                           (const float*)r4d_addr(g.ascale), (const uint8_t*)r4d_addr(g.b),
                           (const uint16_t*)r4d_addr(g.bscale), (uint16_t*)r4d_addr(g.y),
                           (uint32_t)g.M, (uint32_t)g.K, (uint32_t)g.N, (uint32_t)g.a_ld,
                           (uint32_t)g.sa_ld, (uint32_t)g.w_ld, (uint32_t)g.sw_ld);
    return e == RAD_OK && hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_gemm_i8a8_rb8_tiled(const RadArgs* a, RadStream s) {
    I8GemmOps g;
    const int rc = i8_gemm_ops_checked(a, &g);
    if (rc != RAD_OK) return rc;
    const int e = r4d_gemm_i8a8_rb8_tiled(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_gemm_i8a8_rb32_tiled(const RadArgs* a, RadStream s) {
    I8GemmOps g;
    const int rc = i8_gemm_ops_checked(a, &g);
    if (rc != RAD_OK) return rc;
    const int e = r4d_gemm_i8a8_rb32_tiled(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

